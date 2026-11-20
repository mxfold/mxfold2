from __future__ import annotations

import csv
import logging
import math
import os
import random
import time
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, cast, Callable

import pytorch_optimizer as po

# import numpy as np
import torch
import torch.backends.cudnn
import torch.nn as nn

# import torch.nn.functional as F
import torch.optim as optim
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW
from torch.optim.rmsprop import RMSprop
from torch.optim.sgd import SGD
from torch.optim.asgd import ASGD
from torch.optim.optimizer import Optimizer
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from torch.optim.swa_utils import SWALR, AveragedModel
from torch.utils.data import DataLoader, ConcatDataset, RandomSampler, WeightedRandomSampler
from tqdm import tqdm

from mxfold2.ema import EMA
from mxfold2.sam import SAM, ASAM, GSAM

from mxfold2 import interface  # type: ignore[attr-defined]
from mxfold2.dataset import (
    BPseqDataset,
    FastaDataset,
    ShapeDataset,
    RibonanzaDataset,
    JsonDataset,
    JsonShapeDataset,
    MultiTaskDataset,
)
from mxfold2.compbpseq import compare_bpseq, accuracy
from mxfold2.compreactivity import read_shape_reactivity, pairwise_consistency
from mxfold2.nucleosides import get_modified_positions
from mxfold2.fold.fold import AbstractFold
from mxfold2.common import Common

try:
    import wandb
except ImportError:
    wandb = None


def read_shape_list(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip().split()[0] for line in f if line.strip()]


class Train(Common):
    step: int = 0
    disable_progress_bar: bool = False
    use_wandb: bool = False
    gpu: int = -1

    def __init__(self):
        super(Train, self).__init__()
        self.device_type = 'cpu'  # Will be set in run()
        self.imle_log_file: Optional[str] = None
        self.log_file: Optional[str] = None
        self._frozen_eval_modules: list[nn.Module] = []
        self._l2_sp_anchor: dict[str, torch.Tensor] = {}
        self._l2_sp_weight = 0.0

    def train(self, epoch: int, model: AbstractFold, optimizer: Optimizer,
                loss_fn: nn.Module | dict[str, nn.Module],
                data_loader: DataLoader[tuple[str, str, dict[str, torch.Tensor]]],
                n_dataset: Optional[int] = None,
                loss_weight = defaultdict(lambda: 1.),
                clip_grad_value: float = 0.0, clip_grad_norm: float = 0.0,
                scaler: Optional[GradScaler] = None,
                use_amp: bool = False,
                grad_accum_steps: int = 1) -> None:
        if grad_accum_steps < 1:
            raise ValueError("grad_accum_steps must be at least 1")
        model.train()
        # model.train() recursively re-enables dropout in frozen feature
        # extractors.  Keep explicitly frozen modules in eval mode while the
        # trainable heads remain in training mode.
        for module in self._frozen_eval_modules:
            module.eval()
        if not isinstance(loss_fn, dict):
            loss_fn = {'BPSEQ': loss_fn}
        # Replay/budget samplers can draw fewer items than the backing dataset.
        # Accumulation windows and the final partial update use actual draws.
        progress_total = len(data_loader.sampler) if n_dataset is None else n_dataset
        n_dataset = len(data_loader.sampler)
        loss_total, num = 0., 0
        running_loss, n_running_loss = 0, 0
        imle_rows = []
        start = time.time()

        # Check if using SAM optimizer
        is_sam = isinstance(optimizer, SAM)

        # Gradient accumulation counter
        accumulated_samples = 0
        optimizer.zero_grad()  # Zero gradients once at the start

        with tqdm(total=progress_total, disable=self.disable_progress_bar) as pbar:
            for fnames, seqs, vals in data_loader:
                logging.info(f"Step: {self.step}, {fnames}")
                self.step += 1
                n_batch = len(seqs)
                for i in range(n_batch):
                    accumulated_samples += 1
                    position = (accumulated_samples - 1) % grad_accum_steps
                    window_start = accumulated_samples - 1 - position
                    accumulation_divisor = min(grad_accum_steps, n_dataset - window_start)
                    is_update_step = position + 1 == accumulation_divisor

                    # Define loss computation function for SAM
                    def compute_loss():
                        with autocast(device_type=self.device_type, dtype=torch.float16, enabled=use_amp):
                            if vals['type'][i] in ('BPSEQ', 'BPSEQ2'):
                                loss = torch.sum(loss_fn['BPSEQ'](seqs[i:i+1], vals['target'][i:i+1], fname=fnames[i:i+1]))
                            elif vals['type'][i] == 'SHAPE':
                                loss = torch.sum(loss_fn['SHAPE'](seqs[i:i+1], vals['target'][i:i+1], fname=fnames[i:i+1], dataset_id=vals['dataset_id'][i:i+1]))
                            elif vals['type'][i] == 'MULTI':
                                struct_loss = loss_fn['BPSEQ'](
                                    seqs[i:i+1], vals['target'][i:i+1], fname=fnames[i:i+1]
                                )
                                shape_loss = loss_fn['SHAPE_regress'](
                                    seqs[i:i+1],
                                    vals['shape_target'][i:i+1],
                                    vals['shape_mask'][i:i+1],
                                )
                                alpha, beta = loss_weight['MULTI_intra']
                                loss = torch.sum(alpha * struct_loss + beta * shape_loss)
                            else:
                                raise RuntimeError('not implemented')
                            if self._l2_sp_weight > 0.0:
                                penalty = None
                                for name, parameter in model.named_parameters():
                                    anchor = self._l2_sp_anchor.get(name)
                                    if anchor is None or not parameter.requires_grad:
                                        continue
                                    term = torch.sum((parameter - anchor) ** 2)
                                    penalty = term if penalty is None else penalty + term
                                if penalty is None:
                                    raise RuntimeError('L2-SP has no trainable anchored parameters')
                                loss = loss + 0.5 * self._l2_sp_weight * penalty
                            sample_weight = vals['weight'][i].item() if 'weight' in vals else 1.0
                            return loss * loss_weight[vals['type'][i]] * sample_weight

                    if is_sam:
                        # SAM two-step optimization with gradient accumulation
                        loss = self._sam_step(
                            model, optimizer, compute_loss,
                            clip_grad_value, clip_grad_norm,
                            scaler, use_amp,
                            grad_accum_steps=accumulation_divisor,
                            is_update_step=is_update_step
                        )
                        if is_update_step:
                            optimizer.zero_grad()
                    else:
                        # Standard optimization with gradient accumulation
                        loss = compute_loss()

                        # Scale loss for gradient accumulation and backward pass
                        scaled_loss = loss / accumulation_divisor
                        if scaler is not None:
                            scaler.scale(scaled_loss).backward()
                        else:
                            scaled_loss.backward()

                        # Only update on accumulation boundary
                        if is_update_step:
                            # Gradient clipping with unscaling if using mixed precision
                            if scaler is not None:
                                scaler.unscale_(optimizer)

                            if clip_grad_norm > 0.0:
                                nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm, norm_type=2)
                            elif clip_grad_value > 0.0:
                                nn.utils.clip_grad_value_(model.parameters(), clip_value=clip_grad_value)

                            # Workaround for pytorch_optimizer AdaBelief/Lion: ensure state is initialized
                            # for parameters that didn't have gradients in the first step
                            if isinstance(optimizer, (po.AdaBelief, po.Lion)):
                                for group in optimizer.param_groups:
                                    for p in group['params']:
                                        if p.grad is not None and len(optimizer.state[p]) == 0:
                                            optimizer.state[p]['exp_avg'] = torch.zeros_like(p)
                                            if isinstance(optimizer, po.AdaBelief):
                                                optimizer.state[p]['exp_avg_var'] = torch.zeros_like(p)

                            # Step the optimizer
                            if scaler is not None:
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                optimizer.step()

                            optimizer.zero_grad()

                    loss_total += loss.item()
                    running_loss += loss.item()
                    if vals['type'][i] == 'SHAPE':
                        stats = getattr(loss_fn['SHAPE'], 'last_imle_stats', {})
                        nus = stats.get('selected_nu', [])
                        flips = stats.get('flip_fraction', [])
                        if nus and flips:
                            imle_rows.append([
                                epoch,
                                self.step,
                                fnames[i],
                                getattr(loss_fn['SHAPE'], 'difference_mode', 'forward'),
                                len(nus),
                                sum(nus) / len(nus),
                                min(nus),
                                max(nus),
                                sum(flips) / len(flips),
                                min(flips),
                                max(flips),
                            ])

                num += n_batch
                pbar.set_postfix(train_loss='{:.3e}'.format(loss_total / num))
                pbar.update(n_batch)

                n_running_loss += n_batch
                if n_running_loss >= 100 or num >= n_dataset:
                    running_loss /= n_running_loss
                    if self.use_wandb:
                        log_data = {
                            "train/loss": running_loss,
                            "train/step": (epoch-1) * n_dataset + num,
                        }
                        if self.device_type == 'cuda':
                            log_data["train/gpu_memory_allocated"] = torch.cuda.memory_allocated(self.gpu) / 1024**3
                            log_data["train/gpu_memory_reserved"] = torch.cuda.memory_reserved(self.gpu) / 1024**3
                        # MPS does not provide memory stats via PyTorch API
                        wandb.log(log_data)
                    running_loss, n_running_loss = 0, 0

        elapsed_time = time.time() - start
        avg_loss = loss_total / num
        log_line = f"Train Epoch: {epoch}\tLoss: {avg_loss:.6f}\tTime: {elapsed_time:.3f}s"
        print(log_line)
        # エポック単位のログは loss.log に一行だけ書く（alloc のみ記録、resv/peak は 0.0 で埋める）
        try:
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / 1e6
            else:
                alloc = 0.0
            if self.log_file is not None:
                with open(self.log_file, "a") as f:
                    f.write(f"{epoch},train,{avg_loss:.6f},{elapsed_time:.3f},{self.step},{alloc:.1f}\n")
        except Exception:
            pass
        if self.imle_log_file is not None and imle_rows:
            with open(self.imle_log_file, 'a', newline='') as f:
                csv.writer(f).writerows(imle_rows)

    def _sam_step(
        self,
        model: AbstractFold,
        optimizer: SAM,
        compute_loss: Callable[[], torch.Tensor],
        clip_grad_value: float,
        clip_grad_norm: float,
        scaler: Optional[GradScaler],
        use_amp: bool,
        grad_accum_steps: int = 1,
        is_update_step: bool = True,
    ) -> torch.Tensor:
        """Perform SAM two-step optimization with AMP and gradient accumulation support.

        Args:
            model: The model being trained
            optimizer: SAM optimizer instance
            compute_loss: Callable that computes and returns the loss
            clip_grad_value: Gradient clipping by value
            clip_grad_norm: Gradient clipping by norm
            scaler: GradScaler for AMP (optional)
            use_amp: Whether AMP is enabled
            grad_accum_steps: Number of gradient accumulation steps
            is_update_step: Whether this is an update step (vs accumulation step)

        Returns:
            The loss value from the first forward pass
        """
        is_gsam = isinstance(optimizer, GSAM)

        # === First forward-backward pass ===
        loss = compute_loss()
        scaled_loss = loss / grad_accum_steps

        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        # Only perform SAM two-step update on update steps
        if is_update_step:
            if scaler is not None:
                scaler.unscale_(optimizer)

            # Apply gradient clipping before perturbation
            if clip_grad_norm > 0.0:
                nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=clip_grad_norm, norm_type=2
                )
            elif clip_grad_value > 0.0:
                nn.utils.clip_grad_value_(
                    model.parameters(), clip_value=clip_grad_value
                )

            # Apply perturbation (first step)
            if is_gsam:
                optimizer.first_step(zero_grad=True, loss=loss) # type: ignore
            else:
                optimizer.first_step(zero_grad=True)

            # === Second forward-backward pass (at perturbed weights) ===
            loss_perturbed = compute_loss()
            scaled_loss_perturbed = loss_perturbed / grad_accum_steps

            if scaler is not None:
                scaler.scale(scaled_loss_perturbed).backward()
                scaler.unscale_(optimizer)
            else:
                scaled_loss_perturbed.backward()

            # Apply gradient clipping before update
            if clip_grad_norm > 0.0:
                nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=clip_grad_norm, norm_type=2
                )
            elif clip_grad_value > 0.0:
                nn.utils.clip_grad_value_(
                    model.parameters(), clip_value=clip_grad_value
                )

            # Restore weights and apply update (second step)
            if is_gsam:
                optimizer.second_step(zero_grad=False, loss=loss_perturbed) # type: ignore
            else:
                optimizer.second_step(zero_grad=False)

            # Update scaler if using AMP
            if scaler is not None:
                scaler.update()

        return loss

    def test(
        self,
        epoch: int,
        model: AbstractFold | AveragedModel,
        loss_fn: nn.Module | dict[str, nn.Module],
        data_loader: DataLoader[tuple[str, str, dict[str, torch.Tensor]]],
        use_amp: bool = False,
        shape_paths: list[str] | None = None,
        metric_prefix: str = "test",
    ) -> None:
        model.eval()
        if not isinstance(loss_fn, dict):
            loss_fn = {"BPSEQ": loss_fn}
        n_dataset = len(cast(FastaDataset, data_loader.dataset))
        loss_total, num = 0, 0
        all_sen: list[float] = []
        all_ppv: list[float] = []
        all_fval: list[float] = []
        mod_all_sen: list[float] = []
        mod_all_ppv: list[float] = []
        mod_all_fval: list[float] = []
        shape_scores: list[float] = []
        sample_idx = 0
        start = time.time()
        with (
            torch.no_grad(),
            tqdm(total=n_dataset, disable=self.disable_progress_bar) as pbar,
        ):
            for fnames, seqs, vals in data_loader:
                n_batch = len(seqs)
                for i in range(n_batch):
                    with autocast(
                        device_type=self.device_type,
                        dtype=torch.float16,
                        enabled=use_amp,
                    ):
                        if vals["type"][i] == "BPSEQ":
                            loss = torch.sum(
                                loss_fn["BPSEQ"](  # type: ignore[operator]
                                    seqs[i : i + 1],
                                    vals["target"][i : i + 1],
                                    fname=fnames[i : i + 1],
                                )
                            )
                            # Get predictions for F1 calculation
                            _, _, bps = model(seqs[i : i + 1])
                            ref = vals["target"][i]
                            tp, tn, fp, fn = compare_bpseq(ref, bps[0])
                            sen, ppv, fval, _, _, _ = accuracy(tp, tn, fp, fn)
                            all_sen.append(sen)
                            all_ppv.append(ppv)
                            all_fval.append(fval)
                            # Modified-base-only metrics
                            mod_positions = get_modified_positions(seqs[i])
                            if mod_positions:
                                mod_tp, mod_tn, mod_fp, mod_fn = compare_bpseq(
                                    ref, bps[0], bases=list(seqs[i]), modified_only=True,
                                )
                                if mod_tp + mod_fp + mod_fn > 0:
                                    m_sen, m_ppv, m_fval, _, _, _ = accuracy(
                                        mod_tp, mod_tn, mod_fp, mod_fn
                                    )
                                    mod_all_sen.append(m_sen)
                                    mod_all_ppv.append(m_ppv)
                                    mod_all_fval.append(m_fval)
                            # SHAPE consistency
                            if shape_paths is not None and sample_idx < len(
                                shape_paths
                            ):
                                shape_data = read_shape_reactivity(
                                    shape_paths[sample_idx]
                                )
                                score, _, _ = pairwise_consistency(bps[0], shape_data)
                                if not math.isnan(score):
                                    shape_scores.append(score)
                            sample_idx += 1
                        elif vals["type"][i] == "SHAPE":
                            # I-MLE evaluates an objective gradient even during validation.
                            with torch.enable_grad():
                                loss = torch.sum(
                                    loss_fn["SHAPE"](
                                        seqs[i : i + 1],
                                        vals["target"][i : i + 1],
                                        fname=fnames[i : i + 1],
                                        dataset_id=vals["dataset_id"][i : i + 1],
                                    )
                                )
                        else:
                            raise RuntimeError("not implemented")
                    loss_total += loss.item()
                num += n_batch
                pbar.set_postfix(test_loss="{:.3e}".format(loss_total / num))
                pbar.update(n_batch)

        elapsed_time = time.time() - start

        # Calculate accuracy metrics
        sen = ppv = fval = 0.0
        has_metrics = len(all_sen) > 0
        if has_metrics:
            sen = sum(all_sen) / len(all_sen)
            ppv = sum(all_ppv) / len(all_ppv)
            fval = sum(all_fval) / len(all_fval)

        # Modified-base-only metrics
        mod_f1 = mod_sen = mod_ppv = 0.0
        has_mod_metrics = len(mod_all_fval) > 0
        if has_mod_metrics:
            mod_sen = sum(mod_all_sen) / len(mod_all_sen)
            mod_ppv = sum(mod_all_ppv) / len(mod_all_ppv)
            mod_f1 = sum(mod_all_fval) / len(mod_all_fval)

        if self.use_wandb:
            log_dict: dict[str, float] = {
                f"{metric_prefix}/loss": loss_total / num,
                f"{metric_prefix}/epoch": epoch,
            }
            if has_metrics:
                log_dict.update(
                    {
                        f"{metric_prefix}/f1": fval,
                        f"{metric_prefix}/sensitivity": sen,
                        f"{metric_prefix}/ppv": ppv,
                    }
                )
            if has_mod_metrics:
                log_dict.update(
                    {
                        f"{metric_prefix}/mod_f1": mod_f1,
                        f"{metric_prefix}/mod_sensitivity": mod_sen,
                        f"{metric_prefix}/mod_ppv": mod_ppv,
                    }
                )
            if shape_scores:
                log_dict[f"{metric_prefix}/shape_consistency"] = sum(
                    shape_scores
                ) / len(shape_scores)
            wandb.log(log_dict)

        msg = "Test[{}] Epoch: {}\tLoss: {:.6f}".format(
            metric_prefix, epoch, loss_total / num
        )
        if has_metrics:
            msg += "\tF1: {:.4f}\tSEN: {:.4f}\tPPV: {:.4f}".format(
                fval, sen, ppv
            )
        if has_mod_metrics:
            msg += "\tmod_F1: {:.4f}\tmod_SEN: {:.4f}\tmod_PPV: {:.4f}".format(
                mod_f1, mod_sen, mod_ppv
            )
        if shape_scores:
            msg += "\tSHAPE_consistency: {:.4f}".format(
                sum(shape_scores) / len(shape_scores)
            )
        msg += "\tTime: {:.3f}s".format(elapsed_time)
        print(msg)

    def save_checkpoint(
        self,
        outdir: str,
        epoch: int,
        model: AbstractFold | AveragedModel,
        optimizer: Optimizer,
        scheduler,
        shape_model: Optional[list[nn.Module]] = None,
        ema: Optional[EMA] = None,
        step: Optional[int] = None,
        scaler: Optional[GradScaler] = None,
    ) -> None:
        filename = os.path.join(outdir, "epoch-{}".format(epoch))
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step if step is not None else self.step,
            "rng_state": {
                "torch": torch.get_rng_state(),
                "python": random.getstate(),
            },
        }
        if torch.cuda.is_available():
            checkpoint["rng_state"]["cuda"] = torch.cuda.get_rng_state_all()
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        if shape_model is not None:
            checkpoint["shape_model_state_dict"] = [
                sm.state_dict() for sm in shape_model
            ]
        if ema is not None:
            checkpoint["ema_state_dict"] = ema.state_dict()
        if scaler is not None:
            checkpoint["scaler_state_dict"] = scaler.state_dict()
        if self.use_wandb and wandb.run is not None:
            checkpoint["wandb_run_id"] = wandb.run.id
        torch.save(checkpoint, filename)

    def resume_checkpoint(
        self,
        filename: str | dict[str, Any],
        model: AbstractFold,
        optimizer: Optimizer,
        scheduler,
        shape_model: Optional[list[nn.Module]] = None,
        ema: Optional[EMA] = None,
        scaler: Optional[GradScaler] = None,
        gpu: int = -1,
    ) -> tuple[int, Optional[GradScaler], Optional[str]]:
        checkpoint = (
            filename if isinstance(filename, dict) else torch.load(filename, map_location="cpu")
        )
        epoch = checkpoint["epoch"]
        self.load_model_weights(model, checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Restore step counter
        if "step" in checkpoint:
            self.step = checkpoint["step"]

        # Restore random states
        if "rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["rng_state"]["torch"])
            random.setstate(checkpoint["rng_state"]["python"])
            if "cuda" in checkpoint["rng_state"] and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(checkpoint["rng_state"]["cuda"])

        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if shape_model is not None and "shape_model_state_dict" in checkpoint:
            for i, sm in enumerate(shape_model):
                sm.load_state_dict(checkpoint["shape_model_state_dict"][i])
        if ema is not None and "ema_state_dict" in checkpoint:
            ema.load_state_dict(checkpoint["ema_state_dict"])

        # Load GradScaler state if available
        if "scaler_state_dict" in checkpoint and scaler is None:
            _, device_type = self.get_device(gpu)
            if device_type == "cuda":
                device_str = f"cuda:{gpu}"
            else:
                device_str = device_type
            scaler = GradScaler(device_str)
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        elif scaler is not None and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        wandb_run_id = checkpoint.get("wandb_run_id", None)

        return epoch, scaler, wandb_run_id

    def build_optimizer(self, optimizer: str, model: AbstractFold, lr: float, l2_weight: float,
                        shape_model: Optional[list[nn.Module]] = None,
                        sam_type: Optional[str] = None,
                        sam_rho: float = 0.05,
                        sam_alpha: float = 0.1,
                        lr_overrides: Optional[dict[str, Optional[float]]] = None) -> Optimizer:
        lr_overrides = lr_overrides or {}
        if any(value is not None for value in lr_overrides.values()):
            grouped: dict[str, list[nn.Parameter]] = {
                'embedding': [], 'encoder': [], 'fc_paired': [], 'other': []
            }
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                if '.embedding.' in name:
                    key = 'embedding'
                elif '.encoder.' in name:
                    key = 'encoder'
                elif '.fc_paired.' in name:
                    key = 'fc_paired'
                else:
                    key = 'other'
                grouped[key].append(parameter)
            optim_params = []
            for key in ('embedding', 'encoder', 'fc_paired', 'other'):
                if not grouped[key]:
                    continue
                group_lr = lr_overrides.get(key)
                optim_params.append({
                    'params': grouped[key],
                    'lr': lr if group_lr is None else group_lr,
                    'weight_decay': l2_weight,
                })
        else:
            optim_params = [
                {'params': (p for p in model.parameters() if p.requires_grad),
                 'lr': lr, 'weight_decay': l2_weight},
            ]
        if shape_model is not None:
            for sm in shape_model:
                optim_params.append({'params': sm.parameters(), 'lr': lr, 'weight_decay': l2_weight})

        # Define base optimizer classes and their kwargs
        base_optimizer_config = {
            'Adam': (Adam, {'amsgrad': False}),
            'AdamW': (AdamW, {'amsgrad': False}),
            'RMSprop': (RMSprop, {}),
            'SGD': (SGD, {'nesterov': True, 'momentum': 0.9}),
            'ASGD': (ASGD, {}),
            'AdaBelief': (po.AdaBelief, {}),
            'Lion': (po.Lion, {}),
        }

        if optimizer not in base_optimizer_config:
            raise RuntimeError(f'not implemented: {optimizer}')

        base_optimizer_class, base_kwargs = base_optimizer_config[optimizer]

        # Apply SAM wrapper if requested
        if sam_type is None or sam_type == 'None':
            # Create regular optimizer
            return base_optimizer_class(optim_params, **base_kwargs)
        elif sam_type == 'SAM':
            return SAM(optim_params, base_optimizer_class, rho=sam_rho, **base_kwargs)
        elif sam_type == 'ASAM':
            return ASAM(optim_params, base_optimizer_class, rho=sam_rho, **base_kwargs)
        elif sam_type == 'GSAM':
            return GSAM(optim_params, base_optimizer_class, rho=sam_rho, alpha=sam_alpha, **base_kwargs)
        else:
            raise RuntimeError(f'not implemented SAM type: {sam_type}')

    def build_loss_function(
        self, loss_func: str, model: AbstractFold, args: Namespace
    ) -> nn.Module:
        if loss_func == "hinge" or loss_func == "hinge_mix":
            from mxfold2.loss.structured_loss import StructuredLoss

            return StructuredLoss(
                model,
                loss_pos_paired=args.loss_pos_paired,
                loss_neg_paired=args.loss_neg_paired,
                loss_pos_unpaired=getattr(args, "loss_pos_unpaired", 0.0),
                loss_neg_unpaired=getattr(args, "loss_neg_unpaired", 0.0),
                perturb=args.perturb,
                l1_weight=args.l1_weight,
                l2_weight=args.l2_weight,
                sl_weight=args.score_loss_weight,
            )

        if loss_func == "fy" or loss_func == "fy_mix":
            from mxfold2.loss.fy_loss import FenchelYoungLoss

            return FenchelYoungLoss(
                model,
                perturb=args.perturb,
                l1_weight=args.l1_weight,
                l2_weight=args.l2_weight,
                sl_weight=args.score_loss_weight,
            )

        if loss_func == "f1":
            from mxfold2.loss.f1_loss import F1Loss

            return F1Loss(
                model,
                perturb=args.perturb,
                nu=args.nu,
                l1_weight=args.l1_weight,
                l2_weight=args.l2_weight,
                sl_weight=args.score_loss_weight,
            )

        else:
            raise (ValueError(f"not implemented: {loss_func}"))

    def build_shape_loss_function(self, loss_func: str, model: AbstractFold, args: Namespace,
                                  shape_model: Optional[list[nn.Module]] = None,
                                  lwf_model: Optional[AbstractFold] = None) -> nn.Module:
        from .loss.implicit_mle import AdaptiveStepConfig

        adaptive = AdaptiveStepConfig(
            enabled=args.shape_imle_adaptive_nu,
            minimum=args.shape_imle_nu_min,
            maximum=args.shape_imle_nu_max,
            flip_fraction_low=args.shape_imle_flip_low,
            flip_fraction_high=args.shape_imle_flip_high,
            factor=args.shape_imle_adaptive_factor,
            max_steps=args.shape_imle_adaptive_steps,
        )
        imle_options = dict(
            alignment=args.shape_imle_alignment,
            normalization=args.shape_imle_normalization,
            estimator_scale=args.shape_imle_estimator_scale,
            difference_mode=args.shape_imle_difference,
            num_samples=args.shape_imle_samples,
            adaptive=adaptive,
            train_turner=getattr(args, 'shape_imle_train_turner', False),
        )
        if loss_func == 'shape_nll':
            from mxfold2.loss.shape_nll_loss import ShapeNLLLoss
            return ShapeNLLLoss(model=model,
                            shape_model=shape_model,
                            shape_only=args.shape_only_training,
                            lwf_model=lwf_model,
                            lwf_weight=args.lwf_weight,
                            perturb=args.shape_perturb, nu=args.shape_nu,
                            l1_weight=args.l1_weight, l2_weight=args.l2_weight,
                            sl_weight=args.score_loss_weight,
                            pseudo_fy_weight=args.shape_pseudo_fy_weight,
                            weight_schedule=args.weight_schedule,
                            weight_schedule_start=args.weight_schedule_start,
                            weight_schedule_end=args.weight_schedule_end,
                            **imle_options)

        elif loss_func == 'shape_fy':
            from mxfold2.loss.shape_fy_loss import ShapeFenchelYoungLoss
            return ShapeFenchelYoungLoss(model,
                            perturb=args.shape_perturb,
                            shape_intercept=args.shape_intercept, shape_slope=args.shape_slope,
                            l1_weight=args.l1_weight, l2_weight=args.l2_weight,
                            sl_weight=args.score_loss_weight,
                            weight_schedule=args.weight_schedule,
                            weight_schedule_start=args.weight_schedule_start,
                            weight_schedule_end=args.weight_schedule_end)

        elif loss_func == 'shape_partial_fy':
            from mxfold2.loss.shape_partial_fy_loss import ShapePartialFenchelYoungLoss
            return ShapePartialFenchelYoungLoss(model,
                            perturb=args.shape_perturb,
                            unpaired_threshold=args.shape_unpaired_threshold,
                            l1_weight=args.l1_weight,
                            sl_weight=args.score_loss_weight)

        elif loss_func == 'shape_rank':
            from mxfold2.loss.shape_rank_loss import ShapeRankLoss
            return ShapeRankLoss(model,
                            objective=getattr(args, 'shape_rank_objective', 'pairwise_logistic'),
                            margin=args.shape_margin,
                            tau=args.shape_rank_tau,
                            weighting=getattr(args, 'shape_rank_weighting', 'uniform'),
                            rank_delta=args.shape_rank_delta,
                            teacher_temperature=args.shape_rank_teacher_temperature,
                            teacher_normalization=getattr(args, 'shape_rank_teacher_normalization', 'none'),
                            clip_targets=args.shape_rank_targets == 'clipped',
                            perturb=args.shape_perturb, nu=args.shape_nu,
                            l1_weight=args.l1_weight, l2_weight=args.l2_weight,
                            sl_weight=args.score_loss_weight,
                            pseudo_fy_weight=args.shape_pseudo_fy_weight,
                            weight_schedule=args.weight_schedule,
                            weight_schedule_start=args.weight_schedule_start,
                            weight_schedule_end=args.weight_schedule_end,
                            sample_aggregation=args.shape_rank_sample_aggregation,
                            **imle_options)

        else:
            raise(ValueError(f'not implemented: {loss_func}'))

    def build_shape_regress_loss_function(self, model: AbstractFold):
        net = getattr(getattr(model, "zuker", model), "net", None)
        if net is None:
            raise ValueError("Multitask training requires a neural folding model")

        def shape_loss(seq, target, mask):
            pred = net.predict_shape(seq)
            target = target.to(pred.device)
            mask = mask.to(device=pred.device, dtype=torch.bool)
            return (pred[mask] - target[mask]).square().sum() / mask.sum().clamp_min(1)

        return shape_loss

    def build_scheduler(self, scheduler: str, optimizer: Optimizer, args: Namespace):
        if scheduler == "CyclicLR":
            return optim.lr_scheduler.CyclicLR(
                optimizer=optimizer,
                base_lr=0.001,
                max_lr=args.lr,
                step_size_up=args.scheduler_step_size,
                gamma=args.scheduler_gamma,
                mode="exp_range",
            )
        if scheduler == "CosineAnnealingLR":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=args.scheduler_step_size,
                eta_min=args.lr * 1e-3,
            )

        return None

    def save_config(self, file: str, config: dict[str, Any]) -> None:
        with open(file, "w") as f:
            for k, v in config.items():
                k = "--" + k.replace("_", "-")
                if isinstance(v, bool):
                    if v:
                        f.write("{}\n".format(k))
                elif isinstance(v, list) or isinstance(v, tuple):
                    for vv in v:
                        f.write("{}\n{}\n".format(k, vv))
                elif v is not None:
                    f.write("{}\n{}\n".format(k, v))

    def build_training_loader(
        self, args: Namespace, generator: torch.Generator,
    ) -> tuple[DataLoader, int]:
        inputs = [args.input] if isinstance(args.input, str) else args.input
        weights = (
            [float(weight) for weight in args.dataset_weight.split(",")]
            if args.dataset_weight else [1.0] * len(inputs)
        )
        if len(weights) != len(inputs):
            raise ValueError("The number of --dataset-weight values must match the number of inputs")

        convert = args.convert_t_to_u
        if args.task == "Multitask":
            if args.replay_fraction > 0.0:
                raise ValueError("--replay-fraction is supported for Folding and Implicit_MLE")
            if not args.shape or len(inputs) != len(args.shape):
                raise ValueError("Multitask requires equal numbers of BPSEQ and SHAPE lists")
            datasets = [
                MultiTaskDataset(bpseq, shape, index, convert, weight=weight)
                for index, (bpseq, shape, weight) in enumerate(zip(inputs, args.shape, weights))
            ]
        else:
            datasets = [
                BPseqDataset(path, convert, weight)
                for path, weight in zip(inputs, weights)
            ]

        if args.json_dataset:
            datasets.append(JsonDataset(args.json_dataset, convert))
        replay_start = sum(len(dataset) for dataset in datasets)
        replay_length = 0
        if args.input_2 is not None:
            replay = BPseqDataset(args.input_2, convert, type_name="BPSEQ2")
            datasets.append(replay)
            replay_length = len(replay)
        primary_length = replay_start + replay_length

        n_shape_datasets = 0
        if args.ribonanza:
            ribonanza = RibonanzaDataset(args.ribonanza, convert_t_to_u_flag=convert)
            datasets.append(ribonanza)
            n_shape_datasets += len(ribonanza.dataset_id)

        if args.task != "Multitask":
            for shape in args.shape or ():
                datasets.append(ShapeDataset(shape, n_shape_datasets, convert))
                n_shape_datasets += 1

        if args.json_shape_dataset:
            json_shape = JsonShapeDataset(args.json_shape_dataset, n_shape_datasets, convert)
            datasets.append(json_shape)
            n_shape_datasets += len(json_shape.dataset_id)

        if args.task == "Implicit_MLE" and not n_shape_datasets:
            raise ValueError("Implicit_MLE requires reactivity data")

        for path in args.extra_dataset or ():
            datasets.append(BPseqDataset(path, convert))

        dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
        n_samples = len(dataset)
        if not n_samples:
            raise ValueError("Training data is empty")
        replay_fraction = args.replay_fraction
        budget = args.samples_per_epoch
        if budget < 0:
            raise ValueError("--samples-per-epoch must be nonnegative")
        if not 0.0 <= replay_fraction < 1.0:
            raise ValueError("--replay-fraction must be in [0, 1)")
        if replay_fraction and (not replay_length or replay_length == n_samples):
            raise ValueError("--replay-fraction requires non-empty primary and --input-2 data")

        auxiliary_length = n_samples - primary_length
        downsample = args.downsampling > 0.0 and primary_length > 0 and auxiliary_length > 0
        sampler = None
        if replay_fraction or downsample:
            sampling_weights = torch.ones(n_samples, dtype=torch.double)
            if downsample:
                sampling_weights[:primary_length] = 1.0 / primary_length
                sampling_weights[primary_length:] = args.downsampling / auxiliary_length
            if replay_fraction:
                replay_slice = slice(replay_start, replay_start + replay_length)
                sampling_weights[replay_slice] = 0.0
                sampling_weights *= (1.0 - replay_fraction) / sampling_weights.sum()
                sampling_weights[replay_slice] = replay_fraction / replay_length

            draws = budget or n_samples
            if downsample and not (budget or replay_fraction):
                draws = min(n_samples, int(primary_length * (1.0 + args.downsampling) + 0.5))
            sampler = WeightedRandomSampler(
                sampling_weights, draws, replacement=bool(budget or replay_fraction),
                generator=generator,
            )
            logging.info("Sampling %d of %d records per epoch", draws, n_samples)
        elif budget:
            sampler = RandomSampler(dataset, replacement=True, num_samples=budget, generator=generator)
            logging.info("Budgeted sampling: samples/epoch=%d dataset=%d", budget, n_samples)

        loader = DataLoader(
            dataset, batch_size=1, sampler=sampler, shuffle=sampler is None, generator=generator,
        )
        return loader, n_shape_datasets

    def run(self, args: Namespace, conf: Optional[str] = None) -> None:
        self.disable_progress_bar = args.disable_progress_bar
        self.gpu = args.gpu
        loglevel = "INFO" if args.verbose else args.loglevel
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=getattr(logging, loglevel, None),
        )

        if args.shape_imle_samples < 1:
            raise ValueError("--shape-imle-samples must be at least 1")
        if args.shape_imle_samples > 1 and args.shape_perturb <= 0:
            logging.warning("Multiple I-MLE samples without perturbation repeat the same fold")

        generator = torch.Generator()
        if args.seed >= 0:
            generator.manual_seed(args.seed)
        train_loader, n_dataset_id = self.build_training_loader(args, generator)

        # Keep the historical --log-dir interface used by the TSUBAME
        # scripts, while also accepting upstream's --output-dir.  The latter
        # is the checkpoint directory; when only one is supplied it is also a
        # sensible place for loss/IMLE logs.
        log_dir = getattr(args, 'log_dir', None)
        if log_dir is None:
            log_dir = args.output_dir
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            self.log_file = os.path.join(log_dir, "loss.log")
        else:
            os.makedirs("logs", exist_ok=True)
            self.log_file = os.path.join("logs", "loss.log")
        # 追記モードで開き、ファイルが空ならヘッダーを書き込む（既存ログがあれば追記のみ）
        write_header = True
        if os.path.exists(self.log_file) and os.path.getsize(self.log_file) > 0:
            write_header = False
        with open(self.log_file, "a") as f:
            if write_header:
                # 追加列: step, alloc_mb
                f.write("epoch,phase,loss,time,step,alloc_mb\n")
        if n_dataset_id and args.shape_loss_func in ('shape_nll', 'shape_rank'):
            self.imle_log_file = os.path.join(log_dir if log_dir is not None else 'logs', 'imle_stats.csv')
            if not os.path.exists(self.imle_log_file) or os.path.getsize(self.imle_log_file) == 0:
                with open(self.imle_log_file, 'w', newline='') as f:
                    csv.writer(f).writerow([
                        'epoch', 'step', 'file', 'difference', 'samples',
                        'nu_mean', 'nu_min', 'nu_max', 'flip_mean',
                        'flip_min', 'flip_max',
                    ])

        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        test_loaders: list[tuple[DataLoader, list[str] | None]] = []
        if args.test_input is not None:
            test_inputs = [args.test_input] if isinstance(args.test_input, str) else args.test_input
            test_shape_lists = getattr(args, "test_shape", None) or []
            for idx, test_path in enumerate(test_inputs):
                test_dataset = BPseqDataset(
                    test_path,
                    convert_t_to_u_flag=getattr(args, "convert_t_to_u", False),
                )
                test_loader = DataLoader(
                    test_dataset, batch_size=1, shuffle=False
                )  # works well only for batch_size=1!!
                shape_paths = None
                if (
                    idx < len(test_shape_lists)
                    and test_shape_lists[idx].lower() != "none"
                ):
                    shape_paths = read_shape_list(test_shape_lists[idx])
                    if len(shape_paths) != len(test_dataset):
                        raise ValueError(
                            f"SHAPE list length ({len(shape_paths)}) != "
                            f"test dataset length ({len(test_dataset)}) for {test_path}"
                        )
                test_loaders.append((test_loader, shape_paths))

        if args.seed >= 0:
            torch.manual_seed(args.seed)
            random.seed(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        model, config = self.build_model(args)
        config.update({"model": args.model, "param": args.param, "fold": args.fold})
        resume_state = None
        if args.resume is not None:
            resume_state = torch.load(args.resume, map_location="cpu")
            # The unused 1D SHAPE head was introduced on master. Keep the
            # parameter groups of older modification checkpoints intact.
            for name, module in model.named_modules():
                if hasattr(module, "shape_head_1d") and f"{name}.shape_head_1d.weight" not in resume_state["model_state_dict"]:
                    module.shape_head_1d = None

        shape_model = None
        if n_dataset_id and (
            args.shape_loss_func == "shape_nll"
            or (resume_state is not None and "shape_model_state_dict" in resume_state)
        ):
            shape_model = [self.build_shape_model(args) for _ in range(n_dataset_id)]

        if args.init_param != "":
            init_param = Path(args.init_param)
            if not init_param.exists() and conf is not None:
                init_param = Path(conf).parent / init_param
            p = torch.load(init_param, map_location="cpu")
            if (
                shape_model is not None
                and isinstance(p, dict)
                and "shape_model_state_dict" in p
            ):
                for i, sm in enumerate(shape_model):
                    sm.load_state_dict(p["shape_model_state_dict"][i])
            if isinstance(p, dict) and "model_state_dict" in p:
                p = p["model_state_dict"]
            self.load_model_weights(model, p)

        lwf_model = None
        if args.lwf_weight > 0.0:
            lwf_model, _ = self.build_model(args)
            self.load_model_weights(lwf_model, model.state_dict())
            lwf_model.eval().requires_grad_(False)

        device, self.device_type = self.get_device(args.gpu)
        model.to(device)
        if shape_model is not None:
            for sm in shape_model:
                sm.to(device)
        if lwf_model is not None:
            lwf_model.to(device)

        # --- Freeze specified high-level components (encoder, fc_paired) ---
        self._frozen_eval_modules = []
        if hasattr(args, 'freeze') and args.freeze:
            keys = []
            for item in args.freeze:
                for k in item.split(','):
                    k = k.strip()
                    if k:
                        keys.append(k)

            # try to find the network module (handles model.zuker.net or model.net)
            net = None
            if hasattr(model, 'zuker') and hasattr(model.zuker, 'net'):
                net = model.zuker.net
            elif hasattr(model, 'net'):
                net = model.net
            else:
                net = model

            def _register_frozen_eval_module(m: Optional[nn.Module]) -> None:
                if m is not None and all(existing is not m for existing in self._frozen_eval_modules):
                    self._frozen_eval_modules.append(m)

            def _freeze_module(m: Optional[nn.Module]) -> bool:
                if m is None:
                    return False
                for p in m.parameters():
                    p.requires_grad = False
                _register_frozen_eval_module(m)
                return True

            applied = []
            if 'encoder' in keys:
                if hasattr(net, 'encoder'):
                    _freeze_module(getattr(net, 'encoder'))
                    applied.append('encoder')
                else:
                    logging.warning("freeze requested: 'encoder' not found on model")
            if 'fc_paired' in keys:
                if hasattr(net, 'fc_paired'):
                    _freeze_module(getattr(net, 'fc_paired'))
                    applied.append('fc_paired')
                else:
                    logging.warning("freeze requested: 'fc_paired' not found on model")

            # fallback: if a key appears in parameter names, freeze those params
            for k in keys:
                if k in ('encoder','fc_paired'):
                    continue
                matched = False
                for name, p in model.named_parameters():
                    if k in name:
                        p.requires_grad = False
                        matched = True
                if matched:
                    applied.append(k)
                    # Register the corresponding module(s) for eval mode as
                    # well.  This also handles generic keys such as
                    # "embedding", which freeze by parameter-name matching.
                    for module_name, module in model.named_modules():
                        if module_name.rsplit('.', 1)[-1] != k:
                            continue
                        module_params = list(module.parameters())
                        if module_params and all(not p.requires_grad for p in module_params):
                            _register_frozen_eval_module(module)

            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logging.info(f"Freeze keys={keys} applied={applied}. trainable params={trainable}/{total}")

        torch.set_num_threads(args.threads)
        interface.set_num_threads(args.threads)

        self._l2_sp_weight = getattr(args, 'l2_sp_weight', 0.0)
        self._l2_sp_anchor = {}
        if self._l2_sp_weight > 0.0:
            anchor_path = getattr(args, 'l2_sp_anchor', '')
            if not anchor_path:
                raise ValueError('--l2-sp-anchor is required when --l2-sp-weight is positive')
            anchor_checkpoint = torch.load(anchor_path, map_location='cpu')
            if isinstance(anchor_checkpoint, dict) and 'model_state_dict' in anchor_checkpoint:
                anchor_state = anchor_checkpoint['model_state_dict']
            else:
                anchor_state = anchor_checkpoint
            for name, parameter in model.named_parameters():
                if '.embedding.' not in name and '.encoder.' not in name:
                    continue
                value = anchor_state.get(name)
                if value is None:
                    raise ValueError(f'L2-SP anchor is missing parameter: {name}')
                if value.shape != parameter.shape:
                    raise ValueError(
                        f'L2-SP anchor shape mismatch for {name}: {value.shape} != {parameter.shape}'
                    )
                self._l2_sp_anchor[name] = value.detach().to(device=device, dtype=parameter.dtype)
            if not self._l2_sp_anchor:
                raise ValueError('No embedding or encoder parameters found for L2-SP')
            logging.info(
                'L2-SP weight=%g anchored_parameters=%d anchored_numel=%d',
                self._l2_sp_weight,
                len(self._l2_sp_anchor),
                sum(value.numel() for value in self._l2_sp_anchor.values()),
            )

        optimizer = self.build_optimizer(
            args.optimizer,
            model,
            args.lr,
            args.l2_weight,
            shape_model=shape_model,
            sam_type=getattr(args, "sam_type", None),
            sam_rho=getattr(args, "sam_rho", 0.05),
            sam_alpha=getattr(args, "sam_alpha", 0.1),
            lr_overrides={
                "embedding": args.lr_embedding,
                "encoder": args.lr_encoder,
                "fc_paired": args.lr_fc_paired,
            },
        )

        loss_fn = {"BPSEQ": self.build_loss_function(args.loss_func, model, args)}
        if n_dataset_id:
            loss_fn["SHAPE"] = self.build_shape_loss_function(
                args.shape_loss_func, model, args, shape_model=shape_model, lwf_model=lwf_model,
            )
        if args.task == "Multitask":
            loss_fn["SHAPE_regress"] = self.build_shape_regress_loss_function(model)
        loss_weight = {
            "BPSEQ": args.bpseq_loss_weight,
            "BPSEQ2": args.second_loss_weight,
            "SHAPE": args.shape_loss_weight,
            "MULTI": 1.0,
            "MULTI_intra": (args.mt_alpha, args.mt_beta),
        }
        scheduler = self.build_scheduler(args.scheduler, optimizer, args)

        # Initialize GradScaler for mixed precision training
        scaler = None
        use_amp = False
        if (
            hasattr(args, "use_amp")
            and args.use_amp
            and self.device_type in ("cuda", "mps")
        ):
            use_amp = True
            if self.device_type == "cuda":
                scaler = GradScaler(f"cuda:{args.gpu}")
            else:  # MPS
                scaler = GradScaler("mps")
            logging.info(
                f"Using Automatic Mixed Precision (AMP) training on {self.device_type.upper()}"
            )

        # Initialize EMA before resume (so state can be restored)
        if args.ema:
            ema = EMA(model, decay=args.ema_decay)
            ema_start = (
                args.ema_start
                if args.ema_start >= 1.0
                else args.epochs * args.ema_start
            )
        else:
            ema = None
            ema_start = args.epochs

        checkpoint_epoch = 0
        resumed_wandb_run_id: Optional[str] = None
        if resume_state is not None:
            checkpoint_epoch, resumed_scaler, resumed_wandb_run_id = (
                self.resume_checkpoint(
                    resume_state,
                    model,
                    optimizer,
                    scheduler,
                    shape_model,
                    ema,
                    scaler,
                    args.gpu,
                )
            )
            if resumed_scaler is not None:
                scaler = resumed_scaler
                use_amp = True
                logging.info("Resumed with Automatic Mixed Precision (AMP) training")

        # Initialize wandb if project is specified
        self.use_wandb = args.wandb_project is not None
        if self.use_wandb:
            if wandb is None:
                raise RuntimeError("wandb is required when --wandb-project is used")
            wandb_config = vars(args).copy()
            if resumed_wandb_run_id is not None and not args.wandb_new_run:
                wandb.init(
                    project=args.wandb_project,
                    id=resumed_wandb_run_id,
                    resume="allow",
                )
                logging.info(f"Resumed wandb run: {resumed_wandb_run_id}")
            else:
                wandb.init(
                    project=args.wandb_project,
                    name=args.wandb_run_name,
                    tags=args.wandb_tags,
                    config=wandb_config,
                )

        if args.swa:
            swa_model = AveragedModel(model)
            swa_start = (
                args.swa_start if args.swa_start > 1.0 else args.epochs * args.swa_start
            )
            swa_scheduler = SWALR(
                optimizer,
                swa_lr=args.swa_lr,
                anneal_epochs=args.swa_anneal_epochs,
                anneal_strategy=args.swa_anneal_strategy,
            )
        else:
            swa_start = args.epochs
            swa_model = None
            swa_scheduler = None

        for epoch in range(checkpoint_epoch + 1, args.epochs + 1):
            # Set random seed for this epoch to ensure reproducible shuffling
            if args.seed >= 0:
                epoch_seed = args.seed + epoch
                generator.manual_seed(epoch_seed)

            # Update epoch info for Shape loss functions (for weight scheduling inside loss)
            if "SHAPE" in loss_fn and hasattr(loss_fn["SHAPE"], "set_epoch_info"):
                loss_fn["SHAPE"].set_epoch_info(epoch, args.epochs)  # type: ignore[union-attr]

            epoch_start = time.time()
            self.train(
                epoch,
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                data_loader=train_loader,
                loss_weight=loss_weight,
                clip_grad_value=args.clip_grad_value,
                clip_grad_norm=args.clip_grad_norm,
                scaler=scaler,
                use_amp=use_amp,
                grad_accum_steps=getattr(args, "grad_accum_steps", 1),
            )

            # Get current learning rate
            current_lr = args.lr
            if (
                swa_model is not None
                and swa_scheduler is not None
                and epoch > swa_start
            ):
                swa_model.update_parameters(model)
                swa_scheduler.step()
                current_lr = swa_scheduler.get_last_lr()[0]
                logging.info(f"LR = {current_lr}")
            elif scheduler is not None:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
                logging.info(f"LR = {current_lr}")

            # Update EMA after each epoch (after warmup)
            if ema is not None and epoch > ema_start:
                ema.update(model)

            if test_loaders:
                # Save RNG states before test() to ensure test data doesn't affect training reproducibility
                rng_state_torch = torch.get_rng_state()
                rng_state_python = random.getstate()
                rng_state_cuda = (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                )

                # Use SWA model if available, otherwise EMA model (if started), otherwise regular model
                eval_model = cast(
                    AbstractFold | AveragedModel,
                    swa_model
                    or (ema.shadow if ema and epoch > ema_start else None)
                    or model,
                )
                eval_loss_fn = loss_fn
                if eval_model is not model:
                    eval_fold = eval_model.module if isinstance(eval_model, AveragedModel) else eval_model
                    eval_loss_fn = {"BPSEQ": self.build_loss_function(args.loss_func, eval_fold, args)}
                for test_idx, (t_loader, t_shape_paths) in enumerate(test_loaders):
                    prefix = "test" if len(test_loaders) == 1 else f"test_{test_idx}"
                    self.test(
                        epoch,
                        model=eval_model,
                        loss_fn=eval_loss_fn,
                        data_loader=t_loader,
                        use_amp=use_amp,
                        shape_paths=t_shape_paths,
                        metric_prefix=prefix,
                    )

                # Restore RNG states after test()
                torch.set_rng_state(rng_state_torch)
                random.setstate(rng_state_python)
                if rng_state_cuda is not None:
                    torch.cuda.set_rng_state_all(rng_state_cuda)

            epoch_time = time.time() - epoch_start
            if self.use_wandb:
                log_dict = {
                    "epoch": epoch,
                    "epoch_time": epoch_time,
                    "learning_rate": current_lr,
                }
                wandb.log(log_dict)

            checkpoint_dir = args.output_dir or args.log_dir
            checkpoint_interval = max(1, args.checkpoint_every_n_epochs)
            if checkpoint_dir is not None and (
                epoch % checkpoint_interval == 0 or epoch == args.epochs
            ):
                self.save_checkpoint(
                    checkpoint_dir,
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    shape_model,
                    ema=ema,
                    step=self.step,
                    scaler=scaler,
                )

        if args.param is not None:
            # Priority: SWA model > EMA model (if started) > regular model
            final_model = (
                swa_model
                or (ema.shadow if ema and args.epochs > ema_start else None)
                or model
            )
            torch.save(final_model.state_dict(), args.param)
        if args.save_config is not None:
            self.save_config(args.save_config, config)

        if self.use_wandb:
            wandb.finish()

    @classmethod
    def add_args(cls, parser):
        subparser = parser.add_parser("train", help="training")
        # input
        subparser.add_argument(
            "input",
            type=str,
            nargs="+",
            help="Training data of the list(s) of BPSEQ-formatted files (can specify multiple files)",
        )
        subparser.add_argument(
            "--test-input",
            type=str,
            action="append",
            help="Test data of the list of BPSEQ-formatted files (can be specified multiple times)",
        )
        subparser.add_argument(
            "--test-shape",
            type=str,
            action="append",
            help="SHAPE reactivity list file for test data "
            "(pairs with --test-input by position; use 'none' to skip, "
            "can be specified multiple times)",
        )
        subparser.add_argument(
            "--gpu",
            type=int,
            default=-1,
            help="use GPU with the specified ID (default: -1 = CPU)",
        )
        subparser.add_argument(
            "--threads",
            type=int,
            default=1,
            metavar="N",
            help="the number of threads (default: 1)",
        )
        subparser.add_argument(
            "--seed", type=int, default=0, metavar="S", help="random seed (default: 0)"
        )
        subparser.add_argument(
            "--param",
            type=str,
            default="param.pth",
            help="output file name of trained parameters",
        )
        subparser.add_argument(
            "--init-param",
            type=str,
            default="",
            help="the file name of the initial parameters",
        )
        subparser.add_argument(
            "--shape",
            type=str,
            action="append",
            help="specify the file name that includes SHAPE reactivity",
        )
        subparser.add_argument(
            "--ribonanza",
            type=str,
            help="specify the file name that includes SHAPE reactivity with Ribonanza format",
        )
        subparser.add_argument(
            "--extra-dataset",
            type=str,
            action="append",
            help="Extra dataset for training (BPSEQ format) with downsampling",
        )
        subparser.add_argument(
            "--json-dataset",
            type=str,
            action="append",
            help="Extra dataset for training (JSON format)",
        )
        subparser.add_argument(
            "--json-shape-dataset",
            type=str,
            action="append",
            help="SHAPE dataset for training (JSON format)",
        )
        subparser.add_argument(
            "--convert-t-to-u",
            action="store_true",
            help="convert T to U in input sequences",
        )
        subparser.add_argument(
            "--dataset-weight",
            type=str,
            default=None,
            help="comma-separated weights for each input dataset "
            "(must match the number of input files). "
            "e.g., --dataset-weight 1.0,0.5 means the first dataset has weight 1.0 "
            "and the second has weight 0.5. Default: all 1.0",
        )

        subparser.add_argument('--input-2', type=str, default=None,
                            help='Second training data of the list of BPSEQ-formatted files')
        subparser.add_argument('--replay-fraction', type=float, default=0.0,
                            help='Sampling fraction for --input-2 replay data (default: 0; replacement sampling)')
        subparser.add_argument('--samples-per-epoch', type=int, default=0,
                            help='Bound draws per epoch (0: original full-dataset behavior)')
        subparser.add_argument('--second-loss-weight', type=float, default=1.0,
                            help='loss weight for the second BPSEQ dataset (default: 1.0)')
        subparser.add_argument('--freeze', type=str, action='append',
                               help='freeze components for fine-tuning(Not --resume, but --init-param). repeatable or comma-separated. supported keys: encoder,fc_paired')
        subparser.add_argument('--l2-sp-anchor', type=str, default='',
                            help='Checkpoint anchoring embedding and encoder parameters for L2-SP')
        subparser.add_argument('--l2-sp-weight', type=float, default=0.0,
                            help='L2-SP coefficient for squared distance from --l2-sp-anchor')

        gparser = subparser.add_argument_group("Training environment")
        subparser.add_argument(
            "--epochs",
            type=int,
            default=10,
            metavar="N",
            help="number of epochs to train (default: 10)",
        )
        subparser.add_argument(
            "--output-dir",
            type=str,
            default=None,
            help="Directory for storing checkpoints",
        )
        subparser.add_argument(
            "--resume", type=str, default=None, help="Checkpoint file for resume"
        )
        subparser.add_argument(
            "--save-config", type=str, default=None, help="save model configurations"
        )
        subparser.add_argument(
            "--disable-progress-bar",
            action="store_true",
            help="disable the progress bar in training",
        )
        subparser.add_argument(
            "--verbose",
            action="store_true",
            help="enable verbose outputs for debugging",
        )
        subparser.add_argument(
            "--loglevel",
            choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
            default="WARNING",
            help="set the log level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')",
        )
        subparser.add_argument(
            "--use-amp",
            action="store_true",
            help="use automatic mixed precision (AMP) for faster training on GPUs",
        )

        subparser.add_argument('--log-dir', type=str, default=None,
                            help='Directory for loss/IMLE logs and, when --output-dir is omitted, checkpoints')
        subparser.add_argument('--checkpoint-every-n-epochs', type=int, default=1,
                            help='Save checkpoints every N epochs (always save the final epoch)')

        gparser = subparser.add_argument_group("Weights & Biases (wandb)")
        gparser.add_argument(
            "--wandb-project",
            type=str,
            default=None,
            help="wandb project name (enables wandb logging when specified)",
        )
        gparser.add_argument(
            "--wandb-run-name", type=str, default=None, help="wandb run name (optional)"
        )
        gparser.add_argument(
            "--wandb-tags",
            type=str,
            nargs="*",
            default=None,
            help="wandb tags (optional)",
        )
        gparser.add_argument(
            "--wandb-new-run",
            action="store_true",
            help="start a new wandb run even when resuming from a checkpoint with a saved run ID",
        )

        cls.add_task_args(subparser)
        cls.add_fold_args(subparser)

        gparser = subparser.add_argument_group("Setting for optimizer")
        gparser.add_argument(
            "--optimizer",
            choices=("Adam", "AdamW", "RMSprop", "SGD", "ASGD", "AdaBelief", "Lion"),
            default="AdamW",
        )
        gparser.add_argument(
            "--lr",
            type=float,
            default=0.001,
            help="the learning rate for optimizer (default: 0.001)",
        )
        gparser.add_argument('--lr-embedding', type=float, default=None,
                            help='Optional learning rate override for embedding parameters')
        gparser.add_argument('--lr-encoder', type=float, default=None,
                            help='Optional learning rate override for encoder parameters')
        gparser.add_argument('--lr-fc-paired', type=float, default=None,
                            help='Optional learning rate override for 2D paired-head parameters')

        gparser.add_argument(
            "--clip-grad-value",
            type=float,
            default=0.0,
            help="gradient clipping by values (default=0, no clipping)",
        )
        gparser.add_argument(
            "--clip-grad-norm",
            type=float,
            default=0.0,
            help="gradient clipping by norm (default=0, no clipping)",
        )
        gparser.add_argument(
            "--scheduler",
            choices=("None", "CyclicLR", "CosineAnnealingLR"),
            default="None",
            help="learning rate scheduler ('None', 'CyclicLR', 'CosineAnnealingLR')",
        )
        gparser.add_argument(
            "--scheduler-step-size",
            type=int,
            default=5,
            help="scheduler step size (default=5)",
        )
        gparser.add_argument(
            "--scheduler-gamma",
            type=float,
            default=0.95,
            help="scheduler decoy rate (default=0.95)",
        )
        gparser.add_argument(
            "--swa",
            default=False,
            action="store_true",
            help="use stochastic weight averaging (SWA)",
        )
        gparser.add_argument(
            "--swa-start",
            type=float,
            default=0.75,
            help="start epoch of SWA: epochs * swa_start for swa_start<1.0, or swa_start otherwise",
        )
        gparser.add_argument(
            "--swa-anneal-epochs",
            type=int,
            default=10,
            help="SWA anneal-epochs (default: 10)",
        )
        gparser.add_argument(
            "--swa-anneal-strategy",
            choices=("linear", "cos"),
            default="linear",
            help="SWA anneal strategy ('linear', 'cos')",
        )
        gparser.add_argument(
            "--swa-lr",
            type=float,
            default=0.01,
            help="SWA learning rate (default: 0.01)",
        )
        gparser.add_argument(
            "--ema",
            default=False,
            action="store_true",
            help="use exponential moving average (EMA) for model weights",
        )
        gparser.add_argument(
            "--ema-decay",
            type=float,
            default=0.999,
            help="EMA decay factor (default: 0.999)",
        )
        gparser.add_argument(
            "--ema-start",
            type=float,
            default=0,
            help="epoch to start EMA (default: 0). If < 1.0, fraction of total epochs.",
        )
        gparser.add_argument(
            "--sam-type",
            choices=("None", "SAM", "ASAM", "GSAM"),
            default="None",
            help="SAM optimizer type ('None', 'SAM', 'ASAM', 'GSAM')",
        )
        gparser.add_argument(
            "--sam-rho",
            type=float,
            default=0.05,
            help="SAM perturbation radius (default: 0.05, use 0.5 for ASAM)",
        )
        gparser.add_argument(
            "--sam-alpha",
            type=float,
            default=0.1,
            help="GSAM gap weighting parameter (default: 0.1)",
        )
        gparser.add_argument(
            "--grad-accum-steps",
            type=int,
            default=1,
            help="Gradient accumulation steps (default: 1, no accumulation)",
        )

        gparser = subparser.add_argument_group("Setting for loss function")
        gparser.add_argument(
            "--loss-func",
            choices=("hinge", "hinge_mix", "fy", "fy_mix", "f1"),
            default="hinge",
            help="loss fuction (default: hinge)",
        )
        gparser.add_argument(
            "--l1-weight",
            type=float,
            default=0.0,
            help="the weight for L1 regularization (default: 0)",
        )
        gparser.add_argument(
            "--l2-weight",
            type=float,
            default=0.0,
            help="the weight for L2 regularization (default: 0)",
        )
        gparser.add_argument(
            "--score-loss-weight",
            type=float,
            default=0.0,
            help="the weight for score loss for {hinge,fy} loss (default: 0)",
        )
        gparser.add_argument(
            "--lwf-weight",
            type=float,
            default=0.0,
            help="the weight for learn without forgetting (LwF) in the incremental training (default: 0)",
        )
        gparser.add_argument(
            "--perturb",
            type=float,
            default=0.1,
            help="standard deviation of perturbation for fy loss (default: 0.1)",
        )
        gparser.add_argument(
            "--nu",
            type=float,
            default=0.1,
            help="weight for distribution (default: 0.1)",
        )
        gparser.add_argument(
            "--loss-pos-paired",
            type=float,
            default=0.5,
            help="the penalty for positive base-pairs for loss augmentation (default: 0.5)",
        )
        gparser.add_argument(
            "--loss-neg-paired",
            type=float,
            default=0.005,
            help="the penalty for negative base-pairs for loss augmentation (default: 0.005)",
        )
        gparser.add_argument('--bpseq-loss-weight', type=float, default=1.,
                            help='weight for BPSEQ examples mixed into Implicit_MLE training (default: 1)')
        gparser.add_argument('--loss-pos-unpaired', type=float, default=0.,
                            help='the penalty for positive unpaired bases for loss augmentation (default: 0)')
        gparser.add_argument('--loss-neg-unpaired', type=float, default=0.,
                            help='the penalty for negative unpaired bases for loss augmentation (default: 0)')

        gparser.add_argument(
            "--shape-loss-func",
            choices=("shape_nll", "shape_fy", "shape_partial_fy", "shape_rank"),
            default="shape_nll",
            help="loss fuction for SHAPE training data (default: shape)",
        )
        gparser.add_argument(
            "--shape-perturb",
            type=float,
            default=0.1,
            help="standard deviation of perturbation for shape loss (default: 0.1)",
        )
        gparser.add_argument(
            "--shape-nu",
            type=float,
            default=0.1,
            help="weight for distribution for shape loss (default: 0.1)",
        )
        gparser.add_argument(
            "--shape-margin",
            type=float,
            default=0.0,
            help="margin for shape rank loss (default: 0.0)",
        )
        gparser.add_argument(
            "--shape-pseudo-fy-weight",
            type=float,
            default=0.0,
            help="weight for FY loss using Turner structure as pseudo ground truth (default: 0.0)",
        )
        subparser.add_argument(
            "--shape-intercept",
            type=float,
            default=-0.8,
            help="Specify an intercept used with SHAPE restraints. Default is -0.8 kcal/mol.",
        )
        gparser.add_argument(
            "--shape-slope",
            type=float,
            default=2.6,
            help="Specify a slope used with SHAPE restraints. Default is 2.6.",
        )
        gparser.add_argument(
            "--shape-loss-weight",
            type=float,
            default=1.0,
            help="weight for SHAPE loss function (default=1)",
        )
        gparser.add_argument('--shape-unpaired-threshold', type=float, default=0.8,
                            help='Minimum measured reactivity forced unpaired by shape_partial_fy (default: 0.8)')
        gparser.add_argument('--shape-rank-tau', type=float, default=0.2,
                            help='temperature for SHAPE Rank Loss (default=0.2)')
        gparser.add_argument('--shape-rank-objective', choices=('pairwise_logistic', 'mann_whitney'), default='pairwise_logistic',
                            help='rank objective: legacy logistic or nonparametric 1-AUROC (default=pairwise_logistic)')
        gparser.add_argument('--shape-rank-weighting', choices=('uniform', 'gap', 'log_gap'), default='uniform',
                            help='pair weighting for SHAPE Rank Loss (uniform, gap, or log_gap)')
        gparser.add_argument('--shape-rank-delta', type=float, default=0.0,
                            help='ignore SHAPE order pairs separated by at most this value (default=0)')
        gparser.add_argument('--shape-rank-teacher-temperature', type=float, default=0.0,
                            help='soft-label temperature in SHAPE reactivity units; 0 uses hard ordering (default=0)')
        gparser.add_argument('--shape-rank-teacher-normalization', choices=('none', 'robust', 'percentile'), default='none',
                            help='normalize SHAPE targets within each sequence before rank teacher construction (default=none)')
        gparser.add_argument('--shape-rank-sample-aggregation', choices=('independent', 'mean_paired'), default='independent',
                            help='evaluate rank loss per MAP sample or once on mean pairedness (default=independent)')
        gparser.add_argument('--shape-rank-targets', choices=('raw', 'clipped'), default='raw',
                            help='use raw or [0.01,1] clipped reactivities for rank loss (default=raw)')
        gparser.add_argument('--shape-imle-alignment', choices=('aligned',), default='aligned',
                            help='map the 1-based L+1 pairedness gradient to L bases (default=aligned)')
        gparser.add_argument('--shape-imle-normalization', choices=('none', 'rms', 'max'), default='none',
                            help='normalize pairedness gradients before the second fold (default=none)')
        gparser.add_argument('--shape-imle-estimator-scale', choices=('divide_nu', 'none'), default='divide_nu',
                            help='divide count differences by nu for finite-difference estimation (default=divide_nu)')
        gparser.add_argument('--shape-imle-difference', choices=('forward', 'central'), default='forward',
                            help='one-sided or central finite difference using coupled perturbations (default=forward)')
        gparser.add_argument('--shape-imle-samples', type=int, default=1,
                            help='number of coupled perturb-and-MAP samples to average (default=1)')
        gparser.add_argument('--shape-imle-train-turner', action='store_true',
                            help='opt in to legacy IMLE updates of mixed-model Turner parameters; frozen by default, as in FY/SHAPE-fy')
        gparser.add_argument('--shape-imle-adaptive-nu', action='store_true',
                            help='adapt nu per sequence to a target paired-state flip fraction')
        gparser.add_argument('--shape-imle-nu-min', type=float, default=1e-3,
                            help='minimum nu for per-sequence adaptation (default=1e-3)')
        gparser.add_argument('--shape-imle-nu-max', type=float, default=1e3,
                            help='maximum nu for per-sequence adaptation (default=1e3)')
        gparser.add_argument('--shape-imle-flip-low', type=float, default=0.01,
                            help='minimum target paired-state flip fraction (default=0.01)')
        gparser.add_argument('--shape-imle-flip-high', type=float, default=0.10,
                            help='maximum target paired-state flip fraction (default=0.10)')
        gparser.add_argument('--shape-imle-adaptive-factor', type=float, default=2.,
                            help='multiplicative nu adjustment factor (default=2)')
        gparser.add_argument('--shape-imle-adaptive-steps', type=int, default=4,
                            help='maximum fold attempts for adaptive nu (default=4)')
        gparser.add_argument('--mt-alpha', type=float, default=1.0, help='multitask: weight for structure loss')
        gparser.add_argument('--mt-beta',  type=float, default=1.0, help='multitask: weight for SHAPE loss')

        gparser.add_argument(
            "--downsampling",
            type=float,
            default=0.0,
            help="downsampling for SHAPE data",
        )
        gparser.add_argument(
            "--shape-only-training",
            action="store_true",
            help="training only shape model (available for shape_nll loss)",
        )
        gparser.add_argument(
            "--modified-only",
            action="store_true",
            help="train using only losses involving modified bases (non-ACGU bases)",
        )

        cls.add_network_args(subparser)

        subparser.set_defaults(func=lambda args, conf: Train().run(args, conf))
