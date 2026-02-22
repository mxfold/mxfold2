from __future__ import annotations

import logging
import math
import os
import random
import time
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, cast

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
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from tqdm import tqdm

from mxfold2.ema import EMA
from mxfold2.sam import SAM, ASAM, GSAM

from mxfold2 import interface
from mxfold2.dataset import (
    BPseqDataset,
    FastaDataset,
    ShapeDataset,
    RibonanzaDataset,
    JsonDataset,
    JsonShapeDataset,
)
from mxfold2.compbpseq import compare_bpseq, accuracy
from mxfold2.compreactivity import read_shape_reactivity, pairwise_consistency
from mxfold2.fold.fold import AbstractFold
from mxfold2.common import Common

import wandb


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
        self.device_type = "cpu"  # Will be set in run()

    def train(
        self,
        epoch: int,
        model: AbstractFold,
        optimizer: Optimizer,
        loss_fn: nn.Module | dict[str, nn.Module],
        data_loader: DataLoader[tuple[str, str, dict[str, torch.Tensor]]],
        n_dataset: Optional[int] = None,
        loss_weight=defaultdict(lambda: 1.0),
        clip_grad_value: float = 0.0,
        clip_grad_norm: float = 0.0,
        scaler: Optional[GradScaler] = None,
        use_amp: bool = False,
        grad_accum_steps: int = 1,
    ) -> None:
        model.train()
        if not isinstance(loss_fn, dict):
            loss_fn = {"BPSEQ": loss_fn}
        if n_dataset is None:
            n_dataset = len(cast(FastaDataset, data_loader.dataset))
        loss_total, num = 0.0, 0
        loss_type = defaultdict(lambda: 0.0)
        num_type = defaultdict(lambda: 0)
        running_loss, n_running_loss = 0, 0
        start = time.time()

        # Check if using SAM optimizer
        is_sam = isinstance(optimizer, SAM)

        # Gradient accumulation counter
        accumulated_samples = 0
        optimizer.zero_grad()  # Zero gradients once at the start

        with tqdm(total=n_dataset, disable=self.disable_progress_bar) as pbar:
            for fnames, seqs, vals in data_loader:
                logging.info(f"Step: {self.step}, {fnames}")
                self.step += 1
                n_batch = len(seqs)
                for i in range(n_batch):
                    accumulated_samples += 1
                    is_update_step = accumulated_samples % grad_accum_steps == 0

                    # Define loss computation function for SAM
                    def compute_loss():
                        with autocast(
                            device_type=self.device_type,
                            dtype=torch.float16,
                            enabled=use_amp,
                        ):
                            if vals["type"][i] == "BPSEQ":
                                loss = torch.sum(
                                    loss_fn["BPSEQ"](
                                        seqs[i : i + 1],
                                        vals["target"][i : i + 1],
                                        fname=fnames[i : i + 1],
                                    )
                                )
                            elif vals["type"][i] == "SHAPE":
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
                            return loss * loss_weight[vals["type"][i]]

                    if is_sam:
                        # SAM two-step optimization with gradient accumulation
                        loss = self._sam_step(
                            model,
                            optimizer,
                            compute_loss,
                            clip_grad_value,
                            clip_grad_norm,
                            scaler,
                            use_amp,
                            grad_accum_steps=grad_accum_steps,
                            is_update_step=is_update_step,
                        )
                        if is_update_step:
                            optimizer.zero_grad()
                    else:
                        # Standard optimization with gradient accumulation
                        loss = compute_loss()

                        # Scale loss for gradient accumulation and backward pass
                        scaled_loss = loss / grad_accum_steps
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
                                nn.utils.clip_grad_norm_(
                                    model.parameters(),
                                    max_norm=clip_grad_norm,
                                    norm_type=2,
                                )
                            elif clip_grad_value > 0.0:
                                nn.utils.clip_grad_value_(
                                    model.parameters(), clip_value=clip_grad_value
                                )

                            # Workaround for pytorch_optimizer AdaBelief/Lion: ensure state is initialized
                            # for parameters that didn't have gradients in the first step
                            if isinstance(optimizer, (po.AdaBelief, po.Lion)):
                                for group in optimizer.param_groups:
                                    for p in group["params"]:
                                        if (
                                            p.grad is not None
                                            and len(optimizer.state[p]) == 0
                                        ):
                                            optimizer.state[p]["exp_avg"] = (
                                                torch.zeros_like(p)
                                            )
                                            if isinstance(optimizer, po.AdaBelief):
                                                optimizer.state[p]["exp_avg_var"] = (
                                                    torch.zeros_like(p)
                                                )

                            # Step the optimizer
                            if scaler is not None:
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                optimizer.step()

                            optimizer.zero_grad()

                    loss_total += loss.item()
                    running_loss += loss.item()
                    loss_type[vals["type"][i]] += loss.item()
                    num_type[vals["type"][i]] += 1

                num += n_batch
                pbar.set_postfix(train_loss="{:.3e}".format(loss_total / num))
                pbar.update(n_batch)

                n_running_loss += n_batch
                if n_running_loss >= 100 or num >= n_dataset:
                    running_loss /= n_running_loss
                    if self.use_wandb:
                        log_data = {
                            "train/loss": running_loss,
                            "train/step": (epoch - 1) * n_dataset + num,
                        }
                        if self.device_type == "cuda":
                            log_data["train/gpu_memory_allocated"] = (
                                torch.cuda.memory_allocated(self.gpu) / 1024**3
                            )
                            log_data["train/gpu_memory_reserved"] = (
                                torch.cuda.memory_reserved(self.gpu) / 1024**3
                            )
                        # MPS does not provide memory stats via PyTorch API
                        wandb.log(log_data)
                    running_loss, n_running_loss = 0, 0

        # Handle remaining accumulated gradients at end of epoch
        if accumulated_samples % grad_accum_steps != 0:
            if is_sam:
                # For SAM, we need to do the full two-step update with remaining gradients
                # This is handled by calling _sam_step with is_update_step=True
                pass  # Already handled in the loop
            else:
                if scaler is not None:
                    scaler.unscale_(optimizer)

                if clip_grad_norm > 0.0:
                    nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=clip_grad_norm, norm_type=2
                    )
                elif clip_grad_value > 0.0:
                    nn.utils.clip_grad_value_(
                        model.parameters(), clip_value=clip_grad_value
                    )

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()

        elapsed_time = time.time() - start
        for tp in loss_type.keys():
            logging.debug(
                f"loss[{tp}]={loss_type[tp] / num_type[tp]:.6f}={loss_type[tp]:.3f}/{num_type[tp]}"
            )
        print(
            f"Train Epoch: {epoch}\tLoss: {loss_total / num:.6f}\tTime: {elapsed_time:.3f}s"
        )

    def _sam_step(
        self,
        model: AbstractFold,
        optimizer: SAM,
        compute_loss: callable,
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
                optimizer.first_step(zero_grad=True, loss=loss)
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
                optimizer.second_step(zero_grad=False, loss=loss_perturbed)
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
        all_tp, all_tn, all_fp, all_fn = 0, 0, 0, 0
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
                                loss_fn["BPSEQ"](
                                    seqs[i : i + 1],
                                    vals["target"][i : i + 1],
                                    fname=fnames[i : i + 1],
                                )
                            )
                            # Get predictions for F1 calculation
                            _, _, bps = model(seqs[i : i + 1])
                            ref = vals["target"][i]
                            tp, tn, fp, fn = compare_bpseq(ref, bps[0])
                            all_tp += tp
                            all_tn += tn
                            all_fp += fp
                            all_fn += fn
                            # SHAPE consistency
                            if shape_paths is not None and sample_idx < len(shape_paths):
                                shape_data = read_shape_reactivity(shape_paths[sample_idx])
                                score, _, _ = pairwise_consistency(bps[0], shape_data)
                                if not math.isnan(score):
                                    shape_scores.append(score)
                            sample_idx += 1
                        elif vals["type"][i] == "SHAPE":
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
        sen = ppv = fval = mcc = 0.0
        has_metrics = all_tp + all_fn + all_fp > 0
        if has_metrics:
            sen, ppv, fval, mcc, _, _ = accuracy(all_tp, all_tn, all_fp, all_fn)

        if self.use_wandb:
            log_dict: dict[str, float] = {
                f"{metric_prefix}/loss": loss_total / num,
                f"{metric_prefix}/epoch": epoch,
            }
            if has_metrics:
                log_dict.update({
                    f"{metric_prefix}/f1": fval,
                    f"{metric_prefix}/sensitivity": sen,
                    f"{metric_prefix}/ppv": ppv,
                    f"{metric_prefix}/mcc": mcc,
                })
            if shape_scores:
                log_dict[f"{metric_prefix}/shape_consistency"] = (
                    sum(shape_scores) / len(shape_scores)
                )
            wandb.log(log_dict)

        msg = "Test[{}] Epoch: {}\tLoss: {:.6f}".format(
            metric_prefix, epoch, loss_total / num
        )
        if has_metrics:
            msg += "\tF1: {:.4f}\tSEN: {:.4f}\tPPV: {:.4f}\tMCC: {:.4f}".format(
                fval, sen, ppv, mcc
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
        torch.save(checkpoint, filename)

    def resume_checkpoint(
        self,
        filename: str,
        model: AbstractFold,
        optimizer: Optimizer,
        scheduler,
        shape_model: Optional[list[nn.Module]] = None,
        ema: Optional[EMA] = None,
        scaler: Optional[GradScaler] = None,
        gpu: int = -1,
    ) -> tuple[int, Optional[GradScaler]]:
        checkpoint = torch.load(filename)
        epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model_state_dict"])
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

        return epoch, scaler

    def build_optimizer(
        self,
        optimizer: str,
        model: AbstractFold,
        lr: float,
        l2_weight: float,
        shape_model: Optional[list[nn.Module]] = None,
        sam_type: Optional[str] = None,
        sam_rho: float = 0.05,
        sam_alpha: float = 0.1,
    ) -> Optimizer:
        optim_params = [
            {"params": model.parameters(), "lr": lr, "weight_decay": l2_weight},
        ]
        if shape_model is not None:
            for sm in shape_model:
                optim_params.append(
                    {"params": sm.parameters(), "lr": 0.001, "weight_decay": l2_weight}
                )

        # Define base optimizer classes and their kwargs
        base_optimizer_config = {
            "Adam": (Adam, {"amsgrad": False}),
            "AdamW": (AdamW, {"amsgrad": False}),
            "RMSprop": (RMSprop, {}),
            "SGD": (SGD, {"nesterov": True, "momentum": 0.9}),
            "ASGD": (ASGD, {}),
            "AdaBelief": (po.AdaBelief, {}),
            "Lion": (po.Lion, {}),
        }

        if optimizer not in base_optimizer_config:
            raise RuntimeError(f"not implemented: {optimizer}")

        base_optimizer_class, base_kwargs = base_optimizer_config[optimizer]

        # Apply SAM wrapper if requested
        if sam_type is None or sam_type == "None":
            # Create regular optimizer
            return base_optimizer_class(optim_params, **base_kwargs)
        elif sam_type == "SAM":
            return SAM(optim_params, base_optimizer_class, rho=sam_rho, **base_kwargs)
        elif sam_type == "ASAM":
            return ASAM(optim_params, base_optimizer_class, rho=sam_rho, **base_kwargs)
        elif sam_type == "GSAM":
            return GSAM(
                optim_params,
                base_optimizer_class,
                rho=sam_rho,
                alpha=sam_alpha,
                **base_kwargs,
            )
        else:
            raise RuntimeError(f"not implemented SAM type: {sam_type}")

    def build_loss_function(
        self, loss_func: str, model: AbstractFold, args: Namespace
    ) -> nn.Module:
        if loss_func == "hinge" or loss_func == "hinge_mix":
            from mxfold2.loss.structured_loss import StructuredLoss

            return StructuredLoss(
                model,
                loss_pos_paired=args.loss_pos_paired,
                loss_neg_paired=args.loss_neg_paired,
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

    def build_shape_loss_function(
        self,
        loss_func: str,
        model: AbstractFold,
        args: Namespace,
        shape_model: Optional[nn.Module] = None,
        lwf_model: Optional[AbstractFold] = None,
    ) -> nn.Module:
        if loss_func == "shape_nll":
            from mxfold2.loss.shape_nll_loss import ShapeNLLLoss

            return ShapeNLLLoss(
                model=model,
                shape_model=shape_model,
                perturb=args.shape_perturb,
                nu=args.shape_nu,
                l1_weight=args.l1_weight,
                l2_weight=args.l2_weight,
                lwf_model=lwf_model,
                lwf_weight=args.lwf_weight,
                sl_weight=args.score_loss_weight,
                shape_only=args.shape_only_training,
                pseudo_fy_weight=args.shape_pseudo_fy_weight,
                weight_schedule=args.weight_schedule,
                weight_schedule_start=args.weight_schedule_start,
                weight_schedule_end=args.weight_schedule_end,
            )

        elif loss_func == "shape_fy":
            from mxfold2.loss.shape_fy_loss import ShapeFenchelYoungLoss

            return ShapeFenchelYoungLoss(
                model,
                perturb=args.shape_perturb,
                shape_intercept=args.shape_intercept,
                shape_slope=args.shape_slope,
                l1_weight=args.l1_weight,
                l2_weight=args.l2_weight,
                sl_weight=args.score_loss_weight,
                weight_schedule=args.weight_schedule,
                weight_schedule_start=args.weight_schedule_start,
                weight_schedule_end=args.weight_schedule_end,
            )

        elif loss_func == "shape_rank":
            from mxfold2.loss.shape_rank_loss import ShapeRankLoss

            return ShapeRankLoss(
                model,
                margin=args.shape_margin,
                perturb=args.shape_perturb,
                nu=args.shape_nu,
                l1_weight=args.l1_weight,
                l2_weight=args.l2_weight,
                sl_weight=args.score_loss_weight,
                pseudo_fy_weight=args.shape_pseudo_fy_weight,
                weight_schedule=args.weight_schedule,
                weight_schedule_start=args.weight_schedule_start,
                weight_schedule_end=args.weight_schedule_end,
            )

        else:
            raise (ValueError(f"not implemented: {loss_func}"))

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

    def run(self, args: Namespace, conf: Optional[str] = None) -> None:
        self.disable_progress_bar = args.disable_progress_bar
        self.gpu = args.gpu
        loglevel = "INFO" if args.verbose else args.loglevel
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=getattr(logging, loglevel, None),
        )

        # Initialize wandb if project is specified
        self.use_wandb = args.wandb_project is not None
        if self.use_wandb:
            wandb_config = vars(args).copy()
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                tags=args.wandb_tags,
                config=wandb_config,
            )

        train_dataset = BPseqDataset(
            args.input, convert_t_to_u_flag=getattr(args, "convert_t_to_u", False)
        )
        n_train_samples = len(train_dataset)
        n_dataset_id = 0
        if args.json_dataset is not None:
            json_dataset = JsonDataset(
                args.json_dataset,
                convert_t_to_u_flag=getattr(args, "convert_t_to_u", False),
            )
            train_dataset = ConcatDataset([train_dataset, json_dataset])
            n_train_samples = len(train_dataset)
        if args.ribonanza is not None:
            ribonanza_dataset = RibonanzaDataset(
                args.ribonanza,
                offset=n_dataset_id,
                convert_t_to_u_flag=getattr(args, "convert_t_to_u", False),
            )
            n_dataset_id += len(ribonanza_dataset.dataset_id)
            train_dataset = ConcatDataset([train_dataset, ribonanza_dataset])
        if args.shape is not None:
            shape_dataset = [
                ShapeDataset(
                    s,
                    i + n_dataset_id,
                    convert_t_to_u_flag=getattr(args, "convert_t_to_u", False),
                )
                for i, s in enumerate(args.shape)
            ]
            n_dataset_id += len(shape_dataset)
            train_dataset = ConcatDataset([train_dataset] + shape_dataset)
        if args.json_shape_dataset is not None:
            json_shape_dataset = JsonShapeDataset(
                args.json_shape_dataset,
                offset=n_dataset_id,
                convert_t_to_u_flag=getattr(args, "convert_t_to_u", False),
            )
            n_dataset_id += len(json_shape_dataset.dataset_id)
            train_dataset = ConcatDataset([train_dataset, json_shape_dataset])
        if args.extra_dataset is not None:
            extra_dataset = [
                BPseqDataset(
                    s, convert_t_to_u_flag=getattr(args, "convert_t_to_u", False)
                )
                for s in args.extra_dataset
            ]
            train_dataset = ConcatDataset([train_dataset] + extra_dataset)
        n_shape_samples = len(train_dataset) - n_train_samples

        # Create generator for reproducible shuffling
        generator = torch.Generator()
        if args.seed >= 0:
            generator.manual_seed(args.seed)

        sampler = None
        if args.downsampling > 0.0:
            weights = [1.0 / n_train_samples] * n_train_samples + [
                args.downsampling / n_shape_samples
            ] * n_shape_samples
            n_train_samples = int(n_train_samples * (1.0 + args.downsampling) + 0.5)
            sampler = WeightedRandomSampler(
                weights, n_train_samples, replacement=False, generator=generator
            )

        train_loader = DataLoader(
            train_dataset,
            sampler=sampler,
            batch_size=1,
            shuffle=True if sampler is None else False,
            generator=generator,
        )  # works well only for batch_size=1!!

        test_loaders: list[tuple[DataLoader, list[str] | None]] = []
        if args.test_input is not None:
            test_shape_lists = getattr(args, "test_shape", None) or []
            for idx, test_path in enumerate(args.test_input):
                test_dataset = BPseqDataset(
                    test_path,
                    convert_t_to_u_flag=getattr(args, "convert_t_to_u", False),
                )
                test_loader = DataLoader(
                    test_dataset, batch_size=1, shuffle=False
                )  # works well only for batch_size=1!!
                shape_paths = None
                if idx < len(test_shape_lists) and test_shape_lists[idx].lower() != "none":
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

        shape_model = None
        if (
            args.shape is not None
            or args.ribonanza is not None
            or args.json_shape_dataset is not None
        ):
            shape_model = [self.build_shape_model(args) for _ in range(n_dataset_id)]

        if args.init_param != "":
            init_param = Path(args.init_param)
            if not init_param.exists() and conf is not None:
                init_param = Path(conf) / init_param
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
            model.load_state_dict(p)

        lwf_model = None
        if args.lwf_weight > 0.0:
            lwf_model, _ = self.build_model(args)
            lwf_model.load_state_dict(model.state_dict())
            lwf_model.eval()

        device, self.device_type = self.get_device(args.gpu)
        model.to(device)
        if shape_model is not None:
            for sm in shape_model:
                sm.to(device)
        if lwf_model is not None:
            lwf_model.to(device)

        torch.set_num_threads(args.threads)
        interface.set_num_threads(args.threads)

        optimizer = self.build_optimizer(
            args.optimizer,
            model,
            args.lr,
            args.l2_weight,
            shape_model=shape_model,
            sam_type=getattr(args, "sam_type", None),
            sam_rho=getattr(args, "sam_rho", 0.05),
            sam_alpha=getattr(args, "sam_alpha", 0.1),
        )

        loss_fn = {
            "BPSEQ": self.build_loss_function(args.loss_func, model, args),
            "SHAPE": self.build_shape_loss_function(
                args.shape_loss_func,
                model,
                args,
                shape_model=shape_model,
                lwf_model=lwf_model,
            ),
        }
        loss_weight = {"BPSEQ": 1.0, "SHAPE": args.shape_loss_weight}
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
        if args.resume is not None:
            checkpoint_epoch, resumed_scaler = self.resume_checkpoint(
                args.resume,
                model,
                optimizer,
                scheduler,
                shape_model,
                ema,
                scaler,
                args.gpu,
            )
            if resumed_scaler is not None:
                scaler = resumed_scaler
                use_amp = True
                logging.info("Resumed with Automatic Mixed Precision (AMP) training")

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
                loss_fn["SHAPE"].set_epoch_info(epoch, args.epochs)

            epoch_start = time.time()
            self.train(
                epoch,
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                data_loader=train_loader,
                n_dataset=n_train_samples,
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
                eval_model = (
                    swa_model
                    or (ema.shadow if ema and epoch > ema_start else None)
                    or model
                )
                for test_idx, (t_loader, t_shape_paths) in enumerate(test_loaders):
                    prefix = "test" if len(test_loaders) == 1 else f"test_{test_idx}"
                    self.test(
                        epoch,
                        model=eval_model,
                        loss_fn=loss_fn,
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

            if args.output_dir is not None:
                self.save_checkpoint(
                    args.output_dir,
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
            "input", type=str, help="Training data of the list of BPSEQ-formatted files"
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
        gparser.add_argument(
            "--loss-pos-unpaired",
            type=float,
            default=0.0,
            help="the penalty for positive unpaired bases for loss augmentation (default: 0)",
        )
        gparser.add_argument(
            "--loss-neg-unpaired",
            type=float,
            default=0.0,
            help="the penalty for negative unpaired bases for loss augmentation (default: 0)",
        )
        gparser.add_argument(
            "--shape-loss-func",
            choices=("shape_nll", "shape_fy", "shape_rank"),
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
