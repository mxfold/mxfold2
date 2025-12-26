"""HPO Trainer that extends the base Train class with F1 evaluation."""

from __future__ import annotations

import gc
import logging
import random
import time
from argparse import Namespace
from collections import defaultdict
from typing import Any, Callable, Optional, cast

import torch
import torch.backends.cudnn
import torch.nn as nn
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from mxfold2 import interface
from mxfold2.compbpseq import accuracy, compare_bpseq
from mxfold2.dataset import FastaDataset
from mxfold2.fold.fold import AbstractFold
from mxfold2.sam import SAM, GSAM
from mxfold2.train import Train


class HPOTrainer(Train):
    """Trainer class for hyperparameter optimization with F1 score evaluation."""

    def __init__(self) -> None:
        super().__init__()
        self.best_f1: float = 0.0
        self.epoch_f1_scores: list[float] = []

    def train_with_validation(
        self,
        args: Namespace,
        train_loader: DataLoader,
        val_loader: DataLoader,
        pruning_callback: Optional[Callable[[int, float], bool]] = None,
    ) -> tuple[AbstractFold, float]:
        """Train the model and evaluate F1 score on validation set.

        Args:
            args: Training arguments
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            pruning_callback: Optional callback function for pruning.
                Takes (epoch, f1_score) and returns True if should prune.

        Returns:
            Tuple of (trained model, best F1 score)
        """
        self.disable_progress_bar = True
        self.gpu = args.gpu
        self.best_f1 = 0.0
        self.epoch_f1_scores = []

        # Set random seed for reproducibility
        if args.seed >= 0:
            torch.manual_seed(args.seed)
            random.seed(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Build model
        model, config = self.build_model(args)

        # Build shape model if needed
        shape_model = None
        if args.shape is not None and args.shape_loss_func == "shape_nll":
            shape_model = [self.build_shape_model(args) for _ in args.shape]

        # Move to GPU if available
        if args.gpu >= 0:
            model.to(torch.device("cuda", args.gpu))
            if shape_model is not None:
                for sm in shape_model:
                    sm.to(torch.device("cuda", args.gpu))

        # Set threads
        torch.set_num_threads(args.threads)
        interface.set_num_threads(args.threads)

        # Build optimizer
        optimizer = self.build_optimizer(
            args.optimizer, model, args.lr, args.l2_weight,
            shape_model=shape_model,
            sam_type=getattr(args, 'sam_type', None),
            sam_rho=getattr(args, 'sam_rho', 0.05),
            sam_alpha=getattr(args, 'sam_alpha', 0.1)
        )

        # Build loss functions
        loss_fn = {
            "BPSEQ": self.build_loss_function(args.loss_func, model, args),
            "SHAPE": self.build_shape_loss_function(
                args.shape_loss_func, model, args, shape_model=shape_model
            ),
        }
        loss_weight = {"BPSEQ": 1.0, "SHAPE": args.shape_loss_weight}

        # Build scheduler
        scheduler = self.build_scheduler(args.scheduler, optimizer, args)

        # Initialize GradScaler for mixed precision
        scaler = None
        use_amp = False
        if hasattr(args, "use_amp") and args.use_amp and args.gpu >= 0:
            use_amp = True
            scaler = GradScaler(f"cuda:{args.gpu}")

        # Training loop
        for epoch in range(1, args.epochs + 1):
            # Set epoch info for loss functions (for weight scheduling)
            for key in loss_fn:
                if hasattr(loss_fn[key], "set_epoch_info"):
                    loss_fn[key].set_epoch_info(epoch, args.epochs)

            # Train one epoch
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
                grad_accum_steps=getattr(args, 'grad_accum_steps', 1),
            )

            # Update scheduler
            if scheduler is not None:
                scheduler.step()

            # Evaluate F1 score on validation set
            f1_score = self.evaluate_f1(model, val_loader, args.gpu)
            self.epoch_f1_scores.append(f1_score)

            if f1_score > self.best_f1:
                self.best_f1 = f1_score

            logging.info(f"Epoch {epoch}: F1 = {f1_score:.4f}, Best F1 = {self.best_f1:.4f}")

            # Check pruning
            if pruning_callback is not None:
                should_prune = pruning_callback(epoch, f1_score)
                if should_prune:
                    logging.info(f"Trial pruned at epoch {epoch}")
                    break

        return model, self.best_f1

    def train(
        self,
        epoch: int,
        model: AbstractFold,
        optimizer: Optimizer,
        loss_fn: nn.Module | dict[str, nn.Module],
        data_loader: DataLoader[tuple[str, str, dict[str, torch.Tensor]]],
        loss_weight: dict[str, float] | None = None,
        clip_grad_value: float = 0.0,
        clip_grad_norm: float = 0.0,
        scaler: Optional[GradScaler] = None,
        use_amp: bool = False,
        grad_accum_steps: int = 1,
    ) -> None:
        """Train for one epoch (simplified version without progress bar and wandb)."""
        if loss_weight is None:
            loss_weight = defaultdict(lambda: 1.0)

        model.train()
        if not isinstance(loss_fn, dict):
            loss_fn = {"BPSEQ": loss_fn}

        n_dataset = len(cast(FastaDataset, data_loader.dataset))
        loss_total, num = 0.0, 0
        start = time.time()

        # Check if using SAM optimizer
        is_sam = isinstance(optimizer, SAM)

        # Gradient accumulation counter
        accumulated_samples = 0
        optimizer.zero_grad()  # Zero gradients once at the start

        for fnames, seqs, vals in data_loader:
            self.step += 1
            n_batch = len(seqs)
            for i in range(n_batch):
                accumulated_samples += 1
                is_update_step = (accumulated_samples % grad_accum_steps == 0)

                # Define loss computation function for SAM
                def compute_loss():
                    with autocast(
                        device_type="cuda", dtype=torch.float16, enabled=use_amp
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
                        model, optimizer, compute_loss,
                        clip_grad_value, clip_grad_norm,
                        scaler, use_amp,
                        grad_accum_steps=grad_accum_steps,
                        is_update_step=is_update_step
                    )
                    if is_update_step:
                        optimizer.zero_grad()
                else:
                    # Standard optimization with gradient accumulation
                    loss = compute_loss()

                    # Scale loss for gradient accumulation
                    scaled_loss = loss / grad_accum_steps
                    if scaler is not None:
                        scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()

                    # Only update on accumulation boundary
                    if is_update_step:
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

                loss_total += loss.item()

            num += n_batch

        # Handle remaining accumulated gradients at end of epoch
        if accumulated_samples % grad_accum_steps != 0:
            if not is_sam:
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
        logging.debug(
            f"Train Epoch: {epoch}\tLoss: {loss_total / num:.6f}\tTime: {elapsed_time:.3f}s"
        )

    def evaluate_f1(
        self,
        model: AbstractFold,
        val_loader: DataLoader,
        gpu: int = -1,
    ) -> float:
        """Evaluate F1 score on the validation set.

        Args:
            model: The model to evaluate
            val_loader: DataLoader for validation data
            gpu: GPU device ID (-1 for CPU)

        Returns:
            Macro F1 score over all validation samples
        """
        model.eval()
        total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0

        with torch.no_grad():
            for headers, seqs, vals in val_loader:
                # Get predictions
                _, _, bps = model(seqs)

                for bp, ref in zip(bps, vals["target"]):
                    # Convert tensors to lists if needed
                    if isinstance(ref, torch.Tensor):
                        ref = ref.tolist()
                    if isinstance(bp, torch.Tensor):
                        bp = bp.tolist()

                    tp, tn, fp, fn = compare_bpseq(ref, bp)
                    total_tp += tp
                    total_tn += tn
                    total_fp += fp
                    total_fn += fn

        # Calculate macro F1 score
        _, _, f1, _ = accuracy(total_tp, total_tn, total_fp, total_fn)
        return f1

    @staticmethod
    def cleanup_gpu_memory() -> None:
        """Clean up GPU memory after a trial."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
