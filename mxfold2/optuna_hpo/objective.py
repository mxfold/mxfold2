"""Optuna objective function for mxfold2 hyperparameter optimization."""

from __future__ import annotations

import logging
import math
from argparse import Namespace
from typing import Any, Optional

import optuna
import torch
from torch.utils.data import ConcatDataset, DataLoader

from mxfold2.dataset import BPseqDataset, ShapeDataset

from .search_space import SearchSpaceConfig, suggest_hyperparameters
from .trainer import HPOTrainer


class MXFold2Objective:
    """Optuna objective function for mxfold2 hyperparameter optimization.

    This class encapsulates the training and evaluation logic for a single
    Optuna trial. It handles:
    - Hyperparameter suggestion based on the search space configuration
    - Building the training arguments (Namespace)
    - Training the model and evaluating F1 score
    - Reporting intermediate results for pruning
    """

    def __init__(
        self,
        train_data: str,
        val_data: str,
        shape_data: Optional[list[str]] = None,
        epochs: int = 30,
        gpu: int = 0,
        threads: int = 4,
        seed: int = 1234,
        search_space_config: Optional[SearchSpaceConfig] = None,
        fixed_params: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize the objective function.

        Args:
            train_data: Path to training data list file
            val_data: Path to validation data list file
            shape_data: Optional list of paths to SHAPE data files
            epochs: Number of epochs per trial
            gpu: GPU device ID (-1 for CPU)
            threads: Number of threads
            seed: Random seed (incremented per trial for diversity)
            search_space_config: Search space configuration
            fixed_params: Fixed parameters that won't be optimized
        """
        self.train_data = train_data
        self.val_data = val_data
        self.shape_data = shape_data
        self.epochs = epochs
        self.gpu = gpu
        self.threads = threads
        self.base_seed = seed
        self.search_space_config = search_space_config or SearchSpaceConfig()
        self.fixed_params = fixed_params or {}

        # Pre-create data loaders
        self._prepare_dataloaders()

    def _prepare_dataloaders(self) -> None:
        """Prepare data loaders for training and validation."""
        # Training data
        train_dataset = BPseqDataset(self.train_data)
        if self.shape_data:
            shape_datasets = [
                ShapeDataset(s, i) for i, s in enumerate(self.shape_data)
            ]
            train_dataset = ConcatDataset([train_dataset] + shape_datasets)

        self.train_loader = DataLoader(
            train_dataset, batch_size=1, shuffle=True
        )

        # Validation data
        val_dataset = BPseqDataset(self.val_data)
        self.val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    def _create_args(
        self, suggested_params: dict[str, Any], trial_number: int
    ) -> Namespace:
        """Create a Namespace object from suggested parameters.

        Args:
            suggested_params: Dictionary of suggested hyperparameters
            trial_number: Trial number (used for seed)

        Returns:
            Namespace object with all training arguments
        """
        # Merge with fixed parameters
        params = {**suggested_params, **self.fixed_params}

        # Create filter configurations
        # If lists are provided in fixed_params, use them directly
        # Otherwise, create 8-layer pattern from single values
        if "num_filters_list" in self.fixed_params:
            num_filters = self.fixed_params["num_filters_list"]
        else:
            num_filters = [params["num_filters"]] * 8

        if "filter_size_list" in self.fixed_params:
            filter_size = self.fixed_params["filter_size_list"]
        else:
            filter_size = [5, 3, 5, 3, 5, 3, 5, 3]

        if "num_paired_filters_list" in self.fixed_params:
            num_paired_filters = self.fixed_params["num_paired_filters_list"]
        else:
            num_paired_filters = [params["num_filters"]] * 8

        if "paired_filter_size_list" in self.fixed_params:
            paired_filter_size = self.fixed_params["paired_filter_size_list"]
        else:
            paired_filter_size = [5, 3, 5, 3, 5, 3, 5, 3]
            
        if "resnet_every_n" in self.fixed_params:
            resnet_every_n = self.fixed_params["resnet_every_n"]
        else:
            resnet_every_n = 1

        args = Namespace(
            # Basic settings
            gpu=self.gpu,
            threads=self.threads,
            seed=self.base_seed + trial_number,  # Different seed per trial
            epochs=self.epochs,
            verbose=False,
            disable_progress_bar=True,
            loglevel="WARNING",
            # Model settings
            model="Mix",
            fold="Zuker",
            max_helix_length=30,
            beam_size=100,
            # Learning parameters
            optimizer=params["optimizer"],
            lr=params["lr"],
            scheduler=params["scheduler"],
            scheduler_step_size=5,
            scheduler_gamma=0.95,
            clip_grad_norm=params["clip_grad_norm"],
            clip_grad_value=0.0,
            # Model architecture
            embed_size=params["embed_size"],
            num_filters=num_filters,
            filter_size=filter_size,
            pool_size=None,
            dilation=0,
            num_lstm_layers=params["num_lstm_layers"],
            num_lstm_units=params["num_lstm_units"],
            num_transformer_layers=0,
            num_transformer_hidden_units=2048,
            num_transformer_att=8,
            num_hidden_units=None,
            num_paired_filters=num_paired_filters,
            paired_filter_size=paired_filter_size,
            resnet_every_n=resnet_every_n,
            dropout_rate=params["dropout_rate"],
            fc_dropout_rate=params["fc_dropout_rate"],
            num_att=params["num_att"],
            pair_join=params["pair_join"],
            no_split_lr=False,
            paired_opt="symmetric",
            mix_type="average",
            weight_turner=None,
            weight_positional=None,
            additional_params=True,
            weight_schedule=params.get("weight_schedule", "none"),
            weight_schedule_start=params.get("weight_schedule_start", 1),
            weight_schedule_end=params.get("weight_schedule_end", None),
            # Loss settings
            loss_func=params["loss_func"],
            perturb=params["perturb"],
            nu=params["nu"],
            l1_weight=0.0,
            l2_weight=params["l2_weight"],
            score_loss_weight=params["score_loss_weight"],
            loss_pos_paired=0.5,
            loss_neg_paired=0.005,
            loss_pos_unpaired=0.0,
            loss_neg_unpaired=0.0,
            # SHAPE settings
            shape=self.shape_data,
            shape_model="Wu",
            shape_loss_func=params["shape_loss_func"],
            shape_perturb=params["shape_perturb"],
            shape_nu=params["shape_nu"],
            shape_margin=params["shape_margin"],
            shape_pseudo_fy_weight=params["shape_pseudo_fy_weight"],
            shape_intercept=-0.8,
            shape_slope=2.6,
            shape_loss_weight=1.0,
            # Other settings
            param=None,
            init_param="",
            use_amp=False,
            swa=False,
            ema=False,
        )

        return args

    def __call__(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna optimization.

        Args:
            trial: Optuna trial object

        Returns:
            F1 score on the validation set (to be maximized)
        """
        # Suggest hyperparameters
        suggested_params = suggest_hyperparameters(trial, self.search_space_config)

        # Create training arguments
        args = self._create_args(suggested_params, trial.number)

        # Create trainer
        trainer = HPOTrainer()

        # Pruning callback
        def pruning_callback(epoch: int, f1_score: float) -> bool:
            trial.report(f1_score, epoch)
            return trial.should_prune()

        try:
            # Train and evaluate
            _, best_f1 = trainer.train_with_validation(
                args,
                self.train_loader,
                self.val_loader,
                pruning_callback=pruning_callback,
            )

            # Handle NaN or invalid scores
            if math.isnan(best_f1) or math.isinf(best_f1):
                logging.warning(f"Trial {trial.number} returned invalid F1 score")
                raise optuna.TrialPruned()

            return best_f1

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logging.warning(f"Trial {trial.number} ran out of GPU memory")
                HPOTrainer.cleanup_gpu_memory()
                raise optuna.TrialPruned()
            raise

        finally:
            # Clean up GPU memory
            HPOTrainer.cleanup_gpu_memory()
