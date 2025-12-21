"""CLI entry point for mxfold2 hyperparameter optimization."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.trial import TrialState

from .objective import MXFold2Objective
from .search_space import SearchSpaceConfig


class BestModelSaveCallback:
    """Callback to save the best model parameters."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.best_value: float = float("-inf")

    def __call__(
        self, study: optuna.Study, trial: optuna.trial.FrozenTrial
    ) -> None:
        if trial.state == TrialState.COMPLETE and trial.value is not None:
            if trial.value > self.best_value:
                self.best_value = trial.value
                params_path = self.output_dir / "best_params.json"
                with open(params_path, "w") as f:
                    json.dump(
                        {
                            "trial_number": trial.number,
                            "f1_score": trial.value,
                            "params": trial.params,
                        },
                        f,
                        indent=2,
                    )


class EarlyStoppingCallback:
    """Callback to stop optimization if no improvement for N trials."""

    def __init__(self, patience: int = 50) -> None:
        self.patience = patience
        self.best_value: float = float("-inf")
        self.no_improvement_count: int = 0

    def __call__(
        self, study: optuna.Study, trial: optuna.trial.FrozenTrial
    ) -> None:
        if trial.state == TrialState.COMPLETE and trial.value is not None:
            if trial.value > self.best_value:
                self.best_value = trial.value
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1

            if self.no_improvement_count >= self.patience:
                logging.info(
                    f"Stopping optimization: no improvement for {self.patience} trials"
                )
                study.stop()


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for the HPO CLI."""
    parser = argparse.ArgumentParser(
        description="MXFold2 Hyperparameter Optimization with Optuna"
    )

    # Data settings
    data_group = parser.add_argument_group("Data settings")
    data_group.add_argument(
        "--train-data",
        type=str,
        required=True,
        help="Path to training data list file",
    )
    data_group.add_argument(
        "--val-data",
        type=str,
        required=True,
        help="Path to validation data list file",
    )
    data_group.add_argument(
        "--shape-data",
        type=str,
        nargs="*",
        default=None,
        help="Paths to SHAPE data list files",
    )

    # Optimization settings
    opt_group = parser.add_argument_group("Optimization settings")
    opt_group.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of trials (default: 100)",
    )
    opt_group.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of epochs per trial (default: 30)",
    )
    opt_group.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout in seconds (default: None)",
    )
    opt_group.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stopping patience (trials without improvement, default: 50)",
    )

    # Pruning settings
    pruning_group = parser.add_argument_group("Pruning settings")
    pruning_group.add_argument(
        "--no-pruning",
        action="store_true",
        help="Disable pruning",
    )
    pruning_group.add_argument(
        "--pruning-warmup",
        type=int,
        default=10,
        help="Number of warmup epochs before pruning (default: 10)",
    )
    pruning_group.add_argument(
        "--n-startup-trials",
        type=int,
        default=5,
        help="Number of startup trials before pruning (default: 5)",
    )

    # Execution settings
    exec_group = parser.add_argument_group("Execution settings")
    exec_group.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID (-1 for CPU, default: 0)",
    )
    exec_group.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of threads (default: 4)",
    )
    exec_group.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed (default: 1234)",
    )

    # Storage settings
    storage_group = parser.add_argument_group("Storage settings")
    storage_group.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL (e.g., sqlite:///hpo.db)",
    )
    storage_group.add_argument(
        "--study-name",
        type=str,
        default="mxfold2-hpo",
        help="Study name (default: mxfold2-hpo)",
    )
    storage_group.add_argument(
        "--output-dir",
        type=str,
        default="./hpo_results",
        help="Output directory (default: ./hpo_results)",
    )

    # Search space settings
    search_group = parser.add_argument_group("Search space settings")
    search_group.add_argument(
        "--search-space-config",
        type=str,
        default=None,
        help="YAML config file for search space",
    )
    search_group.add_argument(
        "--fixed-params",
        type=str,
        default=None,
        help="JSON string or file path for fixed parameters (e.g., '{\"num_filters\": 64}')",
    )

    # Logging settings
    log_group = parser.add_argument_group("Logging settings")
    log_group.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser


def save_results(
    study: optuna.Study,
    output_dir: Path,
) -> None:
    """Save optimization results to files.

    Args:
        study: Completed Optuna study
        output_dir: Output directory
    """
    # Save summary
    results = {
        "best_trial_number": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
        "n_completed": len(
            [t for t in study.trials if t.state == TrialState.COMPLETE]
        ),
        "n_pruned": len(
            [t for t in study.trials if t.state == TrialState.PRUNED]
        ),
    }

    with open(output_dir / "optimization_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Try to generate visualizations
    try:
        from optuna.visualization import (
            plot_optimization_history,
            plot_param_importances,
            plot_parallel_coordinate,
        )

        fig1 = plot_optimization_history(study)
        fig1.write_html(str(output_dir / "optimization_history.html"))

        # Only plot param importances if there are enough completed trials
        completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if len(completed) >= 2:
            fig2 = plot_param_importances(study)
            fig2.write_html(str(output_dir / "param_importances.html"))

            fig3 = plot_parallel_coordinate(study)
            fig3.write_html(str(output_dir / "parallel_coordinate.html"))

    except Exception as e:
        logging.warning(f"Failed to generate visualizations: {e}")

    # Save all trials as CSV
    try:
        df = study.trials_dataframe()
        df.to_csv(output_dir / "trials.csv", index=False)
    except Exception as e:
        logging.warning(f"Failed to save trials CSV: {e}")


def main() -> None:
    """Main entry point for the HPO CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=log_level,
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load search space configuration
    search_space_config: Optional[SearchSpaceConfig] = None
    if args.search_space_config:
        search_space_config = SearchSpaceConfig.from_yaml(args.search_space_config)
        logging.info(f"Loaded search space config from {args.search_space_config}")

    # Load fixed parameters
    fixed_params: Optional[dict] = None
    if args.fixed_params:
        # Try to parse as JSON string first
        try:
            fixed_params = json.loads(args.fixed_params)
        except json.JSONDecodeError:
            # If not JSON, try to load as file
            fixed_params_path = Path(args.fixed_params)
            if fixed_params_path.exists():
                with open(fixed_params_path) as f:
                    fixed_params = json.load(f)
            else:
                raise ValueError(f"Invalid fixed-params: {args.fixed_params}")
        logging.info(f"Fixed parameters: {fixed_params}")

    # Configure sampler
    sampler = TPESampler(
        seed=args.seed,
        multivariate=True,
        group=True,
    )

    # Configure pruner
    if args.no_pruning:
        pruner = optuna.pruners.NopPruner()
    else:
        pruner = MedianPruner(
            n_startup_trials=args.n_startup_trials,
            n_warmup_steps=args.pruning_warmup,
            interval_steps=1,
            n_min_trials=3,
        )

    # Create or load study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=True,
    )

    logging.info(f"Study name: {args.study_name}")
    logging.info(f"Storage: {args.storage or 'in-memory'}")
    logging.info(f"Output directory: {output_dir}")

    # Create objective function
    objective = MXFold2Objective(
        train_data=args.train_data,
        val_data=args.val_data,
        shape_data=args.shape_data,
        epochs=args.epochs,
        gpu=args.gpu,
        threads=args.threads,
        seed=args.seed,
        search_space_config=search_space_config,
        fixed_params=fixed_params,
    )

    # Setup callbacks
    callbacks = [
        BestModelSaveCallback(output_dir),
        EarlyStoppingCallback(patience=args.patience),
    ]

    # Run optimization
    print("\n" + "=" * 60)
    print("Starting Hyperparameter Optimization")
    print("=" * 60)
    print(f"  Trials: {args.n_trials}")
    print(f"  Epochs per trial: {args.epochs}")
    print(f"  GPU: {args.gpu}")
    print(f"  Pruning: {'disabled' if args.no_pruning else 'enabled'}")
    if fixed_params:
        print(f"  Fixed params: {list(fixed_params.keys())}")
    print("=" * 60 + "\n")

    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        callbacks=callbacks,
        show_progress_bar=True,
    )

    # Print results
    print("\n" + "=" * 60)
    print("Optimization Completed!")
    print("=" * 60)
    print(f"  Best trial: {study.best_trial.number}")
    print(f"  Best F1 score: {study.best_value:.4f}")
    print("\n  Best parameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    print("=" * 60)

    # Save results
    save_results(study, output_dir)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
