"""Search space definition for hyperparameter optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import optuna
import yaml


@dataclass
class SearchSpaceConfig:
    """Configuration for hyperparameter search space."""

    # Learning parameters
    optimizer: list[str] = field(
        default_factory=lambda: ["Adam", "AdamW", "RMSprop", "SGD", "AdaBelief", "Lion"]
    )
    lr_min: float = 1e-4
    lr_max: float = 1e-2
    lr_log: bool = True
    scheduler: list[str] = field(
        default_factory=lambda: ["None", "CyclicLR", "CosineAnnealingLR"]
    )
    dropout_rate_min: float = 0.0
    dropout_rate_max: float = 0.5
    fc_dropout_rate_min: float = 0.0
    fc_dropout_rate_max: float = 0.5
    l2_weight_min: float = 1e-5
    l2_weight_max: float = 0.1
    l2_weight_log: bool = True
    clip_grad_norm_min: float = 0.0
    clip_grad_norm_max: float = 5.0

    # Model architecture
    embed_size: list[int] = field(default_factory=lambda: [0, 32, 64, 128])
    num_filters_min: int = 32
    num_filters_max: int = 128
    num_filters_step: int = 16
    filter_size: list[int] = field(default_factory=lambda: [3, 5, 7])
    num_lstm_layers_min: int = 0
    num_lstm_layers_max: int = 3
    num_lstm_units_min: int = 16
    num_lstm_units_max: int = 128
    num_lstm_units_step: int = 16
    num_att: list[int] = field(default_factory=lambda: [0, 4, 8])
    pair_join: list[str] = field(default_factory=lambda: ["cat", "add", "mul"])

    # Loss function parameters
    loss_func: list[str] = field(default_factory=lambda: ["hinge", "fy", "f1"])
    perturb_min: float = 0.1
    perturb_max: float = 1.0
    nu_min: float = 0.01
    nu_max: float = 1.0
    nu_log: bool = True
    score_loss_weight_min: float = 0.0
    score_loss_weight_max: float = 0.5

    # SHAPE loss parameters
    shape_loss_func: list[str] = field(
        default_factory=lambda: ["shape_nll", "shape_fy", "shape_rank"]
    )
    shape_perturb_min: float = 0.1
    shape_perturb_max: float = 1.0
    shape_nu_min: float = 0.1
    shape_nu_max: float = 10.0
    shape_nu_log: bool = True
    shape_margin_min: float = 0.0
    shape_margin_max: float = 1.0
    shape_pseudo_fy_weight_min: float = 0.0
    shape_pseudo_fy_weight_max: float = 1.0

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SearchSpaceConfig":
        """Load configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        # Flatten nested dict structure from YAML
        flat_data = {}
        for key, value in data.items():
            if isinstance(value, dict):
                # Handle range specifications like {min: 0.1, max: 1.0}
                if "min" in value:
                    flat_data[f"{key}_min"] = value["min"]
                if "max" in value:
                    flat_data[f"{key}_max"] = value["max"]
                if "log" in value:
                    flat_data[f"{key}_log"] = value["log"]
                if "step" in value:
                    flat_data[f"{key}_step"] = value["step"]
            else:
                flat_data[key] = value

        return cls(**flat_data)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        data = {
            "optimizer": self.optimizer,
            "lr": {"min": self.lr_min, "max": self.lr_max, "log": self.lr_log},
            "scheduler": self.scheduler,
            "dropout_rate": {"min": self.dropout_rate_min, "max": self.dropout_rate_max},
            "fc_dropout_rate": {"min": self.fc_dropout_rate_min, "max": self.fc_dropout_rate_max},
            "l2_weight": {"min": self.l2_weight_min, "max": self.l2_weight_max, "log": self.l2_weight_log},
            "clip_grad_norm": {"min": self.clip_grad_norm_min, "max": self.clip_grad_norm_max},
            "embed_size": self.embed_size,
            "num_filters": {"min": self.num_filters_min, "max": self.num_filters_max, "step": self.num_filters_step},
            "filter_size": self.filter_size,
            "num_lstm_layers": {"min": self.num_lstm_layers_min, "max": self.num_lstm_layers_max},
            "num_lstm_units": {"min": self.num_lstm_units_min, "max": self.num_lstm_units_max, "step": self.num_lstm_units_step},
            "num_att": self.num_att,
            "pair_join": self.pair_join,
            "loss_func": self.loss_func,
            "perturb": {"min": self.perturb_min, "max": self.perturb_max},
            "nu": {"min": self.nu_min, "max": self.nu_max, "log": self.nu_log},
            "score_loss_weight": {"min": self.score_loss_weight_min, "max": self.score_loss_weight_max},
            "shape_loss_func": self.shape_loss_func,
            "shape_perturb": {"min": self.shape_perturb_min, "max": self.shape_perturb_max},
            "shape_nu": {"min": self.shape_nu_min, "max": self.shape_nu_max, "log": self.shape_nu_log},
            "shape_margin": {"min": self.shape_margin_min, "max": self.shape_margin_max},
            "shape_pseudo_fy_weight": {"min": self.shape_pseudo_fy_weight_min, "max": self.shape_pseudo_fy_weight_max},
        }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def suggest_hyperparameters(
    trial: optuna.Trial, config: SearchSpaceConfig
) -> dict[str, Any]:
    """Suggest hyperparameters from the search space using an Optuna trial.

    Args:
        trial: Optuna trial object
        config: Search space configuration

    Returns:
        Dictionary of suggested hyperparameters
    """
    params: dict[str, Any] = {}

    # Learning parameters
    params["optimizer"] = trial.suggest_categorical("optimizer", config.optimizer)
    params["lr"] = trial.suggest_float(
        "lr", config.lr_min, config.lr_max, log=config.lr_log
    )
    params["scheduler"] = trial.suggest_categorical("scheduler", config.scheduler)
    params["dropout_rate"] = trial.suggest_float(
        "dropout_rate", config.dropout_rate_min, config.dropout_rate_max
    )
    params["fc_dropout_rate"] = trial.suggest_float(
        "fc_dropout_rate", config.fc_dropout_rate_min, config.fc_dropout_rate_max
    )
    params["l2_weight"] = trial.suggest_float(
        "l2_weight", config.l2_weight_min, config.l2_weight_max, log=config.l2_weight_log
    )
    params["clip_grad_norm"] = trial.suggest_float(
        "clip_grad_norm", config.clip_grad_norm_min, config.clip_grad_norm_max
    )

    # Model architecture
    params["embed_size"] = trial.suggest_categorical("embed_size", config.embed_size)
    params["num_filters"] = trial.suggest_int(
        "num_filters",
        config.num_filters_min,
        config.num_filters_max,
        step=config.num_filters_step,
    )
    params["filter_size"] = trial.suggest_categorical("filter_size", config.filter_size)
    params["num_lstm_layers"] = trial.suggest_int(
        "num_lstm_layers", config.num_lstm_layers_min, config.num_lstm_layers_max
    )

    # Conditional: num_lstm_units only when num_lstm_layers > 0
    if params["num_lstm_layers"] > 0:
        params["num_lstm_units"] = trial.suggest_int(
            "num_lstm_units",
            config.num_lstm_units_min,
            config.num_lstm_units_max,
            step=config.num_lstm_units_step,
        )
    else:
        params["num_lstm_units"] = 0

    params["num_att"] = trial.suggest_categorical("num_att", config.num_att)
    params["pair_join"] = trial.suggest_categorical("pair_join", config.pair_join)

    # Loss function parameters
    params["loss_func"] = trial.suggest_categorical("loss_func", config.loss_func)
    params["perturb"] = trial.suggest_float(
        "perturb", config.perturb_min, config.perturb_max
    )

    # Conditional: nu is more important for f1 loss
    if params["loss_func"] == "f1":
        params["nu"] = trial.suggest_float(
            "nu", config.nu_min, config.nu_max, log=config.nu_log
        )
    else:
        params["nu"] = 0.1  # default value

    params["score_loss_weight"] = trial.suggest_float(
        "score_loss_weight", config.score_loss_weight_min, config.score_loss_weight_max
    )

    # SHAPE loss parameters
    params["shape_loss_func"] = trial.suggest_categorical(
        "shape_loss_func", config.shape_loss_func
    )
    params["shape_perturb"] = trial.suggest_float(
        "shape_perturb", config.shape_perturb_min, config.shape_perturb_max
    )
    params["shape_nu"] = trial.suggest_float(
        "shape_nu", config.shape_nu_min, config.shape_nu_max, log=config.shape_nu_log
    )

    # Conditional: shape_margin only for shape_rank loss
    if params["shape_loss_func"] == "shape_rank":
        params["shape_margin"] = trial.suggest_float(
            "shape_margin", config.shape_margin_min, config.shape_margin_max
        )
    else:
        params["shape_margin"] = 0.0

    # Conditional: shape_pseudo_fy_weight for shape_rank and shape_nll
    if params["shape_loss_func"] in ["shape_rank", "shape_nll"]:
        params["shape_pseudo_fy_weight"] = trial.suggest_float(
            "shape_pseudo_fy_weight", config.shape_pseudo_fy_weight_min, config.shape_pseudo_fy_weight_max
        )
    else:
        params["shape_pseudo_fy_weight"] = 0.0

    return params
