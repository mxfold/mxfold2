from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn


class EMA:
    """Exponential Moving Average for model weights.

    Maintains a shadow copy of the model with exponentially averaged weights.
    The update formula is: ema_weight = decay * ema_weight + (1 - decay) * model_weight

    Args:
        model: The model to track
        decay: The decay factor (default: 0.999)
        device: Device to store the shadow model (default: same as model)
    """

    def __init__(self, model: nn.Module, decay: float = 0.999, device: Optional[str] = None):
        self.decay = decay
        self.shadow = copy.deepcopy(model)

        # Move to specified device if provided
        if device is not None:
            self.shadow.to(device)

        # Disable gradients for shadow model
        for p in self.shadow.parameters():
            p.requires_grad_(False)

        # Set to eval mode
        self.shadow.eval()

    def update(self, model: nn.Module) -> None:
        """Update the shadow model with current model weights.

        Args:
            model: The current model to update from
        """
        with torch.no_grad():
            model_params = dict(model.named_parameters())
            shadow_params = dict(self.shadow.named_parameters())

            for name, shadow_p in shadow_params.items():
                if name in model_params:
                    model_p = model_params[name]
                    # EMA update: shadow = decay * shadow + (1 - decay) * model
                    shadow_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)

            # Also update buffers (e.g., BatchNorm running stats)
            model_buffers = dict(model.named_buffers())
            shadow_buffers = dict(self.shadow.named_buffers())

            for name, shadow_b in shadow_buffers.items():
                if name in model_buffers:
                    shadow_b.data.copy_(model_buffers[name].data)

    def state_dict(self) -> dict:
        """Return the state dict of the shadow model."""
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        """Load state dict into the shadow model."""
        self.shadow.load_state_dict(state_dict)

    def to(self, device) -> 'EMA':
        """Move the shadow model to the specified device."""
        self.shadow.to(device)
        return self
