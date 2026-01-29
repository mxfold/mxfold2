"""Sharpness-Aware Minimization (SAM) optimizer wrappers.

This module provides SAM and its variants (ASAM, GSAM) as optimizer wrappers
that can be used with any base optimizer (Adam, AdamW, SGD, etc.).

SAM minimizes both the loss value and the loss sharpness, leading to better
generalization. It requires two forward-backward passes per optimization step.

References:
    - SAM: https://arxiv.org/abs/2010.01412 (Foret et al., 2021)
    - ASAM: https://arxiv.org/abs/2102.11600 (Kwon et al., 2021)
    - GSAM: https://arxiv.org/abs/2203.08065 (Zhuang et al., 2022)
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer


class SAM(Optimizer):
    """Sharpness-Aware Minimization optimizer wrapper.

    SAM minimizes both the loss value and the loss sharpness by computing
    gradients at perturbed weights. This requires two forward-backward passes
    per optimization step.

    Args:
        params: Model parameters
        base_optimizer: Base optimizer class (e.g., torch.optim.AdamW)
        rho: Perturbation radius (default: 0.05)
        adaptive: Whether to use adaptive perturbation (ASAM) (default: False)
        **kwargs: Arguments passed to base optimizer
    """

    def __init__(
        self,
        params: Iterable,
        base_optimizer: type[Optimizer],
        rho: float = 0.05,
        adaptive: bool = False,
        **kwargs
    ) -> None:
        defaults = dict(rho=rho, adaptive=adaptive)
        super(SAM, self).__init__(params, defaults)

        # Create base optimizer with the same parameter groups
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.rho = rho
        self.adaptive = adaptive

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        """Compute perturbation and apply to weights (ascent step).

        This should be called after the first backward pass.
        Stores the original weights and applies epsilon perturbation.

        Args:
            zero_grad: If True, zero gradients after this step
        """
        grad_norm = self._grad_norm()

        for group in self.param_groups:
            scale = group.get('rho', self.rho) / (grad_norm + 1e-12)

            for p in group['params']:
                if p.grad is None:
                    continue

                # Store original weights
                self.state[p]['old_p'] = p.data.clone()

                # Compute epsilon (perturbation direction)
                if group.get('adaptive', self.adaptive):
                    # ASAM: adaptive perturbation scaled by parameter magnitude
                    eps = torch.pow(p.data, 2) * p.grad * scale
                else:
                    eps = p.grad * scale

                # Apply perturbation: w + eps
                p.add_(eps)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        """Restore weights and apply gradient update (descent step).

        This should be called after the second backward pass.
        Restores original weights and applies base optimizer update.

        Args:
            zero_grad: If True, zero gradients after this step
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Restore original weights
                if 'old_p' in self.state[p]:
                    p.data = self.state[p]['old_p']

        # Apply base optimizer step with gradients from perturbed position
        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> torch.Tensor:
        """Single step combining first_step and second_step.

        This is for compatibility with standard optimizer interface.
        Requires a closure that computes the loss.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            Loss value from the closure.
        """
        if closure is None:
            raise RuntimeError("SAM.step() requires a closure")

        # First forward-backward pass
        with torch.enable_grad():
            loss = closure()

        self.first_step(zero_grad=True)

        # Second forward-backward pass at perturbed weights
        with torch.enable_grad():
            closure()

        self.second_step()

        return loss

    def _grad_norm(self) -> torch.Tensor:
        """Compute the gradient norm across all parameter groups."""
        shared_device = self.param_groups[0]['params'][0].device

        grad_norms = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if group.get('adaptive', self.adaptive):
                        grad_norms.append((torch.abs(p.data) * p.grad).norm(p=2).to(shared_device))
                    else:
                        grad_norms.append(p.grad.norm(p=2).to(shared_device))

        if not grad_norms:
            return torch.tensor(0.0, device=shared_device)

        return torch.norm(torch.stack(grad_norms), p=2)

    def state_dict(self) -> dict:
        """Return state dict including base optimizer state."""
        return {
            'sam_state': super().state_dict(),
            'base_optimizer_state': self.base_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load state dict including base optimizer state."""
        super().load_state_dict(state_dict['sam_state'])
        self.base_optimizer.load_state_dict(state_dict['base_optimizer_state'])


class ASAM(SAM):
    """Adaptive Sharpness-Aware Minimization optimizer.

    ASAM uses element-wise adaptive perturbation scaling based on
    parameter magnitudes, which works better with scale-invariant networks.

    Args:
        params: Model parameters
        base_optimizer: Base optimizer class
        rho: Perturbation radius (default: 0.5, typically larger than SAM)
        **kwargs: Arguments passed to base optimizer
    """

    def __init__(
        self,
        params: Iterable,
        base_optimizer: type[Optimizer],
        rho: float = 0.5,
        **kwargs
    ) -> None:
        super().__init__(params, base_optimizer, rho=rho, adaptive=True, **kwargs)


class GSAM(SAM):
    """Surrogate Gap Guided Sharpness-Aware Minimization optimizer.

    GSAM modifies SAM by using a surrogate gap term to better guide
    the perturbation direction towards flatter minima.

    Args:
        params: Model parameters
        base_optimizer: Base optimizer class
        rho: Perturbation radius (default: 0.05)
        alpha: Gap weighting parameter (default: 0.1)
        **kwargs: Arguments passed to base optimizer
    """

    def __init__(
        self,
        params: Iterable,
        base_optimizer: type[Optimizer],
        rho: float = 0.05,
        alpha: float = 0.1,
        **kwargs
    ) -> None:
        self.alpha = alpha
        self._first_loss: Optional[torch.Tensor] = None
        super().__init__(params, base_optimizer, rho=rho, adaptive=False, **kwargs)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False, loss: Optional[torch.Tensor] = None) -> None:
        """GSAM first step - stores loss for gap calculation.

        Args:
            zero_grad: If True, zero gradients after this step
            loss: The loss value from the first forward pass (for gap calculation)
        """
        if loss is not None:
            self._first_loss = loss.detach().clone()
        super().first_step(zero_grad=zero_grad)

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False, loss: Optional[torch.Tensor] = None) -> None:
        """GSAM second step - applies gap-weighted gradient.

        Args:
            zero_grad: If True, zero gradients after this step
            loss: The loss value from the second forward pass (at perturbed weights)
        """
        if loss is not None and self._first_loss is not None:
            # Compute surrogate gap: L(w + eps) - L(w)
            gap = loss.detach() - self._first_loss

            # Scale gradients by gap-aware factor
            # Positive gap means loss increased (good for SAM), amplify the gradient
            # Negative gap means loss decreased (bad), reduce the gradient
            scale_factor = 1.0 + self.alpha * torch.sign(gap)

            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        p.grad.mul_(scale_factor)

        super().second_step(zero_grad=zero_grad)
        self._first_loss = None
