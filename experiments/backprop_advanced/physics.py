"""
Differentiable bouncing ball physics — PyTorch batched utilities.

Extends experiments/physics.py with torch-based batch simulation
and autograd-compatible contact resolution.
No Retriever dependency.
"""

from __future__ import annotations

import os
import sys

import torch

# Shared pure-Python physics (PhysicsConfig, finite_difference_gradient)
_experiments_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _experiments_dir not in sys.path:
    sys.path.insert(0, _experiments_dir)
from physics_config import PhysicsConfig, finite_difference_gradient  # re-exported for callers

__all__ = [
    "PhysicsConfig",
    "finite_difference_gradient",
    "step_batch",
    "simulate_batch",
    "batch_loss",
]


def step_batch(
    x: torch.Tensor,
    v: torch.Tensor,
    cfg: PhysicsConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    One semi-implicit Euler step for a batch of balls.

    Works for both scalar and [B] tensors.
    Contact resolution uses torch.where to keep the autograd graph connected.
    """
    v_pred = v - cfg.g * cfg.dt
    x_pred = x + v_pred * cfg.dt

    # Differentiable contact: keep computation graph alive
    hit = (x_pred.detach() < 0.0).float()
    x_next = torch.where(hit > 0.5, x_pred * 0.0, x_pred)
    v_next = torch.where(hit > 0.5, -cfg.e * v_pred, v_pred)
    return x_next, v_next


def simulate_batch(
    theta_batch: torch.Tensor,
    cfg: PhysicsConfig,
) -> torch.Tensor:
    """
    Run T steps of physics for a batch of initial velocities.

    Args:
        theta_batch: initial velocities, shape [B] or scalar, requires_grad=True
        cfg: physics configuration

    Returns:
        x_T: final heights, shape [B] or scalar, with full autograd graph
    """
    B = theta_batch.shape[0] if theta_batch.dim() > 0 else 1
    x = torch.full((B,), cfg.x_init, dtype=theta_batch.dtype)
    v = theta_batch.clone() if B > 1 else theta_batch.unsqueeze(0).clone()

    for _ in range(cfg.T):
        x, v = step_batch(x, v, cfg)

    return x if B > 1 else x.squeeze(0)


def batch_loss(
    theta_batch: torch.Tensor,
    cfg: PhysicsConfig,
) -> torch.Tensor:
    """
    MSE loss for a batch: L_i = (x_T_i - x_target)^2

    Returns:
        losses: shape [B], full autograd graph alive
    """
    x_T = simulate_batch(theta_batch, cfg)
    return (x_T - cfg.x_target) ** 2
