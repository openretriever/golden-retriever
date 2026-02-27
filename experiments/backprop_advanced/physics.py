"""
Differentiable bouncing ball physics (scalar and batched).

Standalone module — no Retriever dependency.
Works with both scalar θ and batched θ [B].
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PhysicsConfig:
    g: float = 9.81       # gravity
    e: float = 0.8        # restitution coefficient
    dt: float = 0.01      # time step
    T: int = 100          # simulation horizon
    x_target: float = 0.5
    x_init: float = 1.0


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


def finite_difference_gradient(
    theta_val: float,
    cfg: PhysicsConfig,
    eps: float = 1e-6,
) -> tuple[float, float]:
    """
    Scalar finite-difference gradient dL/dθ at a single θ value.

    Uses central differences: (L(θ+ε/2) - L(θ-ε/2)) / ε

    Returns:
        (gradient, loss_at_theta)
    """
    def simulate_loss(th: float) -> float:
        x, v = cfg.x_init, th
        for _ in range(cfg.T):
            v_pred = v - cfg.g * cfg.dt
            x_pred = x + v_pred * cfg.dt
            if x_pred < 0.0:
                x, v = 0.0, -cfg.e * v_pred
            else:
                x, v = x_pred, v_pred
        return (x - cfg.x_target) ** 2

    loss = simulate_loss(theta_val)
    loss_plus = simulate_loss(theta_val + eps / 2)
    loss_minus = simulate_loss(theta_val - eps / 2)
    grad = (loss_plus - loss_minus) / eps
    return grad, loss
