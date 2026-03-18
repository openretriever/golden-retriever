"""
Shared bouncing ball physics configuration and numerical utilities.

Pure Python — no framework dependency.
Used by both determinism_testing and backprop_advanced experiments.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PhysicsConfig:
    g: float = 9.81       # gravity
    e: float = 0.8        # restitution coefficient
    dt: float = 0.01      # time step
    T: int = 100          # simulation horizon
    x_target: float = 0.5
    x_init: float = 1.0


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
