#!/usr/bin/env python3
"""
Verify True Gradient Computation

This script analytically computes the gradient and compares it with PyTorch autograd
to verify correctness. It also explains why pub/sub yields different mean gradients.

The key insight: Pub/sub's arrival-time jitter corrupts the execution trace, leading
to gradients of CORRUPTED trajectories, not the true trajectory.
"""

from __future__ import annotations

import numpy as np
import torch
import sys
import os

# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Shared physics (experiments/physics.py)
_experiments_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _experiments_dir not in sys.path:
    sys.path.insert(0, _experiments_dir)
from physics_config import PhysicsConfig, finite_difference_gradient


def pytorch_gradient(g: float, e: float, dt: float, x_init: float, x_target: float, theta_val: float, T: int) -> tuple[float, float]:
    """Compute gradient using PyTorch autograd."""
    x = torch.tensor(x_init, dtype=torch.float64, requires_grad=True)
    theta = torch.tensor(theta_val, dtype=torch.float64, requires_grad=True)
    v = theta.clone()

    for t in range(T):
        v_pred = v - g * dt
        x_pred = x + v_pred * dt

        # Differentiable impact
        hit = (x_pred < 0.0).float()
        x = torch.where(hit > 0.5, torch.zeros_like(x_pred), x_pred)
        v = torch.where(hit > 0.5, -e * v_pred, v_pred)

    loss = (x - x_target) ** 2

    if loss.requires_grad:
        loss.backward()
        gradient = theta.grad.item() if theta.grad is not None else 0.0
    else:
        gradient = 0.0

    return gradient, loss.item()


def main():
    # Physics parameters (same as benchmark)
    g = 9.81
    e = 0.8
    dt = 0.01
    T = 100
    x_init = 1.0
    x_target = 0.5
    theta = 3.0

    print("=" * 70)
    print("GRADIENT VERIFICATION")
    print("=" * 70)
    print(f"Parameters: θ={theta}, g={g}, e={e}, dt={dt}, T={T}")
    print(f"Target height: {x_target}")
    print()

    # Compute finite difference gradient
    print("Computing finite difference gradient...")
    cfg = PhysicsConfig(g=g, e=e, dt=dt, T=T, x_target=x_target, x_init=x_init)
    grad_fd, loss_fd = finite_difference_gradient(theta, cfg)

    # Compute PyTorch gradient
    print("Computing PyTorch autograd gradient...")
    grad_pytorch, loss_pytorch = pytorch_gradient(g, e, dt, x_init, x_target, theta, T)

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Finite difference gradient:  {grad_fd:.10f}")
    print(f"PyTorch autograd gradient:   {grad_pytorch:.10f}")
    print(f"Difference:                  {abs(grad_fd - grad_pytorch):.2e}")
    print()
    print(f"Finite difference loss:      {loss_fd:.10f}")
    print(f"PyTorch loss:                {loss_pytorch:.10f}")
    print("=" * 70)

    if abs(grad_fd - grad_pytorch) < 1e-4:
        print("\n✅ VERIFIED: PyTorch gradient matches finite difference")
        print("   This is the TRUE gradient for the deterministic trajectory.")
    else:
        print("\n⚠️  WARNING: Gradients don't match")
        print(f"   Relative error: {abs(grad_fd - grad_pytorch) / abs(grad_pytorch):.2e}")

    # Explain why pub/sub has different mean
    print()
    print("=" * 70)
    print("WHY PUB/SUB HAS DIFFERENT MEAN GRADIENT")
    print("=" * 70)
    print(f"True gradient (Retriever):  {grad_pytorch:.6f}")
    print()

    # Try to load pub/sub results
    results_file = "experiments/determinism_testing/results/determinism_results.csv"
    if os.path.exists(results_file):
        pubsub_grads = []
        with open(results_file, 'r') as f:
            f.readline()  # skip header
            for line in f:
                parts = line.strip().split(',')
                if parts[0] == 'pubsub':
                    pubsub_grads.append(float(parts[2]))

        if pubsub_grads:
            pubsub_mean = np.mean(pubsub_grads)
            pubsub_std = np.std(pubsub_grads)
            print(f"Pub/Sub mean gradient:      {pubsub_mean:.6f}")
            print(f"Pub/Sub std:                {pubsub_std:.6f}")
            print(f"Difference from true:       {abs(pubsub_mean - grad_pytorch):.6f}")
            print()

    print("Pub/sub with arrival-time jitter produces CORRUPTED traces:")
    print("  - State/contact mismatches (reading stale state)")
    print("  - Different impact timings")
    print("  - Different final positions")
    print()
    print("Each corrupted trace has a different gradient.")
    print("The MEAN of corrupted gradients ≠ TRUE gradient!")
    print()
    print("This is why arrival-time semantics breaks gradient-based learning:")
    print("  E[∇L(corrupted_trajectory)] ≠ ∇L(true_trajectory)")
    print()
    print("Key insight:")
    print("  • Retriever: Computes ∇L(true_trajectory) = " + f"{grad_pytorch:.6f}")
    print("  • Pub/Sub:  Computes E[∇L(corrupted_trajectory)] ≠ true gradient")
    print()
    print("For gradient descent to work, you need the TRUE gradient!")
    print("Retriever's event-time semantics provides this guarantee.")
    print("=" * 70)

    # Show trace statistics
    print()
    print("TRACE STATISTICS:")
    x = x_init
    v = theta
    impacts = 0
    for t in range(T):
        v_pred = v - g * dt
        x_pred = x + v_pred * dt
        if x_pred < 0.0:
            x = 0.0
            v = -e * v_pred
            impacts += 1
        else:
            x = x_pred
            v = v_pred
    print(f"  Total impacts: {impacts}")
    print(f"  Final height:  {x:.6f}")
    print(f"  Target height: {x_target:.6f}")
    print(f"  Error:         {x - x_target:.6f}")


if __name__ == "__main__":
    main()
