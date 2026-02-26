#!/usr/bin/env python3
"""
Advanced Gradient Backpropagation through Retriever Pipeline.

Demonstrates:
  1. Batch vectorized physics inside a Retriever Flow
     - One Flow step advances B balls simultaneously
     - Single loss.sum().backward() computes B gradients
  2. Gradient verification against finite differences
  3. Gradient descent optimization on a batch of initial velocities

Key concept: The Retriever in-process stepper acts as a PyTorch model —
tensors flow by reference through InMemoryChannel, so the full autograd
graph survives across Flow boundaries.

Usage:
    python bouncing_ball_backprop.py                   # default: B=8, verify + optimize
    python bouncing_ball_backprop.py --batch-size 32   # larger batch
    python bouncing_ball_backprop.py --no-verify       # skip FD check
    python bouncing_ball_backprop.py --steps 200       # more gradient descent steps
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np

# Project root (GoldenRetriever/)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Local physics module
sys.path.insert(0, os.path.dirname(__file__))
from physics import PhysicsConfig, simulate_batch, batch_loss, finite_difference_gradient

# Retriever imports
from retriever.flow import Flow, Pipeline, Rate, Trigger, io


# =============================================================================
# Flow I/O Types
# =============================================================================

@io
class ClockTick:
    tick: Optional[int] = None
    dt: Optional[float] = None


@io
class BallBatchState:
    """Batch ball state with raw torch.Tensor values [B]. Keeps autograd graph alive."""
    tick: Optional[int] = None
    x: Optional[Any] = None    # torch.Tensor [B]
    v: Optional[Any] = None    # torch.Tensor [B]


@io
class BatchSinkState:
    """Captures the final [B] tensor batch for loss computation."""
    x: Optional[Any] = None    # torch.Tensor [B]
    done: Optional[bool] = None


# =============================================================================
# Flows
# =============================================================================

class TickClockFlow(Flow[None, ClockTick]):
    """Generates logical clock ticks."""

    def __init__(self, dt: float, max_ticks: int):
        super().__init__()
        self.dt = float(dt)
        self.max_ticks = int(max_ticks)

    def init_config(self) -> dict:
        return {"dt": self.dt, "max_ticks": self.max_ticks}

    def init(self) -> None:
        self.tick = 0

    def step(self, _: None) -> ClockTick:
        if self.tick >= self.max_ticks:
            return ClockTick()
        t = self.tick
        self.tick += 1
        return ClockTick(tick=t, dt=self.dt)


class BouncingBallBatchFlow(Flow[ClockTick, BallBatchState]):
    """
    Vectorized ball flow: one step advances B balls simultaneously.

    theta_vals: List of B initial velocities, each a float.
    After init(), self.theta is a [B] tensor with requires_grad=True.
    Gradients accumulate in self.theta.grad after loss.backward().
    """

    def __init__(self, theta_vals: List[float], cfg: PhysicsConfig):
        super().__init__()
        self.theta_vals = list(theta_vals)
        self.cfg = cfg

    def init_config(self) -> dict:
        return {"theta_vals": self.theta_vals}

    def init(self) -> None:
        B = len(self.theta_vals)
        self.theta = torch.tensor(self.theta_vals, dtype=torch.float64, requires_grad=True)
        self.x = torch.full((B,), self.cfg.x_init, dtype=torch.float64)
        self.v = self.theta.clone()

    def step(self, inp: ClockTick) -> BallBatchState:
        if inp.tick is None or inp.dt is None:
            return BallBatchState()

        dt = float(inp.dt)

        # Vectorized semi-implicit Euler
        v_pred = self.v - self.cfg.g * dt
        x_pred = self.x + v_pred * dt

        # Differentiable contact (torch.where keeps grad graph alive)
        hit = (x_pred.detach() < 0.0).float()
        self.x = torch.where(hit > 0.5, x_pred * 0.0, x_pred)
        self.v = torch.where(hit > 0.5, -self.cfg.e * v_pred, v_pred)

        # Return raw tensors — grad graph survives across Flow boundary (in-process)
        return BallBatchState(tick=inp.tick, x=self.x, v=self.v)


class BatchTensorSinkFlow(Flow[BallBatchState, BatchSinkState]):
    """Captures the final [B] tensor for the caller to compute loss."""

    def __init__(self, max_ticks: int):
        super().__init__()
        self.max_ticks = int(max_ticks)

    def init_config(self) -> dict:
        return {"max_ticks": self.max_ticks}

    def init(self) -> None:
        self.last_x: Optional[torch.Tensor] = None

    def step(self, inp: BallBatchState) -> BatchSinkState:
        if inp.tick is None or inp.x is None:
            return BatchSinkState()
        self.last_x = inp.x
        if inp.tick >= self.max_ticks - 1:
            return BatchSinkState(x=self.last_x, done=True)
        return BatchSinkState()


# =============================================================================
# Pipeline Builder
# =============================================================================

def build_batch_pipeline(
    theta_vals: List[float],
    cfg: PhysicsConfig,
) -> Tuple[Pipeline, BouncingBallBatchFlow, BatchTensorSinkFlow]:
    """
    Build a 3-node pipeline:  Clock → BatchBall → BatchSink

    The in-process stepper passes tensors by reference —
    the PyTorch computation graph spans all three Flows.
    """
    pipe = Pipeline("bouncing_ball_batch")

    clock = TickClockFlow(dt=cfg.dt, max_ticks=cfg.T)
    ball = BouncingBallBatchFlow(theta_vals=theta_vals, cfg=cfg)
    sink = BatchTensorSinkFlow(max_ticks=cfg.T)

    with pipe:
        h_clock = clock @ Rate(hz=100)
        h_ball = ball @ Trigger("tick")
        h_sink = sink @ Trigger("x")
        h_clock >> h_ball >> h_sink

    return pipe, ball, sink


# =============================================================================
# Section 1: Batch Gradient Computation
# =============================================================================

def run_batch_pipeline(
    theta_vals: List[float],
    cfg: PhysicsConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run B simulations through the Retriever pipeline, compute B gradients.

    Returns:
        grads [B]: dL/dθ for each θ_i
        losses [B]: final loss values
    """
    pipe, ball, sink = build_batch_pipeline(theta_vals, cfg)

    # Forward pass via in-process stepper
    for _ in range(cfg.T + 5):
        pipe.step(dt=cfg.dt)

    assert sink.last_x is not None, "Pipeline produced no output"
    assert sink.last_x.grad_fn is not None, \
        "final_x has no grad_fn — computation graph broken"

    # Compute batch loss and backprop
    losses = (sink.last_x - cfg.x_target) ** 2   # [B]
    losses.sum().backward()                         # single backward for all B

    grads = ball.theta.grad.clone()                 # [B]
    return grads, losses.detach()


# =============================================================================
# Section 2: Gradient Verification
# =============================================================================

def verify_batch_gradients(
    theta_vals: List[float],
    cfg: PhysicsConfig,
    eps: float = 1e-6,
    tol: float = 1e-4,
) -> bool:
    """
    Compare batch autograd against finite differences for each θ_i.

    Returns True if all errors are below tol.
    """
    print(f"\n{'='*60}")
    print("BATCH GRADIENT VERIFICATION (autograd vs finite differences)")
    print(f"{'='*60}")
    print(f"  B={len(theta_vals)}, eps={eps:.0e}, tol={tol:.0e}")
    print()

    # Batch autograd via Retriever pipeline
    grads_auto, losses = run_batch_pipeline(theta_vals, cfg)

    header = f"  {'theta':>8}  {'autograd':>14}  {'FD':>14}  {'|error|':>12}  {'pass?':>6}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    all_pass = True
    for i, th in enumerate(theta_vals):
        grad_fd, _ = finite_difference_gradient(th, cfg, eps=eps)
        err = abs(grads_auto[i].item() - grad_fd)
        ok = err < tol
        all_pass = all_pass and ok
        mark = "OK" if ok else "FAIL"
        print(f"  {th:8.2f}  {grads_auto[i].item():14.8f}  {grad_fd:14.8f}  {err:12.2e}  {mark:>6}")

    print()
    if all_pass:
        print(f"  PASS: all {len(theta_vals)} gradients match finite differences (tol={tol:.0e})")
    else:
        print(f"  FAIL: some gradients exceed tolerance")
    return all_pass


# =============================================================================
# Section 3: Gradient Descent Optimization
# =============================================================================

def optimize_batch(
    theta_init: List[float],
    cfg: PhysicsConfig,
    lr: float = 0.1,
    steps: int = 100,
    print_every: int = 10,
) -> Dict:
    """
    Gradient descent on B initial velocities simultaneously.

    Uses the Retriever pipeline as the differentiable forward model.
    Rebuilds the pipeline each step (fresh computation graph).

    Returns dict with theta_history, loss_history, final_theta, optimal_loss.
    """
    print(f"\n{'='*60}")
    print("BATCH GRADIENT DESCENT")
    print(f"{'='*60}")
    print(f"  B={len(theta_init)}, lr={lr}, steps={steps}")
    print(f"  θ_init = {[f'{t:.1f}' for t in theta_init]}")
    print()

    theta_vals = list(theta_init)
    theta_history = [list(theta_vals)]
    loss_history = []

    for step in range(steps):
        grads, losses = run_batch_pipeline(theta_vals, cfg)

        loss_mean = losses.mean().item()
        loss_history.append(losses.tolist())

        if step % print_every == 0 or step == steps - 1:
            theta_str = "  ".join(f"{t:.4f}" for t in theta_vals)
            print(f"  step {step:4d}: loss_mean={loss_mean:.6f}  theta=[{theta_str}]")

        # Gradient step
        theta_vals = [t - lr * g.item() for t, g in zip(theta_vals, grads)]
        theta_history.append(list(theta_vals))

    # Final evaluation
    _, final_losses = run_batch_pipeline(theta_vals, cfg)
    final_loss_mean = final_losses.mean().item()

    print()
    print(f"  Final theta: {[f'{t:.4f}' for t in theta_vals]}")
    print(f"  Final loss mean: {final_loss_mean:.8f}")
    print(f"  All converged: {final_loss_mean < 1e-4}")

    return {
        "theta_history": theta_history,
        "loss_history": loss_history,
        "final_theta": theta_vals,
        "final_loss": final_loss_mean,
    }


# =============================================================================
# Section 4: Batch vs Sequential Comparison
# =============================================================================

def compare_batch_vs_sequential(
    batch_sizes: List[int],
    cfg: PhysicsConfig,
    theta_base: float = 3.0,
    theta_step: float = 0.5,
) -> None:
    """
    Compare batch pipeline against B sequential scalar pipelines.
    Verifies gradient correctness and measures timing.
    """
    print(f"\n{'='*60}")
    print("BATCH vs SEQUENTIAL COMPARISON")
    print(f"{'='*60}")
    print(f"  {'B':>4}  {'batch_ms':>10}  {'seq_ms':>10}  {'speedup':>8}  {'grad_match':>12}")
    print("  " + "-" * 52)

    for B in batch_sizes:
        theta_vals = [theta_base + i * theta_step for i in range(B)]

        # Batch
        t0 = time.perf_counter()
        grads_batch, losses_batch = run_batch_pipeline(theta_vals, cfg)
        batch_ms = (time.perf_counter() - t0) * 1000

        # Sequential (B independent scalar pipelines)
        t0 = time.perf_counter()
        grads_seq = []
        for th in theta_vals:
            g, _ = run_batch_pipeline([th], cfg)
            grads_seq.append(g[0].item())
        seq_ms = (time.perf_counter() - t0) * 1000

        # Check gradients match
        match = all(
            abs(grads_batch[i].item() - grads_seq[i]) < 1e-6
            for i in range(B)
        )
        speedup = seq_ms / batch_ms if batch_ms > 0 else float("inf")
        match_str = "OK" if match else "MISMATCH"

        print(f"  {B:4d}  {batch_ms:10.1f}  {seq_ms:10.1f}  {speedup:8.2f}x  {match_str:>12}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch gradient backprop through Retriever pipeline"
    )
    parser.add_argument("--batch-size", "-B", type=int, default=8,
                        help="Number of initial velocities in batch (default: 8)")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="Learning rate for gradient descent (default: 0.1)")
    parser.add_argument("--steps", type=int, default=100,
                        help="Gradient descent steps (default: 100)")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip finite difference verification")
    parser.add_argument("--no-compare", action="store_true",
                        help="Skip batch vs sequential comparison")
    parser.add_argument("--horizon", "-T", type=int, default=100,
                        help="Simulation horizon (default: 100)")
    args = parser.parse_args()

    cfg = PhysicsConfig(g=9.81, e=0.8, dt=0.01, T=args.horizon,
                        x_target=0.5, x_init=1.0)

    B = args.batch_size
    theta_init = [2.0 + i * 1.0 for i in range(B)]   # spread from 2.0 to 2+B

    print("=" * 60)
    print("ADVANCED GRADIENT BACKPROPAGATION — BOUNCING BALL")
    print("=" * 60)
    print(f"  Batch size B: {B}")
    print(f"  Horizon T:    {args.horizon}")
    print(f"  θ_init:       {[f'{t:.1f}' for t in theta_init]}")
    print(f"  lr:           {args.lr}")
    print(f"  steps:        {args.steps}")

    # Gradient verification
    if not args.no_verify:
        ok = verify_batch_gradients(theta_init, cfg)
        if not ok:
            print("\nVerification FAILED — aborting")
            sys.exit(1)

    # Gradient descent optimization
    result = optimize_batch(theta_init, cfg, lr=args.lr, steps=args.steps)

    # Batch vs sequential comparison
    if not args.no_compare:
        compare_batch_vs_sequential([1, 4, 8, min(B, 16)], cfg)

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
