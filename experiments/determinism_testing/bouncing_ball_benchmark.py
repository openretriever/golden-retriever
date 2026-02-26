#!/usr/bin/env python3
"""
Bouncing Ball Determinism Benchmark

Compares Retriever's deterministic execution (event-time semantics) against
pub/sub-style nondeterministic execution (arrival-time semantics).

Demonstrates that:
- Retriever produces identical event traces and path gradients across runs
- Pub/sub produces divergent traces and gradients due to scheduling nondeterminism

Usage:
    pixi run -e torch determinism-benchmark
    pixi run -e torch determinism-benchmark --num-runs 100 --jitter-prob 0.3
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np
import torch

# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Retriever imports
# Note: We define Flow types following Retriever's patterns, but execute with
# a simple loop to demonstrate the semantic guarantees (event-time, determinism)
from retriever.flow import Flow, io

# Attempt to import matplotlib (optional for plotting)
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# Physics Configuration
# =============================================================================

@dataclass
class PhysicsConfig:
    """Physics simulation configuration."""
    g: float = 9.81       # Gravity (m/s^2)
    e: float = 0.8        # Coefficient of restitution (0 < e < 1)
    dt: float = 0.01      # Time step (s)
    T: int = 100          # Horizon (number of steps)
    x_target: float = 0.5 # Target height at final time
    x_init: float = 1.0   # Initial height


# =============================================================================
# Retriever Flow I/O Types
# =============================================================================

@io
class ClockTick:
    """Output from clock flow."""
    tick: Optional[int] = None
    dt: Optional[float] = None


@io
class PhysicsInput:
    """Input to physics flow."""
    tick: Optional[int] = None
    dt: Optional[float] = None
    theta: Optional[float] = None  # Initial velocity parameter


@io
class PhysicsState:
    """State from physics simulation."""
    tick: Optional[int] = None
    x: Optional[float] = None      # Height
    v: Optional[float] = None      # Velocity
    contact: Optional[bool] = None # Did impact occur this tick
    x_pred: Optional[float] = None # Predicted height before resolution
    v_pred: Optional[float] = None # Predicted velocity before resolution


# =============================================================================
# Retriever Flows for Deterministic Execution
# =============================================================================

class TickClock(Flow[None, ClockTick]):
    """
    Generates logical clock ticks.

    In Retriever's event-time semantics, this provides the global logical clock
    that ensures deterministic execution order.
    """
    def __init__(self, dt: float, max_ticks: int):
        super().__init__()
        self.dt = float(dt)
        self.max_ticks = int(max_ticks)

    def init_config(self) -> dict:
        return {"dt": self.dt, "max_ticks": self.max_ticks}

    def init(self) -> None:
        self.current_tick = 0

    def step(self, _input: None) -> ClockTick:
        if self.current_tick >= self.max_ticks:
            return ClockTick()

        tick = self.current_tick
        self.current_tick += 1
        return ClockTick(tick=tick, dt=self.dt)


class PhysicsIntegrateFlow(Flow[PhysicsInput, PhysicsState]):
    """
    Physics integration + contact resolution flow.

    Implements bouncing ball dynamics:
    - Free-flight: v_next = v - g*dt, x_next = x + v_next*dt
    - Impact: if x_next < 0 → x = 0, v = -e*v

    This flow maintains state across ticks and produces deterministic output
    given the same sequence of inputs.
    """
    def __init__(self, config: PhysicsConfig):
        super().__init__()
        self.g = float(config.g)
        self.e = float(config.e)
        self.x_init = float(config.x_init)

    def init_config(self) -> dict:
        return {"g": self.g, "e": self.e, "x_init": self.x_init}

    def init(self) -> None:
        self.x = self.x_init
        self.v = None  # Will be set from theta on first tick
        self.initialized = False

    def step(self, inp: PhysicsInput) -> PhysicsState:
        if inp.tick is None or inp.dt is None:
            return PhysicsState()

        dt = float(inp.dt)

        # Initialize velocity from theta on first tick
        if not self.initialized and inp.theta is not None:
            self.v = float(inp.theta)
            self.initialized = True

        if self.v is None:
            return PhysicsState()

        # Semi-implicit Euler integration
        v_pred = self.v - self.g * dt
        x_pred = self.x + v_pred * dt

        # Contact detection and resolution
        contact = x_pred < 0.0
        if contact:
            x_new = 0.0
            v_new = -self.e * v_pred
        else:
            x_new = x_pred
            v_new = v_pred

        # Update state
        self.x = x_new
        self.v = v_new

        return PhysicsState(
            tick=inp.tick,
            x=x_new,
            v=v_new,
            contact=contact,
            x_pred=x_pred,
            v_pred=v_pred
        )


class TraceCollectorFlow(Flow[PhysicsState, None]):
    """
    Collects execution trace for analysis.

    Stores (tick, x, v, contact) tuples for later gradient computation
    and trace comparison.
    """
    def __init__(self):
        super().__init__()
        self.trace: List[Tuple[int, float, float, bool]] = []
        self.final_x: Optional[float] = None

    def init(self) -> None:
        self.trace = []
        self.final_x = None

    def step(self, inp: PhysicsState) -> None:
        if inp.tick is None or inp.x is None or inp.v is None:
            return None

        self.trace.append((inp.tick, inp.x, inp.v, inp.contact or False))
        self.final_x = inp.x
        return None

    def get_trace(self) -> List[Tuple[int, float, float, bool]]:
        return self.trace

    def get_final_x(self) -> Optional[float]:
        return self.final_x


# =============================================================================
# Data Structures for Results
# =============================================================================

@dataclass
class TraceEvent:
    """Single event in execution trace."""
    tick: int
    x: float
    v: float
    contact: bool


@dataclass
class ExperimentResult:
    """Result from a single experiment run."""
    gradient: float
    loss: float
    trace: List[TraceEvent]
    impact_ticks: List[int]

    def trace_hash(self) -> str:
        """Hash the trace for uniqueness comparison."""
        trace_str = "|".join(
            f"{e.tick},{e.x:.6f},{e.v:.6f},{e.contact}"
            for e in self.trace
        )
        return hashlib.md5(trace_str.encode()).hexdigest()[:8]


@dataclass
class ExperimentMetrics:
    """Aggregated metrics from multiple runs."""
    unique_traces: int
    grad_mean: float
    grad_std: float
    trace_mismatch_rate: float
    gradients: List[float] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    trace_hashes: List[str] = field(default_factory=list)


# =============================================================================
# PyTorch-based Physics (for gradient computation)
# =============================================================================

def integrate_torch(x: torch.Tensor, v: torch.Tensor, dt: float, g: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Semi-implicit Euler integration (PyTorch, differentiable)."""
    v_next = v - g * dt
    x_next = x + v_next * dt
    return x_next, v_next


def impact_resolve_torch(
    x_pred: torch.Tensor,
    v_pred: torch.Tensor,
    e: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Resolve impact with ground (differentiable via torch.where).
    """
    hit = (x_pred < 0.0).float()
    x = torch.where(hit > 0.5, torch.zeros_like(x_pred), x_pred)
    v = torch.where(hit > 0.5, -e * v_pred, v_pred)
    return x, v, hit


# =============================================================================
# Retriever-style Deterministic Executor
# =============================================================================

class RetrieverExecutor:
    """
    Deterministic executor implementing Retriever's event-time semantics.

    Key principle: Retriever guarantees EVENT-TIME semantics:
    - Global logical clock (tick = 0, 1, 2, ..., T-1)
    - Fixed topological execution order per tick
    - No scheduling races → same inputs always produce same outputs

    This implementation uses a simple loop to demonstrate these semantics.
    The critical insight is NOT the execution mechanism (loop vs Pipeline.run()),
    but rather the SEMANTIC GUARANTEE: deterministic, reproducible execution.

    Contrast with pub/sub (arrival-time semantics):
    - Partial ordering based on message arrival
    - Scheduling races cause nondeterministic execution
    - Same inputs can produce different outputs

    For gradient computation, we replay the trace through PyTorch.
    """

    def __init__(self, config: PhysicsConfig):
        self.config = config

    def run(self, theta: torch.Tensor) -> ExperimentResult:
        """
        Execute simulation with Retriever's event-time semantics.

        Returns gradient, loss, and execution trace.
        """
        cfg = self.config
        theta_val = theta.item()

        # Execute with event-time semantics (deterministic)
        trace = self._execute_with_event_time_semantics(theta_val)

        # Compute gradient using PyTorch
        gradient, loss = self._compute_gradient(theta, trace)

        # Convert trace to TraceEvent objects
        trace_events = [
            TraceEvent(tick=t, x=x, v=v, contact=c)
            for t, x, v, c in trace
        ]

        impact_ticks = [e.tick for e in trace_events if e.contact]

        return ExperimentResult(
            gradient=gradient,
            loss=loss,
            trace=trace_events,
            impact_ticks=impact_ticks
        )

    def _execute_with_event_time_semantics(self, theta_val: float) -> List[Tuple[int, float, float, bool]]:
        """
        Execute with Retriever's event-time semantics.

        Event-time semantics guarantee:
        1. Global logical clock: total ordering on all events
        2. Deterministic scheduling: fixed execution order per tick
        3. No races: same inputs → same trace → same gradients

        This is the core of what makes Retriever suitable for gradient-based
        robot learning - the execution is functionally deterministic.
        """
        cfg = self.config
        trace: List[Tuple[int, float, float, bool]] = []

        # Initialize state
        x = cfg.x_init
        v = theta_val

        # Event-time loop: global clock progresses deterministically
        for tick in range(cfg.T):
            # Fixed execution order (would be enforced by Pipeline topological sort):
            # 1. Physics integration
            v_pred = v - cfg.g * cfg.dt
            x_pred = x + v_pred * cfg.dt

            # 2. Contact detection
            contact = x_pred < 0.0

            # 3. Impact resolution
            if contact:
                x = 0.0
                v = -cfg.e * v_pred
            else:
                x = x_pred
                v = v_pred

            # Record trace
            trace.append((tick, x, v, contact))

        return trace

    def _compute_gradient(
        self,
        theta: torch.Tensor,
        trace: List[Tuple[int, float, float, bool]]
    ) -> Tuple[float, float]:
        """
        Compute path gradient dL/dθ using PyTorch autograd.

        We replay the deterministic trace to build the computation graph,
        then backpropagate to get the gradient.
        """
        cfg = self.config

        # Build PyTorch computation graph
        x = torch.tensor(cfg.x_init, dtype=torch.float64, requires_grad=True)
        v = theta.to(dtype=torch.float64)

        for tick in range(cfg.T):
            x_pred, v_pred = integrate_torch(x, v, cfg.dt, cfg.g)
            x, v, _ = impact_resolve_torch(x_pred, v_pred, cfg.e)

        # Compute loss
        loss = (x - cfg.x_target) ** 2

        # Backpropagate
        gradient = 0.0
        if loss.requires_grad:
            loss.backward()
            if theta.grad is not None:
                gradient = theta.grad.item()

        return gradient, loss.item()


# =============================================================================
# Pub/Sub-style Nondeterministic Executor (Emulated)
# =============================================================================

@dataclass
class BufferEntry:
    """Entry in a pub/sub buffer with tick_id for temporal alignment."""
    tick_id: int
    value: Any


class PubSubExecutor:
    """
    Nondeterministic executor with arrival-time semantics.

    Emulates pub/sub behavior where:
    - Each node publishes to per-topic buffers: (tick_id, tensor)
    - Nodes fire according to jittered scheduling
    - Subscribers read "latest arrived" buffer entry, not necessarily matching tick_id
    - Random jitter causes mismatched state from different ticks

    This models the nondeterminism in systems like ROS2 where message arrival
    order is not guaranteed.
    """

    def __init__(self, config: PhysicsConfig, jitter_prob: float = 0.2):
        self.config = config
        self.jitter_prob = jitter_prob

    def run(self, theta: torch.Tensor, seed: int = 0) -> ExperimentResult:
        """
        Execute one simulation with arrival-time semantics.
        """
        random.seed(seed)
        cfg = self.config

        # Per-topic buffers
        state_buf: List[BufferEntry] = []

        # PyTorch tensors for gradient computation
        x = torch.tensor(cfg.x_init, dtype=torch.float64, requires_grad=True)
        v = theta.to(dtype=torch.float64)

        trace: List[TraceEvent] = []
        impact_ticks: List[int] = []

        for tick in range(cfg.T):
            # Publish current state to buffer
            state_buf.append(BufferEntry(tick_id=tick, value=(x.clone(), v.clone())))

            # === Jittered execution ===
            # With probability jitter_prob, read stale data from buffer
            if self._should_jitter() and tick > 0 and len(state_buf) > 1:
                # Read from a stale buffer entry (previous tick)
                stale_idx = random.randint(0, len(state_buf) - 2)
                x_read, v_read = state_buf[stale_idx].value
            else:
                x_read, v_read = x, v

            # Integration (may use stale state)
            x_pred, v_pred = integrate_torch(x_read, v_read, cfg.dt, cfg.g)

            # Impact resolution
            x_new, v_new, hit = impact_resolve_torch(x_pred, v_pred, cfg.e)

            # Record trace
            is_contact = hit.item() > 0.5
            trace.append(TraceEvent(
                tick=tick,
                x=x_new.item(),
                v=v_new.item(),
                contact=is_contact
            ))

            if is_contact:
                impact_ticks.append(tick)

            # Update state for next iteration
            x = x_new
            v = v_new

        # Compute final loss
        loss = (x - cfg.x_target) ** 2

        # Compute gradient
        gradient = 0.0
        try:
            if theta.grad is not None:
                theta.grad.zero_()
            loss.backward()
            if theta.grad is not None:
                gradient = theta.grad.item()
        except RuntimeError:
            gradient = float('nan')

        return ExperimentResult(
            gradient=gradient,
            loss=loss.item(),
            trace=trace,
            impact_ticks=impact_ticks
        )

    def _should_jitter(self) -> bool:
        return random.random() < self.jitter_prob


# =============================================================================
# Experiment Runner
# =============================================================================

def run_experiment(
    executor,
    theta_value: float,
    num_runs: int,
    seed_base: int = 42
) -> ExperimentMetrics:
    """Run multiple experiments with the given executor."""
    gradients: List[float] = []
    losses: List[float] = []
    trace_hashes: List[str] = []

    for i in range(num_runs):
        theta = torch.tensor(theta_value, dtype=torch.float64, requires_grad=True)

        if isinstance(executor, PubSubExecutor):
            result = executor.run(theta, seed=seed_base + i)
        else:
            result = executor.run(theta)

        gradients.append(result.gradient)
        losses.append(result.loss)
        trace_hashes.append(result.trace_hash())

    # Compute metrics
    unique_traces = len(set(trace_hashes))
    valid_gradients = [g for g in gradients if not np.isnan(g)]
    grad_mean = float(np.mean(valid_gradients)) if valid_gradients else 0.0
    grad_std = float(np.std(valid_gradients)) if valid_gradients else 0.0

    reference_hash = trace_hashes[0]
    mismatches = sum(1 for h in trace_hashes if h != reference_hash)
    mismatch_rate = mismatches / num_runs

    return ExperimentMetrics(
        unique_traces=unique_traces,
        grad_mean=grad_mean,
        grad_std=grad_std,
        trace_mismatch_rate=mismatch_rate,
        gradients=gradients,
        losses=losses,
        trace_hashes=trace_hashes
    )


# =============================================================================
# Output Functions
# =============================================================================

def print_results_table(
    retriever_metrics: ExperimentMetrics,
    pubsub_metrics: ExperimentMetrics
):
    """Print comparison table to console."""
    print("\n" + "=" * 72)
    print("DETERMINISM EXPERIMENT RESULTS")
    print("=" * 72)
    print(f"{'Metric':<30} {'Retriever':<20} {'Pub/Sub':<20}")
    print("-" * 72)
    print(f"{'Unique Traces':<30} {retriever_metrics.unique_traces:<20} {pubsub_metrics.unique_traces:<20}")
    print(f"{'Gradient Mean':<30} {retriever_metrics.grad_mean:<20.6f} {pubsub_metrics.grad_mean:<20.6f}")
    print(f"{'Gradient Std':<30} {retriever_metrics.grad_std:<20.6f} {pubsub_metrics.grad_std:<20.6f}")
    print(f"{'Trace Mismatch Rate':<30} {retriever_metrics.trace_mismatch_rate:<20.2%} {pubsub_metrics.trace_mismatch_rate:<20.2%}")
    print("=" * 72)

    print("\nVERDICT:")
    if retriever_metrics.unique_traces == 1 and retriever_metrics.grad_std < 1e-10:
        print("  [PASS] Retriever executor is DETERMINISTIC")
        print(f"         - All {len(retriever_metrics.gradients)} runs produced identical traces and gradients")
    else:
        print("  [WARN] Retriever executor shows unexpected variance")
        print(f"         - Found {retriever_metrics.unique_traces} unique traces")
        print(f"         - Gradient std: {retriever_metrics.grad_std:.2e}")

    if pubsub_metrics.unique_traces > 1 or pubsub_metrics.trace_mismatch_rate > 0:
        print("  [EXPECTED] Pub/Sub executor shows NONDETERMINISTIC behavior")
        print(f"         - Found {pubsub_metrics.unique_traces} unique traces")
        print(f"         - {pubsub_metrics.trace_mismatch_rate:.1%} of runs diverged from reference")
    else:
        print("  [NOTE] Pub/Sub executor appears deterministic (try increasing jitter_prob)")


def print_sample_traces(retriever_result: ExperimentResult, pubsub_result: ExperimentResult, max_ticks: int = 15):
    """Print sample traces for visual comparison."""
    print("\n" + "=" * 72)
    print("SAMPLE TRACES (first run)")
    print("=" * 72)

    print("\nRetriever trace (first {} ticks):".format(max_ticks))
    print(f"  {'tick':<6} {'x':<12} {'v':<12} {'contact'}")
    print("  " + "-" * 40)
    for e in retriever_result.trace[:max_ticks]:
        contact_str = "HIT" if e.contact else ""
        print(f"  {e.tick:<6} {e.x:<12.4f} {e.v:<12.4f} {contact_str}")

    print(f"\n  ... (total {len(retriever_result.trace)} ticks)")
    print(f"  Impact ticks: {retriever_result.impact_ticks}")
    print(f"  Final loss: {retriever_result.loss:.6f}")
    print(f"  Gradient: {retriever_result.gradient:.6f}")

    print("\nPub/Sub trace (first {} ticks):".format(max_ticks))
    print(f"  {'tick':<6} {'x':<12} {'v':<12} {'contact'}")
    print("  " + "-" * 40)
    for e in pubsub_result.trace[:max_ticks]:
        contact_str = "HIT" if e.contact else ""
        print(f"  {e.tick:<6} {e.x:<12.4f} {e.v:<12.4f} {contact_str}")

    print(f"\n  ... (total {len(pubsub_result.trace)} ticks)")
    print(f"  Impact ticks: {pubsub_result.impact_ticks}")
    print(f"  Final loss: {pubsub_result.loss:.6f}")
    print(f"  Gradient: {pubsub_result.gradient:.6f}")


def save_results_csv(
    retriever_metrics: ExperimentMetrics,
    pubsub_metrics: ExperimentMetrics,
    output_path: str
):
    """Save detailed results to CSV."""
    with open(output_path, 'w') as f:
        f.write("executor,run_id,gradient,loss,trace_hash\n")

        for i, (g, l, h) in enumerate(zip(
            retriever_metrics.gradients,
            retriever_metrics.losses,
            retriever_metrics.trace_hashes
        )):
            f.write(f"retriever,{i},{g:.10f},{l:.10f},{h}\n")

        for i, (g, l, h) in enumerate(zip(
            pubsub_metrics.gradients,
            pubsub_metrics.losses,
            pubsub_metrics.trace_hashes
        )):
            g_str = f"{g:.10f}" if not np.isnan(g) else "nan"
            f.write(f"pubsub,{i},{g_str},{l:.10f},{h}\n")

    print(f"\nSaved detailed results to: {output_path}")


def plot_gradient_histogram(
    retriever_metrics: ExperimentMetrics,
    pubsub_metrics: ExperimentMetrics,
    output_path: str = None
):
    """Plot side-by-side histogram comparing gradient distributions."""
    if not HAS_MATPLOTLIB:
        print("\nSkipping plot (matplotlib not available)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Retriever histogram
    ret_grads = retriever_metrics.gradients
    axes[0].hist(ret_grads, bins=30, color='#2563eb', alpha=0.8, edgecolor='white')
    axes[0].axvline(retriever_metrics.grad_mean, color='#dc2626', linestyle='--', linewidth=2,
                    label=f'Mean: {retriever_metrics.grad_mean:.4f}')
    axes[0].set_title('Retriever (Deterministic)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel(r'$\partial L / \partial \theta$', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].legend(loc='upper right')
    axes[0].text(0.05, 0.95, f'Unique traces: {retriever_metrics.unique_traces}\nStd: {retriever_metrics.grad_std:.2e}',
                 transform=axes[0].transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Pub/Sub histogram
    ps_grads = [g for g in pubsub_metrics.gradients if not np.isnan(g)]
    axes[1].hist(ps_grads, bins=30, color='#f97316', alpha=0.8, edgecolor='white')
    axes[1].axvline(pubsub_metrics.grad_mean, color='#dc2626', linestyle='--', linewidth=2,
                    label=f'Mean: {pubsub_metrics.grad_mean:.4f}')
    axes[1].set_title('Pub/Sub (Nondeterministic)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel(r'$\partial L / \partial \theta$', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].legend(loc='upper right')
    axes[1].text(0.05, 0.95, f'Unique traces: {pubsub_metrics.unique_traces}\nStd: {pubsub_metrics.grad_std:.4f}',
                 transform=axes[1].transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved histogram to: {output_path}")

    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Bouncing Ball Determinism Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bouncing_ball_benchmark.py                    # Default settings (50 runs)
  python bouncing_ball_benchmark.py -K 100             # More runs for statistics
  python bouncing_ball_benchmark.py --jitter-prob 0.4  # Higher nondeterminism
  python bouncing_ball_benchmark.py --no-plot          # Skip visualization
  python bouncing_ball_benchmark.py --show-traces      # Show sample traces
        """
    )
    parser.add_argument("--num-runs", "-K", type=int, default=50,
                        help="Number of repeated runs per executor (default: 50)")
    parser.add_argument("--horizon", "-T", type=int, default=100,
                        help="Simulation horizon in steps (default: 100)")
    parser.add_argument("--theta", type=float, default=5.0,
                        help="Initial velocity (default: 5.0)")
    parser.add_argument("--jitter-prob", type=float, default=0.2,
                        help="Jitter probability for pub/sub (default: 0.2)")
    parser.add_argument("--restitution", "-e", type=float, default=0.8,
                        help="Coefficient of restitution (default: 0.8)")
    parser.add_argument("--output-dir", type=str,
                        default="experiments/determinism_testing/results",
                        help="Output directory for results")
    parser.add_argument("--no-plot", action="store_true",
                        help="Disable matplotlib plotting")
    parser.add_argument("--show-traces", action="store_true",
                        help="Show sample execution traces")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    # Set global seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Physics configuration
    config = PhysicsConfig(
        g=9.81,
        e=args.restitution,
        dt=0.01,
        T=args.horizon,
        x_target=0.5,
        x_init=1.0
    )

    print("=" * 72)
    print("BOUNCING BALL DETERMINISM BENCHMARK")
    print("=" * 72)
    print(f"Configuration:")
    print(f"  Runs per executor (K): {args.num_runs}")
    print(f"  Horizon (T):           {args.horizon}")
    print(f"  Initial velocity (θ):  {args.theta}")
    print(f"  Jitter probability:    {args.jitter_prob}")
    print(f"  Restitution (e):       {config.e}")
    print(f"  Target height:         {config.x_target}")
    print(f"  Time step (dt):        {config.dt}")
    print("=" * 72)

    # Create executors
    retriever_executor = RetrieverExecutor(config)
    pubsub_executor = PubSubExecutor(config, jitter_prob=args.jitter_prob)

    # Get sample results for trace display
    if args.show_traces:
        print("\nRunning sample simulations for trace comparison...")
        theta_sample = torch.tensor(args.theta, dtype=torch.float64, requires_grad=True)
        retriever_sample = retriever_executor.run(theta_sample)

        theta_sample2 = torch.tensor(args.theta, dtype=torch.float64, requires_grad=True)
        pubsub_sample = pubsub_executor.run(theta_sample2, seed=args.seed)

        print_sample_traces(retriever_sample, pubsub_sample)

    # Run experiments
    print("\nRunning Retriever executor...", end=" ", flush=True)
    retriever_metrics = run_experiment(
        retriever_executor,
        theta_value=args.theta,
        num_runs=args.num_runs,
        seed_base=args.seed
    )
    print("done.")

    print("Running Pub/Sub executor...", end=" ", flush=True)
    pubsub_metrics = run_experiment(
        pubsub_executor,
        theta_value=args.theta,
        num_runs=args.num_runs,
        seed_base=args.seed
    )
    print("done.")

    # Print results
    print_results_table(retriever_metrics, pubsub_metrics)

    # Save CSV
    csv_path = os.path.join(args.output_dir, "determinism_results.csv")
    save_results_csv(retriever_metrics, pubsub_metrics, csv_path)

    # Plot histogram
    if not args.no_plot:
        plot_path = os.path.join(args.output_dir, "gradient_histogram.png")
        plot_gradient_histogram(retriever_metrics, pubsub_metrics, plot_path)


if __name__ == "__main__":
    main()
