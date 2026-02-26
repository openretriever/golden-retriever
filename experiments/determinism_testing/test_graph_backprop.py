#!/usr/bin/env python3
"""
Tests for graph-level gradient backpropagation.

Verifies that:
1. Forward pass produces tensors with grad_fn (computation graph alive)
2. loss.backward() gives correct gradient through the pipeline
3. Reversed graph topology is correct
4. Gradient matches the trace-replay method (existing approach)
"""

from __future__ import annotations

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from bouncing_ball_pipeline import PhysicsConfig, build_tensor_pipeline, build_bouncing_ball_pipeline
from graph_backprop import reverse_graph, get_backward_order, GraphBackprop, TensorStore


THETA = 3.0
CONFIG = PhysicsConfig(g=9.81, e=0.8, dt=0.01, T=100, x_target=0.5, x_init=1.0)


def test_tensor_pipeline_grad_fn():
    """Test 1: Forward pass produces tensors with grad_fn."""
    pipe, ball, sink = build_tensor_pipeline(CONFIG, THETA)

    for t in range(CONFIG.T + 5):
        pipe.step(dt=CONFIG.dt)

    assert sink.last_x is not None, "No output tensor from pipeline"
    assert isinstance(sink.last_x, torch.Tensor), f"Expected Tensor, got {type(sink.last_x)}"
    assert sink.last_x.grad_fn is not None, "final_x has no grad_fn — computation graph is broken"

    loss = (sink.last_x - CONFIG.x_target) ** 2
    assert loss.grad_fn is not None, "loss has no grad_fn"

    print("  PASS: tensors have grad_fn, computation graph is alive")
    return True


def test_backward_gives_gradient():
    """Test 2: loss.backward() computes gradient for theta."""
    pipe, ball, sink = build_tensor_pipeline(CONFIG, THETA)

    for t in range(CONFIG.T + 5):
        pipe.step(dt=CONFIG.dt)

    loss = (sink.last_x - CONFIG.x_target) ** 2
    loss.backward()

    assert ball.theta.grad is not None, "No gradient for theta"
    grad = ball.theta.grad.item()
    assert abs(grad) > 1e-10, f"Gradient is essentially zero: {grad}"

    print(f"  PASS: gradient = {grad:.10f}")
    return grad


def test_reversed_graph():
    """Test 3: Reversed graph topology is correct."""
    pipe, _, _ = build_tensor_pipeline(CONFIG, THETA)
    graph = pipe.get_graph()

    reversed_adj = reverse_graph(graph)
    backward_order = get_backward_order(graph)

    # Forward: clock → ball → sink
    sources = graph.find_sources()
    sinks = graph.find_sinks()

    # In reversed graph, sinks should point to their predecessors
    for sink_id in sinks:
        preds = reversed_adj[sink_id]
        assert len(preds) > 0, f"Sink {sink_id} has no reversed predecessors"

    # Sources should have no reversed predecessors (they are backward sinks)
    for src_id in sources:
        preds = reversed_adj[src_id]
        assert len(preds) == 0, f"Source {src_id} should have no reversed preds, got {preds}"

    # Backward order should have sinks first, sources last
    assert backward_order[-1] in sources, f"Last in backward order should be a source"
    assert backward_order[0] in sinks, f"First in backward order should be a sink"

    print(f"  PASS: reversed graph correct")
    print(f"    forward sources: {sources}")
    print(f"    forward sinks: {sinks}")
    print(f"    backward order: {backward_order}")
    return True


def test_gradient_matches_trace_replay():
    """Test 4: Graph-level gradient matches the trace-replay method."""
    # Method A: Graph-level backprop (new approach)
    pipe, ball, sink = build_tensor_pipeline(CONFIG, THETA)
    for t in range(CONFIG.T + 5):
        pipe.step(dt=CONFIG.dt)

    loss_a = (sink.last_x - CONFIG.x_target) ** 2
    loss_a.backward()
    grad_a = ball.theta.grad.item()

    # Method B: Trace-replay (existing approach from bouncing_ball_pipeline.py)
    theta_b = torch.tensor(THETA, dtype=torch.float64, requires_grad=True)
    x = torch.tensor(CONFIG.x_init, dtype=torch.float64)
    v = theta_b.clone()

    for tick in range(CONFIG.T):
        v_pred = v - CONFIG.g * CONFIG.dt
        x_pred = x + v_pred * CONFIG.dt
        hit = (x_pred < 0.0).float()
        x = torch.where(hit > 0.5, x_pred * 0.0, x_pred)
        v = torch.where(hit > 0.5, -CONFIG.e * v_pred, v_pred)

    loss_b = (x - CONFIG.x_target) ** 2
    loss_b.backward()
    grad_b = theta_b.grad.item()

    diff = abs(grad_a - grad_b)
    print(f"  Graph-level gradient:  {grad_a:.10f}")
    print(f"  Trace-replay gradient: {grad_b:.10f}")
    print(f"  Difference:            {diff:.2e}")

    assert diff < 1e-6, f"Gradients differ by {diff:.2e} — too large"
    print(f"  PASS: gradients match (diff={diff:.2e})")
    return True


def test_determinism():
    """Test 5: Multiple runs give identical gradients."""
    grads = []
    for i in range(5):
        pipe, ball, sink = build_tensor_pipeline(CONFIG, THETA)
        for t in range(CONFIG.T + 5):
            pipe.step(dt=CONFIG.dt)
        loss = (sink.last_x - CONFIG.x_target) ** 2
        loss.backward()
        grads.append(ball.theta.grad.item())

    all_same = all(g == grads[0] for g in grads)
    print(f"  5 runs: {grads[0]:.10f}")
    print(f"  All identical: {all_same}")
    assert all_same, f"Gradients vary across runs: {grads}"
    print(f"  PASS: deterministic")
    return True


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    tests = [
        ("Test 1: Tensor pipeline preserves grad_fn", test_tensor_pipeline_grad_fn),
        ("Test 2: loss.backward() computes gradient", test_backward_gives_gradient),
        ("Test 3: Reversed graph topology", test_reversed_graph),
        ("Test 4: Gradient matches trace-replay", test_gradient_matches_trace_replay),
        ("Test 5: Determinism across runs", test_determinism),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"\n=== {name} ===")
        try:
            test_fn()
            passed += 1
        except Exception as ex:
            print(f"  FAIL: {ex}")
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 50}")
    sys.exit(1 if failed > 0 else 0)
