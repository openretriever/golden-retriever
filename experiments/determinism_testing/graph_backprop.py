#!/usr/bin/env python3
"""
Graph-Level Gradient Backpropagation through Retriever Pipeline.

Instead of replaying traces through a dedicated TraceCollectorFlow,
this module:
  1. Runs the forward pipeline with tensor-valued outputs (grad graph alive)
  2. Reverses the pipeline DAG edges
  3. Calls loss.backward() — PyTorch autograd propagates through the graph

Key insight: The in-process backend (PipelineStepper) passes data by reference
through InMemoryChannel. No serialization, no .detach(). So the PyTorch
computation graph survives across Flow boundaries.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from retriever.flow.graph import PipelineGraph


# =============================================================================
# TensorStore — Per-Node Activation Cache
# =============================================================================

class TensorStore:
    """Stores intermediate tensor activations keyed by (node_id, tick)."""

    def __init__(self):
        self._store: Dict[Tuple[str, int], Dict[str, torch.Tensor]] = {}

    def save(self, node_id: str, tick: int, tensors: Dict[str, torch.Tensor]):
        self._store[(node_id, tick)] = tensors

    def get(self, node_id: str, tick: int) -> Optional[Dict[str, torch.Tensor]]:
        return self._store.get((node_id, tick))

    def get_all_ticks(self, node_id: str) -> List[int]:
        return sorted(t for (nid, t) in self._store if nid == node_id)

    def clear(self):
        self._store.clear()


# =============================================================================
# Graph Reversal Utilities
# =============================================================================

def reverse_graph(graph: PipelineGraph) -> Dict[str, List[str]]:
    """
    Reverse all edges in a PipelineGraph.

    For each edge A.port → B.port, produces B → A in the reversed adjacency list.
    Used for backward traversal order.
    """
    reversed_adj: Dict[str, List[str]] = {nid: [] for nid in graph.nodes}
    for edge in graph.edges:
        if edge.src_node not in reversed_adj[edge.dst_node]:
            reversed_adj[edge.dst_node].append(edge.src_node)
    return reversed_adj


def get_backward_order(graph: PipelineGraph) -> List[str]:
    """Get nodes in reverse topological order (sinks first, sources last)."""
    groups = graph.get_topological_groups()
    forward_order = [nid for group in groups for nid in group]
    return list(reversed(forward_order))


# =============================================================================
# GraphBackprop — The Backward Pass Engine
# =============================================================================

class GraphBackprop:
    """
    Manages forward tensor storage and backward pass through a pipeline graph.

    For the in-process backend, PyTorch autograd handles gradient propagation
    automatically since tensors are passed by reference. This class provides:
    - Graph structure analysis (reversed edges, backward order)
    - TensorStore for intermediate activations
    - Verification utilities
    """

    def __init__(self, graph: PipelineGraph):
        self.graph = graph
        self.store = TensorStore()
        self._reversed = reverse_graph(graph)
        self._backward_order = get_backward_order(graph)

    @property
    def backward_order(self) -> List[str]:
        return self._backward_order

    @property
    def reversed_adj(self) -> Dict[str, List[str]]:
        return self._reversed

    def verify_grad_graph(self, tensor: torch.Tensor, name: str = "tensor") -> bool:
        """Check if a tensor has a grad_fn (is part of a computation graph)."""
        has_grad = tensor.grad_fn is not None
        if not has_grad:
            print(f"  WARNING: {name} has no grad_fn — autograd will not propagate")
        return has_grad


# =============================================================================
# Bouncing Ball Demo: Graph-Level Backprop
# =============================================================================

def run_bouncing_ball_graph_backprop(
    theta_val: float = 3.0,
    T: int = 100,
    dt: float = 0.01,
    g: float = 9.81,
    e: float = 0.8,
    x_init: float = 1.0,
    x_target: float = 0.5,
    verbose: bool = True,
) -> Tuple[float, float]:
    """
    Run bouncing ball through Retriever pipeline with graph-level backprop.

    Returns:
        (gradient, loss) computed via loss.backward() through the pipeline graph
    """
    from bouncing_ball_pipeline import (
        PhysicsConfig, build_tensor_pipeline,
    )

    config = PhysicsConfig(g=g, e=e, dt=dt, T=T, x_target=x_target, x_init=x_init)
    pipe, ball_flow, tensor_sink = build_tensor_pipeline(config, theta_val)

    # Analyze graph structure
    graph = pipe.get_graph()
    backprop = GraphBackprop(graph)

    if verbose:
        print(f"Pipeline graph: {graph}")
        print(f"Backward order: {backprop.backward_order}")
        print(f"Reversed adjacency: {backprop.reversed_adj}")

    # Forward pass via Pipeline.step()
    for t in range(T + 5):
        pipe.step(dt=dt)

    # Get final tensor from sink
    final_x = tensor_sink.last_x
    if final_x is None:
        raise RuntimeError("Pipeline produced no output tensor")

    if verbose:
        print(f"\nFinal x tensor: {final_x}")
        print(f"  grad_fn: {final_x.grad_fn}")
        backprop.verify_grad_graph(final_x, "final_x")

    # Compute loss
    loss = (final_x - x_target) ** 2

    if verbose:
        print(f"Loss: {loss.item():.6f}")
        print(f"  grad_fn: {loss.grad_fn}")

    # Backward — PyTorch autograd propagates through the entire graph!
    loss.backward()

    gradient = ball_flow.theta.grad
    if gradient is None:
        raise RuntimeError("No gradient computed for theta")

    grad_val = gradient.item()
    loss_val = loss.item()

    if verbose:
        print(f"\ndL/dtheta = {grad_val:.10f}")
        print(f"Loss = {loss_val:.6f}")

    return grad_val, loss_val


if __name__ == "__main__":
    print("=" * 60)
    print("GRAPH-LEVEL BACKPROPAGATION THROUGH RETRIEVER PIPELINE")
    print("=" * 60)

    grad, loss = run_bouncing_ball_graph_backprop(verbose=True)

    print("\n" + "=" * 60)
    print(f"Result: gradient={grad:.10f}, loss={loss:.6f}")
    print("=" * 60)
