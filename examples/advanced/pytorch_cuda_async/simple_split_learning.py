"""
Unified Split Learning Tutorial (Refactored)
============================================
Demonstrates **Distributed Backpropagation** using the High-Level API.

Architecture:
  [Source Node A] --(activations)--> [Compute Node B]
        ^                                  |
        |________(gradients)_______________|

API Features:
- `retriever.connect`: Global wiring.
- `retriever.run`: Global execution.
- `SplitOptimizer` / `RemoteAutograd`: Logic helpers.

Run:
    pixi run python examples/advanced/pytorch_cuda_async/simple_split_learning.py
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from typing import Optional, Any

import torch
import torch.nn as nn
import torch.optim as optim

# Project root setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
src_root = os.path.join(project_root, "src")
if src_root not in sys.path:
    sys.path.insert(0, src_root)

import retriever
from retriever.flow import io, Flow, Rate
from retriever.lib.torch import SplitOptimizer, RemoteAutograd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SplitUnified")

# ============================================================================
# DATA TYPES
# ============================================================================

@io
@dataclass
class Activations:
    data: Any # torch.Tensor
    batch_index: int

@io
@dataclass
class Gradients:
    grads: Any # torch.Tensor
    batch_index: int

# ============================================================================
# FLOWS (Logic Nodes)
# ============================================================================

class Part1Flow(Flow[Gradients, Activations]):
    """ Node A: Source """
    def init(self):
        self.device = torch.device("cpu") # Keep simple for demo
        self.model = nn.Linear(10, 10).to(self.device)
        base_optim = optim.SGD(self.model.parameters(), lr=0.1)
        
        # Helper: Handles Saving/Detaching/Stepping
        self.split_optim = SplitOptimizer(base_optim, verbose=True)
        self.batch_idx = 0

    def run(self, input_grads: Gradients) -> Optional[Activations]:
        # 1. HANDLE BACKWARD
        if input_grads is not None and input_grads.grads is not None:
            # One-liner to handle backward step
            if self.split_optim.backward_pass(input_grads.batch_index, input_grads.grads):
                logger.info(f"[Node A] Updated weights for Batch {input_grads.batch_index}")

        # 2. HANDLE FORWARD
        self.batch_idx += 1
        input_data = torch.randn(1, 10, device=self.device)
        output = self.model(input_data)
        
        # Helper: Handles detaching and saving for backward
        safe_output = self.split_optim.forward_pass(self.batch_idx, output)
        
        logger.info(f"[Node A] Sending Batch {self.batch_idx}")
        return Activations(data=safe_output, batch_index=self.batch_idx)


class Part2Flow(Flow[Activations, Gradients]):
    """ Node B: Compute """
    def init(self):
        self.device = torch.device("cpu")
        self.model = nn.Linear(10, 1).to(self.device)
        self.criterion = nn.MSELoss()
        
    def run(self, input_act: Activations) -> Optional[Gradients]:
        if input_act.data is None: return None
        
        # 1. ATTACH
        input_tensor = RemoteAutograd.attach(input_act.data.to(self.device))
        
        # 2. FORWARD
        output = self.model(input_tensor)
        loss = self.criterion(output, torch.zeros_like(output))
        logger.info(f"[Node B] Loss: {loss.item():.4f}")
        
        # 3. BACKWARD + RETURN GRAD
        grad_to_send = RemoteAutograd.backward_and_return_grad(loss, input_tensor)
        
        return Gradients(grads=grad_to_send, batch_index=input_act.batch_index)

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="dora", choices=["dora", "multiprocessing"])
    parser.add_argument("--duration", type=float, default=2.0)
    args = parser.parse_args()
    
    print("Creating flows...")
    node_a = Part1Flow() @ Rate(hz=10)
    node_b = Part2Flow() @ Rate(hz=10)
    
    print("Connecting graph (Global DSL)...")
    # A -> B
    retriever.connect(node_a, node_b, map={"data": "data", "batch_index": "batch_index"})
    # B -> A
    retriever.connect(node_b, node_a, map={"grads": "grads", "batch_index": "batch_index"})
    
    print("Running Unified Split Learning...")
    try:
        retriever.run(backend=args.backend, duration=args.duration)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
