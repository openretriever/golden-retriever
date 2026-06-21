
import os
import sys
import time
import argparse
import logging
from dataclasses import dataclass
from typing import Optional, Any

import torch
import torch.nn as nn
import torch.optim as optim

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
src_root = os.path.join(project_root, "src")
if src_root not in sys.path:
    sys.path.insert(0, src_root)

from retriever.flow import io, Flow, Pipeline, Rate

logger = logging.getLogger(__name__)

# ============================================================================
# FLOW IO DEFINITIONS
# ============================================================================

@io
@dataclass
class SourceOutput:
    hidden_state: Any # torch.Tensor (Auto zero-copy)
    timestamp: float

@io
@dataclass
class SourceInput:
    gradient: Any # torch.Tensor (Auto zero-copy)

@io
@dataclass
class ComputeOutput:
    gradient: Any # torch.Tensor (Auto zero-copy)

@io
@dataclass
class ComputeInput:
    hidden_state: Any # torch.Tensor (Auto zero-copy)


# ============================================================================
# FLOWS
# ============================================================================

class SourceFlow(Flow[SourceInput, SourceOutput]):
    """
    Hosts the first part of the model (Part A).
    """
    def init(self):
        # Auto-detect device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
            
        print(f"[Source] Initializing on {self.device}")
        
        # Model Part A: Linear(10 -> 10)
        self.model = nn.Linear(10, 10).to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        
        self.iteration = 0
        self.last_output: Optional[torch.Tensor] = None

    def run(self, input: SourceInput) -> Optional[SourceOutput]:
        # Handle incoming gradient from previous step
        if input.gradient is not None:
            # We receive a Tensor immediately (deserialized by backend)
            grad = input.gradient.to(self.device)
            
            if self.last_output is not None:
                # Manual backward with received gradient
                self.optimizer.zero_grad()
                self.last_output.backward(grad)
                self.optimizer.step()
                
                if self.iteration % 10 == 0:
                    print(f"[Source] Step {self.iteration} completed. Grad mean: {grad.mean().item():.4f}")
                    
        # Generate new batch
        self.iteration += 1
        data = torch.randn(1, 10, device=self.device)
        
        # Forward Part A
        hidden = self.model(data)
        
        # Keep graph alive for backprop
        self.last_output = hidden
        
        return SourceOutput(
            hidden_state=hidden.detach(), # Just send the tensor!
            timestamp=time.time()
        )


class ComputeFlow(Flow[ComputeInput, ComputeOutput]):
    """
    Hosts the second part of the model (Part B).
    """
    def init(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
            
        print(f"[Compute] Initializing on {self.device}")
        
        # Model Part B: Linear(10 -> 1)
        self.model = nn.Linear(10, 1).to(self.device)
        
    def run(self, input: ComputeInput) -> Optional[ComputeOutput]:
        if input.hidden_state is None:
            return None
            
        # 1. Receive
        # The backend handles reconstruction. We just ensure it's on our device.
        hidden_val = input.hidden_state.to(self.device)
        
        # 2. Re-attach to graph (Leaf for this process)
        hidden_var = hidden_val.clone().detach().requires_grad_(True)
        
        # 3. Forward Part B
        output = self.model(hidden_var)
        
        # 4. Loss (Dummy target = 1.0)
        target = torch.ones_like(output)
        loss = nn.MSELoss()(output, target)
        
        if torch.rand(1).item() < 0.05: # Log occasionally
            print(f"[Compute] Loss: {loss.item():.4f}")
            
        # 5. Backward
        loss.backward()
        
        # 6. Send gradient back
        grad = hidden_var.grad
        if grad is None:
             return None
             
        # Just send the tensor!
        return ComputeOutput(
            gradient=grad
        )


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="PyTorch Async Split-Learning Example with Zero-Copy Support")
    parser.add_argument("--backend", default="dora", choices=["dora", "multiprocessing"], help="Execution backend")
    parser.add_argument("--duration", type=float, default=20.0, help="Duration to run the example")
    args = parser.parse_args()

    print(f"Starting PyTorch Async Example with backend: {args.backend}")
    print("Zero-copy optimization will be active if CUDA is detected and dora backend is used.")

    pipe = Pipeline("pytorch_cuda_async")

    with pipe:
        # 1. Source Node (Training loop Part A)
        source = SourceFlow() @ Rate(hz=100)
        
        # 2. Compute Node (Training loop Part B)
        compute = ComputeFlow() @ Rate(hz=100)
        
        # 3. Connections
        pipe.connect(source, compute, map={"hidden_state": "hidden_state"})
        pipe.connect(compute, source, map={"gradient": "gradient"})

    # Run
    try:
        pipe.run(backend=args.backend, duration=args.duration)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()
