
import os
import sys
import time
import argparse
import logging
from dataclasses import dataclass
from typing import Optional, Any, Tuple

try:
    import jax
    import jax.numpy as jnp
    import flax.linen as nn
    import optax
except ImportError:
    print("JAX/Flax/Optax not installed. Skipping example.")
    sys.exit(0)

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
src_root = os.path.join(project_root, "src")
if src_root not in sys.path:
    sys.path.insert(0, src_root)

from retriever.flow import flow_io, Flow, Pipeline, Rate
from retriever.lib.jax import JaxIO, JaxSplitOptimizer, JaxRemoteGrad

logger = logging.getLogger(__name__)

# ============================================================================
# FLOW IO DEFINITIONS
# ============================================================================

@flow_io
@dataclass
class SourceOutput:
    hidden_state: Any # jax.numpy.ndarray (Auto zero-copy)
    timestamp: float

@flow_io
@dataclass
class SourceInput:
    gradient: Any # jax.numpy.ndarray (Auto zero-copy)

@flow_io
@dataclass
class ComputeOutput:
    gradient: Any # jax.numpy.ndarray (Auto zero-copy)

@flow_io
@dataclass
class ComputeInput:
    hidden_state: Any # jax.numpy.ndarray (Auto zero-copy)


# ============================================================================
# MODELS
# ============================================================================

class PartA(nn.Module):
    """First half of the model (runs on Source)"""
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=32)(x)
        x = nn.relu(x)
        x = nn.Dense(features=32)(x)
        return x

class PartB(nn.Module):
    """Second half of the model (runs on Compute)"""
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=32)(x)
        x = nn.relu(x)
        x = nn.Dense(features=1)(x)
        return x

# ============================================================================
# FLOWS
# ============================================================================

class SourceFlow(Flow[SourceInput, SourceOutput]):
    """
    Hosts Part A of the model and the Optimizer.
    """
    def init(self):
        print("[Source] Initializing JAX Part A...")
        self.model = PartA()
        
        # Initialize params
        rng = jax.random.PRNGKey(0)
        rng, init_rng = jax.random.split(rng)
        dummy_input = jnp.ones((1, 10))
        self.params = self.model.init(init_rng, dummy_input)
        
        # Initialize Optimizer (Adam)
        self.optimizer_def = optax.adam(learning_rate=0.01)
        self.optimizer_helper = JaxSplitOptimizer(self.optimizer_def, self.params)
        
        self.iteration = 0
        
        # We need a function to compute vector-Jacobian product for the backward pass
        # Since we only get the gradient wrt output, we need to backpropagate it 
        # through Part A to update Part A's params.
        # But wait! In split learning, standard backprop updates weights based on 
        # gradients flowing back.
        # So we need to differentiate Part A wrt its params, given the incoming grad.
        
        @jax.jit
        def train_step_source(params, x, grad_from_compute, opt_state):
            # We want to compute gradients of the outputs wrt params, 
            # and multiply by grad_from_compute (chain rule).
            # This is equivalent to saying: Loss = (Output * grad_from_compute).sum()
            # Gradient of Loss wrt Params is what we want.
            
            def surrogate_loss(p):
                y = self.model.apply(p, x)
                # VJP approach: y_bar * dy/dp
                return jnp.sum(y * grad_from_compute)
                
            grads = jax.grad(surrogate_loss)(params)
            
            # Apply updates
            updates, new_opt_state = self.optimizer_def.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            
            return new_params, new_opt_state

        self.train_step = train_step_source
        self.last_input_data = None # Store input for backprop
        
    def run(self, input: SourceInput) -> Optional[SourceOutput]:
        # Handle incoming gradient from previous step
        if input.gradient is not None and self.last_input_data is not None:
             grad = input.gradient
             
             # Perform update
             self.params, self.optimizer_helper.opt_state = self.train_step(
                 self.params, 
                 self.last_input_data, 
                 grad, 
                 self.optimizer_helper.opt_state
             )
             
             if self.iteration % 10 == 0:
                 print(f"[Source] Step {self.iteration} update complete. Grad mean: {jnp.mean(grad):.4f}")

        # Generate new batch
        self.iteration += 1
        rng = jax.random.PRNGKey(self.iteration)
        data = jax.random.normal(rng, (1, 10))
        self.last_input_data = data
        
        # Forward Part A
        hidden = self.model.apply(self.params, data)
        
        return SourceOutput(
            hidden_state=hidden,
            timestamp=time.time()
        )


class ComputeFlow(Flow[ComputeInput, ComputeOutput]):
    """
    Hosts Part B of the model.
    """
    def init(self):
        print("[Compute] Initializing JAX Part B...")
        self.model = PartB()
        
        rng = jax.random.PRNGKey(1)
        dummy_input = jnp.ones((1, 32)) # Output of Part A is 32 dim
        self.params = self.model.init(rng, dummy_input)
        
        # Helper for gradients
        # We need to compute gradient w.r.t input (hidden_state)
        # to send back to Source.
        
        @jax.jit
        def compute_grad_wrt_input(params, hidden_state, target):
            def loss_fn(h):
                pred = self.model.apply(params, h)
                return jnp.mean((pred - target) ** 2)
                
            return jax.value_and_grad(loss_fn)(hidden_state)
            
        self.compute_grad = compute_grad_wrt_input

    def run(self, input: ComputeInput) -> Optional[ComputeOutput]:
        if input.hidden_state is None:
            return None
            
        hidden_val = input.hidden_state
        
        # Dummy target
        target = jnp.ones((1, 1))
        
        # Compute loss and gradient wrt input
        loss, grad_wrt_input = self.compute_grad(self.params, hidden_val, target)
        
        if jax.random.uniform(jax.random.PRNGKey(int(time.time()))) < 0.05:
            print(f"[Compute] Loss: {loss:.4f}")
            
        # Send gradient back
        return ComputeOutput(
            gradient=grad_wrt_input
        )

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="JAX Async Split-Learning Example")
    parser.add_argument("--backend", default="dora", choices=["dora", "multiprocessing"], help="Execution backend")
    parser.add_argument("--duration", type=float, default=20.0, help="Duration to run the example")
    args = parser.parse_args()

    print(f"Starting JAX Async Example with backend: {args.backend}")
    print("Zero-copy optimization will be active if backend supports it.")

    pipe = Pipeline("jax_async_train")

    with pipe:
        # 1. Source Node (Training loop Part A)
        source = SourceFlow() @ Rate(hz=50)
        
        # 2. Compute Node (Training loop Part B)
        compute = ComputeFlow() @ Rate(hz=50)
        
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
