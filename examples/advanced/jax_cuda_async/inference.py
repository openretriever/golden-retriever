
import os
import sys
import time
import argparse
import logging
from dataclasses import dataclass

try:
    import jax
    import jax.numpy as jnp
    import flax.linen as nn
    import numpy as np
except ImportError:
    print("JAX/Flax not installed. Skipping example.")
    sys.exit(0)

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
src_root = os.path.join(project_root, "src")
if src_root not in sys.path:
    sys.path.insert(0, src_root)

from retriever.flow import flow_io, Flow, Pipeline, Rate
# Import the new JAX library support

# Import the new JAX library support
from retriever.lib.jax import from_jax, JaxIO

# Suppress JAX/ABSL logging
logging.getLogger('jax').setLevel(logging.WARNING)
logging.getLogger('absl').setLevel(logging.WARNING)

# Configure basic logging for the example if not already set
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

# ============================================================================
# MODEL DEFINITION
# ============================================================================

class SimpleMLP(nn.Module):
    """A simple Multi-Layer Perceptron."""
    features: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.features * 2)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.features)(x)
        return x

# ============================================================================
# CUSTOM SOURCE/SINK FLOWS
# ============================================================================

@flow_io
@dataclass
class Nothing:
    pass

class DataGen(Flow[Nothing, JaxIO]):
    """Generates random JAX arrays."""
    def run(self, input: Nothing) -> JaxIO:
        # Generate random data (1, 10)
        # We use numpy for generation usually, then convert to jnp, 
        # or use jax.random if we have a key.
        # For simplicity in this demo, strict JAX usage:
        
        # Note: In a real app you'd manage RNG keys carefully.
        # Here we just use time-seeded for demo variation.
        key = jax.random.PRNGKey(int(time.time() * 1000) % 1000000)
        data = jax.random.normal(key, (1, 10))
        
        return JaxIO(inp=data)

class PrintSink(Flow[JaxIO, Nothing]):
    """Prints the output."""
    def run(self, input: JaxIO) -> None:
        if input.inp is not None:
             print(f"[Sink] Received shape {input.inp.shape}, mean {jnp.mean(input.inp):.4f}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="JAX/Flax Inference Example via Retriever")
    parser.add_argument("--backend", default="dora", choices=["dora", "multiprocessing"], help="Execution backend")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration to run the example")
    args = parser.parse_args()

    print(f"Starting JAX Inference Example with backend: {args.backend}")

    pipe = Pipeline("jax_inference")

    with pipe:
        # 1. Source: Generate Data
        source = DataGen() @ Rate(hz=10)
        
        # 2. Process: Flax Model
        # Create the module
        model = SimpleMLP(features=10)
        # We need a sample input for 'from_jax' to establish input shapes
        # Use simple numpy array to ensure safe pickling/transport across processes
        sample_input = np.ones((1, 10))
        
        mlp_flow = from_jax(model, sample_input) @ Rate(hz=10)
        
        # 3. Sink: Print
        sink = PrintSink() @ Rate(hz=10)
        
        # Connect
        pipe.connect(source, mlp_flow, map={"inp": "inp"})
        pipe.connect(mlp_flow, sink, map={"inp": "inp"})

    # Run
    try:
        pipe.run(backend=args.backend, duration=args.duration)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()
