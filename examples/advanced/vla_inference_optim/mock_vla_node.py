"""
OpenPI VLA Flow Component.

Wraps the `openpi` library (specifically pi0.5) to provide VLA inference.
Implements specific optimizations:
1. Future-State Awareness (VLASH-style): Predicts robot state at execution time
   to account for inference latency.
"""
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from datetime import datetime


from retriever.flow import Flow, io

logger = logging.getLogger(__name__)

@io
@dataclass
class VLAInput:
    """Input to the VLA model."""
    instruction: str
    image: Any  # torch.Tensor or PIL.Image or numpy.ndarray
    state: Dict[str, Any] # Robot state (q, dq, ee_pose, etc.)
    timestamp: float

@io
@dataclass
class VLAAction:
    """Output action from VLA."""
    action: np.ndarray # (7,) or (H, 7)
    latency: float     # Inference latency in seconds
    timestamp: float   # Generation timestamp (Observation time)
    dt: float = 0.1    # Time step between actions

class MockVLAFlow(Flow[VLAInput, VLAAction]):
    """
    Mock VLA Node that simulates variable inference latency and outputs action chunks.
    
    Behavior:
    - Simulates inference time with random jitter.
    - Outputs a trajectory chunk (Horizon=6).
    - Adapts to the input rate (backpressure) via pipeline configuration.
    """

    def __init__(self, model_id: str = "mock"):
        super().__init__()
        self.model_id = model_id
        
        self.device = "cpu"
            
        self._last_inference_latency = 0.05 # Initial guess (50ms)

    def init(self):
        """Lazy load the model."""
        logger.info(f"Initializing Mock VLA ({self.model_id})...")
        # No real model loading

    def _predict_future_state(self, current_state: Dict[str, Any], dt: float) -> Dict[str, Any]:
        """Simple linear extrapolation for state prediction (Mock)."""
        # In a real model, we'd apply kinematics.
        # Here we just pass through or slightly perturb.
        return current_state

    def run(self, inp: VLAInput) -> Optional[VLAAction]:
        start_time = time.time()

        # 1. Future-State Awareness
        # Predict state at (now + expected_latency)
        # We use a moving average or just the last latency as prediction
        expected_finish_time = start_time + self._last_inference_latency
        
        if inp.timestamp is None:
            logger.warning("Input timestamp is None! Defaulting to current time.")
            inp.timestamp = start_time
            
        dt = expected_finish_time - inp.timestamp

        # Clamp dt to be non-negative
        dt = max(0.0, dt)

        inference_state = self._predict_future_state(inp.state, dt)

        logger.debug(f"Input time: {inp.timestamp:.3f}, Est finish: {expected_finish_time:.3f}, dt: {dt:.3f}s. Pred State: {inference_state is not None}")

        # 2. Mock Inference with Jitter
        # Target ~10Hz (100ms) -> Range 100ms to 200ms
        # Base 150ms + Random jitter (+- 50ms)
        jitter = np.random.uniform(-0.05, 0.05)
        sleep_time = max(0.01, 0.15 + jitter)
        time.sleep(sleep_time) 
        
        # Generate Chunk: (Horizon, 7)
        # Simulating a smooth trajectory chunk
        # Horizon = 32 actions (approx 3.2s at 10Hz)
        horizon = 32 
        dt = 0.1
        
        # Create a simple sine wave trajectory to make it easier to visualize alignment
        t_chunk = np.linspace(0, horizon*dt, horizon)
        # Phase shift based on start_time so it looks continuous-ish
        phase = start_time
        action_vec = np.zeros((horizon, 7))
        # Sine wave on 1st dimension
        action_vec[:, 0] = np.sin(t_chunk + phase)

        end_time = time.time()
        latency = end_time - start_time

        # Update latency estimate (EMA)
        alpha = 0.2
        self._last_inference_latency = alpha * latency + (1 - alpha) * self._last_inference_latency
        
        # Calculate effective rate
        effective_hz = 1.0 / (time.time() - start_time)
        if hasattr(self, '_last_log_time') and (time.time() - self._last_log_time > 1.0):
             logger.info(f"MockVLA Rate: {effective_hz:.1f} Hz (Actual: {latency*1000:.1f}ms, Est: {self._last_inference_latency*1000:.1f}ms) ts={datetime.fromtimestamp(time.time()).strftime('%H:%M:%S.%f')[:-3]}")
             self._last_log_time = time.time()
        elif not hasattr(self, '_last_log_time'):
             self._last_log_time = time.time()

        return VLAAction(
            action=action_vec,
            latency=latency,
            timestamp=inp.timestamp, # Use input timestamp (t_obs) as the reference!
            dt=dt
        )
