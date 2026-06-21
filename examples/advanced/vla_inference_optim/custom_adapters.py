from dataclasses import dataclass
from typing import Optional, Any, Iterable
import time
import numpy as np

from retriever.flow.adapter import Adapter, Chunking, register_adapter
from retriever.flow import io
from retriever.flow.types import EventBuffer
try:
    from .mock_vla_node import VLAAction, VLAInput
except ImportError:
    from mock_vla_node import VLAAction, VLAInput

@io
@dataclass
class SingleAction:
    action: np.ndarray
    timestamp: float


@register_adapter("actionchunking")
@dataclass
class ActionChunking(Chunking[VLAAction]):
    """
    VLA-specific chunking adapter with interpolation support.

    Extends the native `Chunking` adapter for VLAAction format,
    adding linear interpolation between chunk steps.

    Use Case: VLA models output action chunks (10 steps at 10Hz).
    A 200Hz robot controller samples smooth intermediate values.
    """
    buffer_size: int = 10
    dt: float = 0.1  # Time between steps in the chunk

    def __call__(
        self, buffer: EventBuffer[VLAAction], now: Optional[float] = None, **kwargs
    ) -> Optional[np.ndarray]:
        if not buffer:
            return None

        current_time = time.time() if now is None else now

        # Get latest chunk
        ts, chunk = buffer[-1]

        # Extract timing and actions from VLAAction
        if hasattr(chunk, 'timestamp') and chunk.timestamp is not None:
            chunk_start = chunk.timestamp
            actions = chunk.action
        else:
            chunk_start = ts
            actions = chunk

        # Calculate temporal offset
        delta_t = current_time - chunk_start
        if delta_t < 0:
            return None

        k_float = delta_t / self.dt
        k = int(k_float)

        if not isinstance(actions, (list, np.ndarray)):
            actions = np.array(actions)

        n_steps = len(actions)
        if k >= n_steps:
            return None  # Chunk exhausted

        # Linear interpolation between steps
        if k + 1 < n_steps:
            alpha = k_float - k
            val = (1.0 - alpha) * np.array(actions[k]) + alpha * np.array(actions[k + 1])
        else:
            val = np.array(actions[k])

        return val

    def _extract_array(self, value: VLAAction) -> Optional[Iterable]:
        """Override to handle VLAAction format."""
        if hasattr(value, 'action'):
            return value.action
        return super()._extract_array(value)



