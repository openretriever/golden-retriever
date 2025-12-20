from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class FRPConfig:
    """Unified FRP configuration used across the framework.

    - rate: allows either human-friendly strings ("30hz", "50ms", "1s") or numeric Hz
    - trigger/feedback: declarative coordination hints
    - buffer_size/reactive: optional knobs for coordination policies
    """

    rate: Optional[Union[str, float]] = None
    trigger: Optional[str] = None
    feedback_from: Optional[str] = None
    feedback_to: Optional[str] = None
    buffer_size: int = 100
    reactive: bool = False

    def rate_hz(self) -> Optional[float]:
        """Return rate as Hz if available, parsing string formats when needed."""
        r = self.rate
        if r is None:
            return None
        if isinstance(r, (int, float)):
            return float(r)
        rs = str(r).strip().lower()
        try:
            if rs.endswith("hz"):
                return float(rs[:-2])
            if rs.endswith("ms"):
                ms = float(rs[:-2])
                return 1000.0 / ms if ms > 0 else None
            if rs.endswith("s"):
                s = float(rs[:-1])
                return 1.0 / s if s > 0 else None
            # Fallback: try raw number (assume Hz)
            return float(rs)
        except Exception:
            return None

