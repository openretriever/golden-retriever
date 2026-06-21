# VLM-Compatible Perception Flow
# Passes webcam frames through without grid-world domain processing

from typing import Optional, Set
from dataclasses import dataclass

from retriever.flow import Flow, io
from retriever.types.symbolic import State, GroundAtom

from ..types.flow_types import PerceptionInput, PerceptionOutput


@io
@dataclass
class VLMPerceptionInput:
    """Input for VLM perception - just raw camera data."""

    data: dict  # {"rgb": bytes, "timestamp": float}


@io
@dataclass
class VLMPerceptionOutput:
    """Output for VLM perception - passes through raw observation."""

    state: Optional[State] = None
    atoms: Set[GroundAtom] = None
    raw_observation: Optional[dict] = None  # Pass through for VLM flows

    def __post_init__(self):
        if self.atoms is None:
            self.atoms = set()


class VLMPerceptionFlow(Flow[VLMPerceptionInput, VLMPerceptionOutput]):
    """VLM-compatible perception that passes webcam data through.

    Unlike the grid-world PerceptionFlow, this doesn't try to extract
    robot/key/door positions. It simply wraps the raw camera data
    for downstream VLM flows.
    """

    def __init__(self, name: str = "VLMPerception"):
        self.name = name

    def init(self):
        print(f"[{self.name}] Initialized VLM perception")

    def step(self, inp: VLMPerceptionInput) -> VLMPerceptionOutput:
        data = inp.data

        if not data:
            rr = self.rr
            if rr:
                rr.log("perception/status", rr.TextLog("Empty input"))
            return VLMPerceptionOutput(state=None, atoms=set(), raw_observation=None)

        # Extract frame from data
        frame = data.get("rgb") or data.get("frame")
        timestamp = data.get("timestamp", 0.0)

        if frame is None:
            rr = self.rr
            if rr:
                rr.log("perception/status", rr.TextLog("No frame in data"))
            return VLMPerceptionOutput(state=None, atoms=set(), raw_observation=None)

        # Log frame to Rerun for visualization
        try:
            from PIL import Image
            import io as iolib
            import numpy as np

            image = Image.open(iolib.BytesIO(frame))
            rr = self.rr
            if rr:
                # Set timeline so images are properly ordered (not overwritten)
                if hasattr(rr, "set_time_seconds"):
                    rr.set_time_seconds("log_time", timestamp)
                else:
                    rr.set_time("log_time", timestamp=timestamp)
                rr.log("camera/webcam", rr.Image(np.array(image)))
                rr.log(
                    "perception/status",
                    rr.TextLog(f"Frame received ({len(frame)} bytes)"),
                )
        except Exception as e:
            rr = self.rr
            if rr:
                rr.log("perception/status", rr.TextLog(f"Frame decode error: {e}"))

        # Create minimal state with raw observation for VLM flows
        state = State(data={"frame_bytes": frame, "timestamp": timestamp})

        # Pass raw observation through for VLM planner/monitor
        raw_obs = {"frame": frame, "timestamp": timestamp}

        return VLMPerceptionOutput(state=state, atoms=set(), raw_observation=raw_obs)
