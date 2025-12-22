"""
Example: Interactive Pipeline Debugging with Rerun

This example demonstrates how to use retriever's Rerun integration
to visualize and debug pipeline execution step-by-step.

Features:
- Record pipeline execution to MCAP file
- Live stream to Rerun viewer during recording
- View recorded MCAP files in Rerun

Run:
    pixi run python examples/advanced/rerun_step_debug/app.py
"""

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

import retriever
from retriever.flow import Flow, Rate, flow_io
from retriever.lib.rerun import rerun_loggable

# =============================================================================
# Define Loggable Types with @rerun_loggable decorator
# =============================================================================


@rerun_loggable({"image": "Image", "brightness": "Scalar"})
@flow_io
@dataclass
class ImageWithMetrics:
    """Image with computed metrics - each field logs separately in Rerun."""

    image: Optional[np.ndarray] = None
    brightness: Optional[float] = None
    timestamp: Optional[float] = None


@rerun_loggable({"detection_count": "Scalar", "confidence": "Scalar"})
@flow_io
@dataclass
class DetectionMetrics:
    """Detection results as metrics for timeline visualization."""

    detection_count: Optional[int] = None
    confidence: Optional[float] = None
    labels: Optional[str] = None  # Auto-detected as Text


@rerun_loggable({"reward": "Scalar", "episode": "Scalar"})
@flow_io
@dataclass
class RewardSignal:
    """Reward signal over time - good for RL debugging."""

    reward: Optional[float] = None
    episode: Optional[int] = None
    info: Optional[str] = None


# =============================================================================
# Flows
# =============================================================================


class ImageGeneratorFlow(Flow[None, ImageWithMetrics]):
    """Generates synthetic images with varying brightness."""

    def __init__(self):
        super().__init__()
        self.step = 0

    def run(self, _) -> ImageWithMetrics:
        # Generate synthetic image with varying pattern
        t = self.step * 0.1
        img = np.zeros((100, 100, 3), dtype=np.uint8)

        # Moving gradient
        for i in range(100):
            for j in range(100):
                val = int(128 + 127 * np.sin(i * 0.1 + t) * np.cos(j * 0.1 + t))
                img[i, j] = [val, val // 2, 255 - val]

        brightness = float(np.mean(img))
        self.step += 1

        return ImageWithMetrics(image=img, brightness=brightness, timestamp=time.time())


class MockDetectorFlow(Flow[ImageWithMetrics, DetectionMetrics]):
    """Mock detector that produces varying detection counts."""

    def run(self, input_data: ImageWithMetrics) -> DetectionMetrics:
        if input_data.image is None:
            return DetectionMetrics()

        # Mock detection based on image brightness
        brightness = input_data.brightness or 0.0
        count = int((brightness / 255.0) * 5)
        confidence = 0.5 + 0.5 * (brightness / 255.0)

        return DetectionMetrics(
            detection_count=count, confidence=confidence, labels=f"object_{count}"
        )


class RewardComputerFlow(Flow[DetectionMetrics, RewardSignal]):
    """Computes reward based on detections."""

    def __init__(self):
        super().__init__()
        self.episode = 0
        self.step_in_episode = 0

    def run(self, input_data: DetectionMetrics) -> RewardSignal:
        if input_data.detection_count is None:
            return RewardSignal()

        # Reward proportional to detections
        reward = float(input_data.detection_count) * 0.1

        # Simulate episode resets
        self.step_in_episode += 1
        if self.step_in_episode >= 20:
            self.episode += 1
            self.step_in_episode = 0

        return RewardSignal(
            reward=reward, episode=self.episode, info=f"step_{self.step_in_episode}"
        )


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 60)
    print("Rerun Step Debug Example")
    print("=" * 60)
    print()
    print("This example demonstrates step-by-step pipeline debugging.")
    print("Recording to MCAP + live streaming to Rerun viewer.")
    print()

    # Reset default pipeline for clean state
    p = retriever.reset_default_pipeline()

    # Build pipeline using >> operator
    with p:
        generator = ImageGeneratorFlow() @ Rate(hz=10)
        detector = MockDetectorFlow() @ Rate(hz=10)
        reward = RewardComputerFlow() @ Rate(hz=10)

        generator >> detector >> reward

    print("Running 50 steps with MCAP recording + Rerun streaming...")
    print()

    # Use record() with visualize=True for live Rerun visualization
    try:
        p.record(
            "step_debug_session.mcap",
            steps=50,
            dt=0.1,
            visualize=True,  # Live Rerun visualization
        )
    finally:
        p.close_stepper()

    print()
    print("Done! Recording saved to step_debug_session.mcap")
    print()
    print("To view later:")
    print("  import retriever")
    print("  retriever.view('step_debug_session.mcap')")


if __name__ == "__main__":
    main()
