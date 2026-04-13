"""Shared memory-oriented flows for the advanced example ladder."""

from __future__ import annotations

from retriever.flow import Flow
from retriever.types.perception import DetectionBatch, PointTarget2D

from examples.advanced.memory_examples.types import ObjectBelief, SceneBelief



class DetectionDropout(Flow[DetectionBatch, DetectionBatch]):
    """Deterministically hide one label every N frames to exercise memory."""

    def __init__(self, *, target_label: str = "red", every_n: int = 3) -> None:
        super().__init__()
        self.target_label = str(target_label)
        self.every_n = int(max(1, every_n))

    def init_config(self) -> dict:
        return {"target_label": self.target_label, "every_n": self.every_n}

    def step(self, batch: DetectionBatch) -> DetectionBatch:
        if batch.frame_index is None or batch.frame_index % self.every_n != 0:
            return batch
        kept = tuple(det for det in batch.detections if det.label != self.target_label)
        return DetectionBatch(detections=kept, header=batch.header, frame_index=batch.frame_index)


class BeliefTracker(Flow[DetectionBatch, SceneBelief]):
    """Keep a small stable scene belief from noisy or intermittent detections."""

    def __init__(self, *, image_width: int = 96, image_height: int = 72, hold_steps: int = 2, alpha: float = 0.35):
        super().__init__()
        self.image_width = float(image_width)
        self.image_height = float(image_height)
        self.hold_steps = int(max(0, hold_steps))
        self.alpha = float(alpha)

    def init_config(self) -> dict:
        return {
            "image_width": int(self.image_width),
            "image_height": int(self.image_height),
            "hold_steps": self.hold_steps,
            "alpha": self.alpha,
        }

    def init(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._memory: dict[str, ObjectBelief] = {}

    def step(self, batch: DetectionBatch) -> SceneBelief:
        if batch.frame_index is None:
            return SceneBelief()

        updated: dict[str, ObjectBelief] = {}
        seen_labels = {det.label for det in batch.detections}

        for det in batch.detections:
            prev = self._memory.get(det.label)
            x_norm = (det.centroid_x or 0.0) / max(self.image_width - 1.0, 1.0)
            y_norm = (det.centroid_y or 0.0) / max(self.image_height - 1.0, 1.0)
            confidence = det.confidence or 0.0
            seen_count = 1
            if prev is not None:
                x_norm = (1.0 - self.alpha) * prev.x_norm + self.alpha * x_norm
                y_norm = (1.0 - self.alpha) * prev.y_norm + self.alpha * y_norm
                confidence = max(prev.confidence * 0.8, confidence)
                seen_count = prev.seen_count + 1
            updated[det.label] = ObjectBelief(
                label=det.label,
                x_norm=x_norm,
                y_norm=y_norm,
                confidence=confidence,
                seen_count=seen_count,
                last_frame_index=batch.frame_index,
                missing_steps=0,
            )

        for label, prev in self._memory.items():
            if label in seen_labels:
                continue
            missing = prev.missing_steps + 1
            if missing > self.hold_steps:
                continue
            updated[label] = ObjectBelief(
                label=prev.label,
                x_norm=prev.x_norm,
                y_norm=prev.y_norm,
                confidence=prev.confidence * 0.85,
                seen_count=prev.seen_count,
                last_frame_index=prev.last_frame_index,
                missing_steps=missing,
            )

        self._memory = updated
        objects = tuple(sorted(updated.values(), key=lambda obj: obj.label))
        return SceneBelief(frame_index=batch.frame_index, objects=objects)


class BeliefPrinter(Flow[SceneBelief, None]):
    def step(self, belief: SceneBelief) -> None:
        if belief.frame_index is None:
            return None
        summary = [
            f"{obj.label}@({obj.x_norm:.2f},{obj.y_norm:.2f}) c={obj.confidence:.2f} seen={obj.seen_count} miss={obj.missing_steps}"
            for obj in belief.objects
        ]
        print(f"[frame={belief.frame_index:02d}] belief={summary}")
        return None


class SelectBeliefTarget(Flow[SceneBelief, PointTarget2D]):
    def __init__(self, *, target_label: str = "red") -> None:
        super().__init__()
        self.target_label = str(target_label)

    def init_config(self) -> dict:
        return {"target_label": self.target_label}

    def step(self, belief: SceneBelief) -> PointTarget2D:
        if belief.frame_index is None:
            return PointTarget2D()
        for obj in belief.objects:
            if obj.label != self.target_label:
                continue
            return PointTarget2D(
                frame_index=belief.frame_index,
                label=obj.label,
                x_norm=obj.x_norm,
                y_norm=obj.y_norm,
                confidence=obj.confidence,
            )
        return PointTarget2D(frame_index=belief.frame_index)
