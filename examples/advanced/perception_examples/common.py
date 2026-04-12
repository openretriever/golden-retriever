"""Shared synthetic perception payloads and flows for advanced examples."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from retriever.flow import Flow, flow_io


@flow_io
@dataclass(frozen=True)
class Frame2D:
    image: np.ndarray | None = None
    frame_id: int | None = None
    t_sim: float | None = None


@dataclass(frozen=True)
class BBox2D:
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class Detection2D:
    label: str
    confidence: float
    bbox: BBox2D
    centroid_x: float
    centroid_y: float
    pixel_count: int


@flow_io
@dataclass(frozen=True)
class DetectionBatch:
    frame_id: int | None = None
    detections: tuple[Detection2D, ...] = ()


@flow_io
@dataclass(frozen=True)
class SegmentationView:
    frame_id: int | None = None
    labels: tuple[str, ...] = ()
    pixel_counts: dict[str, int] = field(default_factory=dict)
    centroids: dict[str, tuple[float, float]] = field(default_factory=dict)


@flow_io
@dataclass(frozen=True)
class PointTarget2D:
    frame_id: int | None = None
    label: str | None = None
    x_norm: float | None = None
    y_norm: float | None = None
    confidence: float | None = None


class SyntheticColorCamera(Flow[None, Frame2D]):
    """Deterministic scene with one red and one blue object moving over time."""

    def __init__(self, *, width: int = 96, height: int = 72, dt: float = 0.1) -> None:
        super().__init__()
        self.width = int(width)
        self.height = int(height)
        self.dt = float(dt)

    def init_config(self) -> dict:
        return {"width": self.width, "height": self.height, "dt": self.dt}

    def init(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.frame_id = 0
        self.t_sim = 0.0

    def step(self, _):  # type: ignore[override]
        self.frame_id += 1
        self.t_sim += self.dt
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        image[..., 1] = 18

        red_x = int((0.15 + 0.55 * abs(math.sin(self.frame_id * 0.23))) * (self.width - 12))
        red_y = int((0.35 + 0.15 * math.cos(self.frame_id * 0.17)) * (self.height - 12))
        blue_x = int((0.20 + 0.45 * abs(math.cos(self.frame_id * 0.19))) * (self.width - 10))
        blue_y = int((0.55 + 0.18 * math.sin(self.frame_id * 0.27)) * (self.height - 10))

        image[red_y : red_y + 12, red_x : red_x + 12, 0] = 255
        image[red_y : red_y + 12, red_x : red_x + 12, 1] = 45
        image[red_y : red_y + 12, red_x : red_x + 12, 2] = 45

        image[blue_y : blue_y + 10, blue_x : blue_x + 10, 0] = 45
        image[blue_y : blue_y + 10, blue_x : blue_x + 10, 1] = 45
        image[blue_y : blue_y + 10, blue_x : blue_x + 10, 2] = 255

        return Frame2D(image=image, frame_id=self.frame_id, t_sim=self.t_sim)


def _mask_stats(mask: np.ndarray) -> tuple[int, float, float, BBox2D] | None:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return None
    y_coords = coords[:, 0]
    x_coords = coords[:, 1]
    x0, x1 = int(x_coords.min()), int(x_coords.max())
    y0, y1 = int(y_coords.min()), int(y_coords.max())
    return (
        int(len(coords)),
        float(x_coords.mean()),
        float(y_coords.mean()),
        BBox2D(x=float(x0), y=float(y0), width=float(x1 - x0 + 1), height=float(y1 - y0 + 1)),
    )


class ColorDetector(Flow[Frame2D, DetectionBatch]):
    MIN_PIXELS = 20

    def step(self, frame: Frame2D) -> DetectionBatch:
        if frame.image is None or frame.frame_id is None:
            return DetectionBatch()

        image = frame.image
        red_mask = (image[..., 0] > 180) & (image[..., 1] < 100) & (image[..., 2] < 100)
        blue_mask = (image[..., 2] > 180) & (image[..., 0] < 100) & (image[..., 1] < 100)

        detections: list[Detection2D] = []
        for label, mask in (("red", red_mask), ("blue", blue_mask)):
            stats = _mask_stats(mask)
            if stats is None:
                continue
            pixel_count, centroid_x, centroid_y, bbox = stats
            if pixel_count < self.MIN_PIXELS:
                continue
            detections.append(
                Detection2D(
                    label=label,
                    confidence=min(0.99, pixel_count / 180.0),
                    bbox=bbox,
                    centroid_x=centroid_x,
                    centroid_y=centroid_y,
                    pixel_count=pixel_count,
                )
            )
        return DetectionBatch(frame_id=frame.frame_id, detections=tuple(detections))


class ColorSegmenter(Flow[Frame2D, SegmentationView]):
    def step(self, frame: Frame2D) -> SegmentationView:
        if frame.image is None or frame.frame_id is None:
            return SegmentationView()

        image = frame.image
        masks = {
            "red": (image[..., 0] > 180) & (image[..., 1] < 100) & (image[..., 2] < 100),
            "blue": (image[..., 2] > 180) & (image[..., 0] < 100) & (image[..., 1] < 100),
        }
        pixel_counts: dict[str, int] = {}
        centroids: dict[str, tuple[float, float]] = {}
        labels: list[str] = []
        for label, mask in masks.items():
            stats = _mask_stats(mask)
            if stats is None:
                continue
            pixel_count, centroid_x, centroid_y, _bbox = stats
            if pixel_count == 0:
                continue
            labels.append(label)
            pixel_counts[label] = pixel_count
            centroids[label] = (centroid_x, centroid_y)
        return SegmentationView(
            frame_id=frame.frame_id,
            labels=tuple(labels),
            pixel_counts=pixel_counts,
            centroids=centroids,
        )


class PointToLabel(Flow[DetectionBatch, PointTarget2D]):
    def __init__(self, *, target_label: str = "red", image_width: int = 96, image_height: int = 72) -> None:
        super().__init__()
        self.target_label = str(target_label)
        self.image_width = float(image_width)
        self.image_height = float(image_height)

    def init_config(self) -> dict:
        return {
            "target_label": self.target_label,
            "image_width": int(self.image_width),
            "image_height": int(self.image_height),
        }

    def step(self, batch: DetectionBatch) -> PointTarget2D:
        if batch.frame_id is None:
            return PointTarget2D()
        for det in batch.detections:
            if det.label != self.target_label:
                continue
            return PointTarget2D(
                frame_id=batch.frame_id,
                label=det.label,
                x_norm=det.centroid_x / max(self.image_width - 1.0, 1.0),
                y_norm=det.centroid_y / max(self.image_height - 1.0, 1.0),
                confidence=det.confidence,
            )
        return PointTarget2D(frame_id=batch.frame_id)


class DetectionPrinter(Flow[DetectionBatch, None]):
    def step(self, batch: DetectionBatch) -> None:
        if batch.frame_id is None:
            return None
        if not batch.detections:
            print(f"[frame={batch.frame_id:02d}] detections=[]")
            return None
        summary = [f"{det.label}@({det.centroid_x:4.1f},{det.centroid_y:4.1f}) c={det.confidence:.2f}" for det in batch.detections]
        print(f"[frame={batch.frame_id:02d}] detections={summary}")
        return None


class SegmentationPrinter(Flow[SegmentationView, None]):
    def step(self, seg: SegmentationView) -> None:
        if seg.frame_id is None:
            return None
        print(
            f"[frame={seg.frame_id:02d}] labels={list(seg.labels)} pixel_counts={seg.pixel_counts} centroids={seg.centroids}"
        )
        return None


class PointPrinter(Flow[PointTarget2D, None]):
    def step(self, point: PointTarget2D) -> None:
        if point.frame_id is None:
            return None
        if point.label is None or point.x_norm is None or point.y_norm is None:
            print(f"[frame={point.frame_id:02d}] target-missing")
            return None
        print(
            f"[frame={point.frame_id:02d}] point_to={point.label} x_norm={point.x_norm:.2f} y_norm={point.y_norm:.2f} conf={float(point.confidence or 0.0):.2f}"
        )
        return None
