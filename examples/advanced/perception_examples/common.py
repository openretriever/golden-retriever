"""Shared synthetic perception flows for advanced examples.

The reusable payloads come from `retriever.types.perception`. This module keeps
only the deterministic scene logic and example-local printers/helpers.
"""

from __future__ import annotations

import math

import numpy as np

from retriever.flow import Flow
from retriever.types.perception import (
    BBox2D,
    Detection2D,
    DetectionBatch,
    Image2D,
    PointTarget2D,
    SegmentationMask2D,
)
from retriever.types.spatial import Header


def _make_header(*, frame_index: int, t_sim: float, source: str) -> Header:
    stamp_ns = max(1, int(round(t_sim * 1_000_000_000)))
    return Header(stamp_ns=stamp_ns, frame_id="synthetic_camera", source=source)


class SyntheticColorCamera(Flow[None, Image2D]):
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
        self.frame_index = 0
        self.t_sim = 0.0

    def step(self, _):  # type: ignore[override]
        self.frame_index += 1
        self.t_sim += self.dt
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        image[..., 1] = 18

        red_x = int((0.15 + 0.55 * abs(math.sin(self.frame_index * 0.23))) * (self.width - 12))
        red_y = int((0.35 + 0.15 * math.cos(self.frame_index * 0.17)) * (self.height - 12))
        blue_x = int((0.20 + 0.45 * abs(math.cos(self.frame_index * 0.19))) * (self.width - 10))
        blue_y = int((0.55 + 0.18 * math.sin(self.frame_index * 0.27)) * (self.height - 10))

        image[red_y : red_y + 12, red_x : red_x + 12, 0] = 255
        image[red_y : red_y + 12, red_x : red_x + 12, 1] = 45
        image[red_y : red_y + 12, red_x : red_x + 12, 2] = 45

        image[blue_y : blue_y + 10, blue_x : blue_x + 10, 0] = 45
        image[blue_y : blue_y + 10, blue_x : blue_x + 10, 1] = 45
        image[blue_y : blue_y + 10, blue_x : blue_x + 10, 2] = 255

        return Image2D(
            data=image,
            encoding="rgb8",
            header=_make_header(frame_index=self.frame_index, t_sim=self.t_sim, source="golden.synthetic_color_camera"),
            frame_index=self.frame_index,
        )


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


def summarize_segmentation(mask_msg: SegmentationMask2D) -> tuple[list[str], dict[str, int], dict[str, tuple[float, float]]]:
    labels: list[str] = []
    pixel_counts: dict[str, int] = {}
    centroids: dict[str, tuple[float, float]] = {}
    for value, label in sorted(mask_msg.label_map.items()):
        if value == 0:
            continue
        coords = np.argwhere(mask_msg.mask == value)
        if len(coords) == 0:
            continue
        ys = coords[:, 0]
        xs = coords[:, 1]
        labels.append(label)
        pixel_counts[label] = int(len(coords))
        centroids[label] = (float(xs.mean()), float(ys.mean()))
    return labels, pixel_counts, centroids


class ColorDetector(Flow[Image2D, DetectionBatch]):
    MIN_PIXELS = 20

    def step(self, frame: Image2D) -> DetectionBatch:
        image = frame.data
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
        return DetectionBatch(detections=tuple(detections), header=frame.header, frame_index=frame.frame_index)


class ColorSegmenter(Flow[Image2D, SegmentationMask2D]):
    def step(self, frame: Image2D) -> SegmentationMask2D:
        image = frame.data
        red_mask = (image[..., 0] > 180) & (image[..., 1] < 100) & (image[..., 2] < 100)
        blue_mask = (image[..., 2] > 180) & (image[..., 0] < 100) & (image[..., 1] < 100)

        mask = np.zeros(image.shape[:2], dtype=np.int32)
        mask[red_mask] = 1
        mask[blue_mask] = 2
        return SegmentationMask2D(
            mask=mask,
            header=frame.header,
            frame_index=frame.frame_index,
            label_map={0: "background", 1: "red", 2: "blue"},
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
        if batch.frame_index is None:
            return PointTarget2D()
        for det in batch.detections:
            if det.label != self.target_label:
                continue
            return PointTarget2D(
                frame_index=batch.frame_index,
                header=batch.header,
                label=det.label,
                x_norm=det.centroid_x / max(self.image_width - 1.0, 1.0) if det.centroid_x is not None else None,
                y_norm=det.centroid_y / max(self.image_height - 1.0, 1.0) if det.centroid_y is not None else None,
                confidence=det.confidence,
            )
        return PointTarget2D(frame_index=batch.frame_index, header=batch.header)


class DetectionPrinter(Flow[DetectionBatch, None]):
    def step(self, batch: DetectionBatch) -> None:
        if batch.frame_index is None:
            return None
        if not batch.detections:
            print(f"[frame={batch.frame_index:02d}] detections=[]")
            return None
        summary = [
            f"{det.label}@({det.centroid_x:4.1f},{det.centroid_y:4.1f}) c={det.confidence:.2f}"
            for det in batch.detections
            if det.centroid_x is not None and det.centroid_y is not None and det.confidence is not None
        ]
        print(f"[frame={batch.frame_index:02d}] detections={summary}")
        return None


class SegmentationPrinter(Flow[SegmentationMask2D, None]):
    def step(self, seg: SegmentationMask2D) -> None:
        if seg.frame_index is None:
            return None
        labels, pixel_counts, centroids = summarize_segmentation(seg)
        print(f"[frame={seg.frame_index:02d}] labels={labels} pixel_counts={pixel_counts} centroids={centroids}")
        return None


class PointPrinter(Flow[PointTarget2D, None]):
    def step(self, point: PointTarget2D) -> None:
        if point.frame_index is None:
            return None
        print(
            f"[frame={point.frame_index:02d}] target={point.label} x_norm={point.x_norm:.2f} y_norm={point.y_norm:.2f} confidence={point.confidence:.2f}"
        )
        return None
