"""
Synthetic perception stepper demo.

Run:
  pixi run demo-synthetic-color-stepper
  pixi run python examples/advanced/perception_debug/synthetic_color_stepper.py --steps 12 --dt 0.1
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np

from retriever.flow import Flow, Pipeline, Rate, Trigger, Latest, io


@io
@dataclass
class SyntheticFrame:
    image: np.ndarray | None = None
    frame_id: int | None = None
    t_sim: float | None = None


@io
@dataclass
class DetectionOut:
    frame_id: int | None = None
    dominant_color: str | None = None
    centroid_x: float | None = None
    centroid_y: float | None = None
    confidence: float | None = None
    red_pixels: int | None = None
    blue_pixels: int | None = None


class SyntheticCamera(Flow[None, SyntheticFrame]):
    def __init__(self, *, width: int = 64, height: int = 48):
        super().__init__()
        self.width = int(width)
        self.height = int(height)

    def init_config(self) -> dict:
        return {"width": self.width, "height": self.height}

    def init(self) -> None:
        self.frame_id = 0
        self.t_sim = 0.0

    def reset(self) -> None:
        self.frame_id = 0
        self.t_sim = 0.0

    def step(self, _) -> SyntheticFrame:
        self.frame_id += 1
        self.t_sim += 0.1

        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        image[..., 1] = 16

        square = 10
        x_center = int((0.2 + 0.6 * abs(math.sin(self.frame_id * 0.25))) * (self.width - 1))
        y_center = self.height // 2
        x0 = max(0, x_center - square // 2)
        x1 = min(self.width, x0 + square)
        y0 = max(0, y_center - square // 2)
        y1 = min(self.height, y0 + square)

        if (self.frame_id // 5) % 2 == 0:
            image[y0:y1, x0:x1, 0] = 255
            image[y0:y1, x0:x1, 1] = 40
            image[y0:y1, x0:x1, 2] = 40
        else:
            image[y0:y1, x0:x1, 0] = 40
            image[y0:y1, x0:x1, 1] = 40
            image[y0:y1, x0:x1, 2] = 255

        image[4:10, 4:10, 0] = 40
        image[4:10, 4:10, 1] = 40
        image[4:10, 4:10, 2] = 255

        return SyntheticFrame(image=image, frame_id=self.frame_id, t_sim=self.t_sim)


class ColorDetector(Flow[SyntheticFrame, DetectionOut]):
    def step(self, frame: SyntheticFrame) -> DetectionOut:
        if frame.image is None or frame.frame_id is None:
            return DetectionOut()

        image = frame.image
        red_mask = (image[..., 0] > 180) & (image[..., 1] < 100) & (image[..., 2] < 100)
        blue_mask = (image[..., 2] > 180) & (image[..., 0] < 100) & (image[..., 1] < 100)

        red_pixels = int(red_mask.sum())
        blue_pixels = int(blue_mask.sum())
        if red_pixels == 0 and blue_pixels == 0:
            return DetectionOut(frame_id=frame.frame_id)

        if red_pixels >= blue_pixels:
            dominant = 'red'
            coords = np.argwhere(red_mask)
            confidence = red_pixels / max(red_pixels + blue_pixels, 1)
        else:
            dominant = 'blue'
            coords = np.argwhere(blue_mask)
            confidence = blue_pixels / max(red_pixels + blue_pixels, 1)

        centroid_y = float(coords[:, 0].mean()) if len(coords) else None
        centroid_x = float(coords[:, 1].mean()) if len(coords) else None
        return DetectionOut(
            frame_id=frame.frame_id,
            dominant_color=dominant,
            centroid_x=centroid_x,
            centroid_y=centroid_y,
            confidence=float(confidence),
            red_pixels=red_pixels,
            blue_pixels=blue_pixels,
        )


class DetectionPrinter(Flow[DetectionOut, None]):
    def step(self, det: DetectionOut) -> None:
        if det.frame_id is None or det.dominant_color is None:
            return None
        print(
            f"[frame={det.frame_id:02d}] color={det.dominant_color:<4} "
            f"centroid=({det.centroid_x:5.1f}, {det.centroid_y:4.1f}) "
            f"conf={float(det.confidence or 0.0):.2f} "
            f"red={det.red_pixels} blue={det.blue_pixels}"
        )
        return None


def build_pipeline(*, dt: float, width: int, height: int) -> Pipeline:
    hz = 1.0 / max(dt, 1e-6)
    pipe = Pipeline('synthetic_color_stepper')
    with pipe:
        camera = SyntheticCamera(width=width, height=height) @ Rate(hz=hz)
        detector = ColorDetector() @ Trigger('image')
        printer = DetectionPrinter() @ Trigger('dominant_color')
        pipe.connect(camera, detector, sync=Latest())
        pipe.connect(detector, printer, sync=Latest())
    return pipe


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Synthetic perception stepper demo.')
    p.add_argument('--steps', type=int, default=12)
    p.add_argument('--dt', type=float, default=0.1)
    p.add_argument('--width', type=int, default=64)
    p.add_argument('--height', type=int, default=48)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pipe = build_pipeline(dt=args.dt, width=args.width, height=args.height)
    try:
        for _ in range(args.steps):
            pipe.step(dt=args.dt)
    finally:
        pipe.close_stepper()


if __name__ == '__main__':
    main()
