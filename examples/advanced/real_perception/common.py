"""Shared helpers for real perception examples.

These examples intentionally reuse the small payload vocabulary from
`perception_examples/common.py` so the real-model path does not invent a second
set of example-only types.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from retriever.flow import Flow

from examples.advanced.perception_examples.common import (
    BBox2D,
    Detection2D,
    DetectionBatch,
    Frame2D,
    PointTarget2D,
    SegmentationView,
    SyntheticColorCamera,
)


DEFAULT_DETECTION_LABELS = ("red block", "blue block")
DEFAULT_SEGMENT_LABELS = ("red block", "blue block")
DEFAULT_POINT_QUERY = "point to the red block"


@dataclass(frozen=True)
class ExampleImage:
    image: np.ndarray
    description: str


def parse_labels(raw: str | Sequence[str] | None, *, fallback: tuple[str, ...]) -> tuple[str, ...]:
    if raw is None:
        return fallback
    if isinstance(raw, str):
        items = tuple(part.strip() for part in raw.split(",") if part.strip())
    else:
        items = tuple(str(part).strip() for part in raw if str(part).strip())
    return items or fallback


def render_default_example_image(*, width: int = 96, height: int = 72, frame_index: int = 4) -> ExampleImage:
    camera = SyntheticColorCamera(width=width, height=height, dt=0.1)
    camera.reset()
    frame = Frame2D()
    for _ in range(max(1, frame_index)):
        frame = camera.step(None)
    assert frame.image is not None
    return ExampleImage(image=frame.image.copy(), description="synthetic red/blue tabletop scene")


def load_example_image(image_path: str | None) -> ExampleImage:
    if image_path is None:
        return render_default_example_image()
    path = Path(image_path)
    image = Image.open(path).convert("RGB")
    return ExampleImage(image=np.array(image), description=str(path))


class StaticImageSource(Flow[None, Frame2D]):
    def __init__(self, *, image_path: str | None = None, dt: float = 0.1) -> None:
        super().__init__()
        self.image_path = image_path
        self.dt = float(dt)

    def init_config(self) -> dict:
        return {"image_path": self.image_path, "dt": self.dt}

    def init(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._frame_id = 0
        self._t_sim = 0.0
        self._example = load_example_image(self.image_path)

    def step(self, _):  # type: ignore[override]
        self._frame_id += 1
        self._t_sim += self.dt
        return Frame2D(image=self._example.image.copy(), frame_id=self._frame_id, t_sim=self._t_sim)


class GeminiVisionBackend:
    def __init__(self, *, model: str = "gemini-2.0-flash", api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self._client = None
        self._types = None

    def _ensure_client(self):
        if self._client is not None:
            return self._client, self._types
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is required for backend=gemini_api")
        try:
            from google import genai
            from google.genai import types as genai_types
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError(
                "google-genai is required for backend=gemini_api. Install the dedicated perception env first."
            ) from exc
        self._client = genai.Client(api_key=self.api_key)
        self._types = genai_types
        return self._client, self._types

    @staticmethod
    def _extract_json_list(text: str) -> list[dict]:
        start = text.find("[")
        end = text.rfind("]") + 1
        if start == -1 or end <= start:
            return []
        try:
            payload = json.loads(text[start:end])
        except json.JSONDecodeError:
            return []
        return payload if isinstance(payload, list) else []

    def _generate_text(self, *, prompt: str, image_rgb: np.ndarray) -> str:
        client, genai_types = self._ensure_client()
        pil_image = Image.fromarray(image_rgb.astype(np.uint8)).convert("RGB")
        response = client.models.generate_content(
            model=self.model,
            contents=[prompt, pil_image],
            config=genai_types.GenerateContentConfig(temperature=0.0),
        )
        text = getattr(response, "text", None)
        if not text:
            raise RuntimeError("Gemini response did not contain text output")
        return text

    def detect(self, image_rgb: np.ndarray, *, labels: tuple[str, ...], frame_id: int | None) -> DetectionBatch:
        label_list = ", ".join(labels)
        prompt = (
            "Detect the requested objects in the image. "
            f"Only use these labels: {label_list}. "
            "Return only JSON as a list of objects with fields "
            "label and box_2d. box_2d must be [y_min, x_min, y_max, x_max] normalized to 0-1000."
        )
        items = self._extract_json_list(self._generate_text(prompt=prompt, image_rgb=image_rgb))
        height, width = image_rgb.shape[:2]
        detections: list[Detection2D] = []
        for item in items:
            label = str(item.get("label", "")).strip()
            box = item.get("box_2d")
            if label == "" or not isinstance(box, list) or len(box) != 4:
                continue
            try:
                y1, x1, y2, x2 = [float(value) for value in box]
            except (TypeError, ValueError):
                continue
            x0 = max(0.0, min(width - 1.0, x1 * width / 1000.0))
            y0 = max(0.0, min(height - 1.0, y1 * height / 1000.0))
            x3 = max(x0, min(width - 1.0, x2 * width / 1000.0))
            y3 = max(y0, min(height - 1.0, y2 * height / 1000.0))
            detections.append(
                Detection2D(
                    label=label,
                    confidence=0.8,
                    bbox=BBox2D(x=x0, y=y0, width=max(1.0, x3 - x0), height=max(1.0, y3 - y0)),
                    centroid_x=(x0 + x3) / 2.0,
                    centroid_y=(y0 + y3) / 2.0,
                    pixel_count=int(max(1.0, (x3 - x0) * (y3 - y0))),
                )
            )
        return DetectionBatch(frame_id=frame_id, detections=tuple(detections))

    def point(self, image_rgb: np.ndarray, *, query: str, frame_id: int | None) -> PointTarget2D:
        prompt = (
            "Point to the object described by the user. "
            f"Description: {query}. "
            "Return only JSON as a list with one object containing label and point. "
            "point must be [y, x] normalized to 0-1000."
        )
        items = self._extract_json_list(self._generate_text(prompt=prompt, image_rgb=image_rgb))
        if not items:
            return PointTarget2D(frame_id=frame_id)
        item = items[0]
        point = item.get("point")
        if not isinstance(point, list) or len(point) != 2:
            return PointTarget2D(frame_id=frame_id)
        try:
            y, x = [float(value) for value in point]
        except (TypeError, ValueError):
            return PointTarget2D(frame_id=frame_id)
        return PointTarget2D(
            frame_id=frame_id,
            label=str(item.get("label", query)),
            x_norm=max(0.0, min(1.0, x / 1000.0)),
            y_norm=max(0.0, min(1.0, y / 1000.0)),
            confidence=0.8,
        )


class GeminiDetector(Flow[Frame2D, DetectionBatch]):
    def __init__(self, *, labels: tuple[str, ...], model: str = "gemini-2.0-flash") -> None:
        super().__init__()
        self.labels = tuple(labels)
        self.model = model
        self._backend = GeminiVisionBackend(model=model)

    def init_config(self) -> dict:
        return {"labels": list(self.labels), "model": self.model}

    def step(self, frame: Frame2D) -> DetectionBatch:
        if frame.image is None:
            return DetectionBatch()
        return self._backend.detect(frame.image, labels=self.labels, frame_id=frame.frame_id)


class GeminiPointer(Flow[Frame2D, PointTarget2D]):
    def __init__(self, *, query: str, model: str = "gemini-2.0-flash") -> None:
        super().__init__()
        self.query = query
        self.model = model
        self._backend = GeminiVisionBackend(model=model)

    def init_config(self) -> dict:
        return {"query": self.query, "model": self.model}

    def step(self, frame: Frame2D) -> PointTarget2D:
        if frame.image is None:
            return PointTarget2D()
        return self._backend.point(frame.image, query=self.query, frame_id=frame.frame_id)


class OwlSamSegmenter(Flow[Frame2D, SegmentationView]):
    def __init__(self, *, labels: tuple[str, ...], score_threshold: float = 0.1) -> None:
        super().__init__()
        self.labels = tuple(labels)
        self.score_threshold = float(score_threshold)
        self._device = None
        self._torch = None
        self._owl_processor = None
        self._owl_model = None
        self._sam_processor = None
        self._sam_model = None

    def init_config(self) -> dict:
        return {"labels": list(self.labels), "score_threshold": self.score_threshold}

    def init(self) -> None:
        try:
            import cv2  # noqa: F401
            import torch
            from transformers import Owlv2ForObjectDetection, Owlv2Processor, SamModel, SamProcessor
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError(
                "transformers and torch are required for backend=owl_sam_local. Install the dedicated perception env first."
            ) from exc
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        self._device = device
        self._torch = torch
        self._owl_processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        self._owl_model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(device)
        self._sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        self._sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)

    def step(self, frame: Frame2D) -> SegmentationView:
        if frame.image is None or frame.frame_id is None:
            return SegmentationView()
        if self._torch is None:
            self.init()
        import cv2

        image_rgb = frame.image.astype(np.uint8)
        pil_image = Image.fromarray(image_rgb)
        torch = self._torch
        assert torch is not None
        assert self._owl_processor is not None and self._owl_model is not None
        assert self._sam_processor is not None and self._sam_model is not None

        inputs = self._owl_processor(text=[list(self.labels)], images=pil_image, return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self._owl_model(**inputs)
        target_sizes = torch.tensor([pil_image.size[::-1]], device=self._device)
        results = self._owl_processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=self.score_threshold,
        )[0]
        boxes = results["boxes"]
        label_indexes = results["labels"]
        if len(boxes) == 0:
            return SegmentationView(frame_id=frame.frame_id)

        input_points = []
        labels: list[str] = []
        for box, label_idx in zip(boxes, label_indexes):
            x1, y1, x2, y2 = box.tolist()
            input_points.append([[(x1 + x2) / 2.0, (y1 + y2) / 2.0]])
            labels.append(self.labels[int(label_idx)])
        point_tensor = torch.tensor(input_points, dtype=torch.float32).unsqueeze(0)
        sam_inputs = self._sam_processor(pil_image, input_points=point_tensor, return_tensors="pt")
        for key, value in sam_inputs.items():
            if isinstance(value, torch.Tensor):
                sam_inputs[key] = value.to(self._device, dtype=torch.float32 if value.dtype == torch.float64 else value.dtype)
        with torch.no_grad():
            sam_outputs = self._sam_model(**sam_inputs)
        masks = self._sam_processor.image_processor.post_process_masks(
            sam_outputs.pred_masks.cpu(),
            sam_inputs["original_sizes"].cpu(),
            sam_inputs["reshaped_input_sizes"].cpu(),
        )
        mask_array = masks[0][:, 0, :, :].numpy() if masks else np.zeros((0,) + image_rgb.shape[:2], dtype=bool)

        pixel_counts: dict[str, int] = {}
        centroids: dict[str, tuple[float, float]] = {}
        ordered_labels: list[str] = []
        for label, mask in zip(labels, mask_array):
            coords = np.argwhere(mask > 0)
            if len(coords) == 0:
                continue
            ys = coords[:, 0]
            xs = coords[:, 1]
            pixel_counts[label] = int(len(coords))
            centroids[label] = (float(xs.mean()), float(ys.mean()))
            ordered_labels.append(label)
        return SegmentationView(
            frame_id=frame.frame_id,
            labels=tuple(ordered_labels),
            pixel_counts=pixel_counts,
            centroids=centroids,
        )
