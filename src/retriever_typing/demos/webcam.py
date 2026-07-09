"""Live webcam color-detection demo, runnable in one line through Retriever Hub.

    pip install "retriever-core[demo]"          # runtime + numpy + opencv
    python -c "from retriever import hub; hub.use('openretriever/golden-retriever:webcam')()"

Hold a red or blue object in front of the camera. The graph is a closed loop:

    CameraSource @ Rate(hz) --Latest--> ColorDetector @ Trigger --Latest--> Display @ Rate

`run()` builds the graph and steps it in-process for a fixed duration. The Flows
are plain `retriever` Flows, so you can also compose your own pipeline from them.
"""
from __future__ import annotations

import numpy as np

from retriever.flow import Flow, Latest, Pipeline, Rate, Trigger, io


@io
class Frame:
    image: object  # HxWx3 uint8 RGB array
    idx: int


@io
class Detections:
    idx: int
    objects: list  # list[tuple[str, float]]


@io
class Report:
    idx: int


class CameraSource(Flow[None, Frame]):
    """Emit RGB frames from a live webcam via OpenCV."""

    def __init__(self, *, camera_index: int = 0, width: int = 640, height: int = 480):
        super().__init__()
        self.camera_index, self.width, self.height = camera_index, width, height
        self.idx, self._cap, self._cv2 = 0, None, None

    def _open(self) -> None:
        try:
            import cv2
        except ImportError:  # pragma: no cover - dependency guard
            raise ImportError(
                "The webcam demo needs OpenCV. Install it with: "
                'pip install "retriever-core[demo]"  (or: pip install opencv-python)'
            ) from None
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open webcam #{self.camera_index}. Check that a camera is "
                "connected and that this process has camera permission."
            )
        self._cap, self._cv2 = cap, cv2

    def step(self, _) -> Frame:
        if self._cap is None:
            self._open()
        ok, bgr = self._cap.read()
        if not ok:
            raise RuntimeError("Webcam frame read failed.")
        self.idx += 1
        return Frame(image=self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB), idx=self.idx)

    def reset(self) -> None:
        if self._cap is not None:
            self._cap.release()
        self._cap, self.idx = None, 0


class ColorDetector(Flow[Frame, Detections]):
    """Report red/blue objects from an RGB frame with simple thresholds."""

    def __init__(self, *, min_fraction: float = 0.02):
        super().__init__()
        self.min_fraction = min_fraction

    def step(self, f: Frame) -> Detections:
        img = np.asarray(f.image).astype("int16")
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        objects: list = []
        red = float(((r > 120) & (r - g > 60) & (r - b > 60)).mean())
        blue = float(((b > 120) & (b - r > 60) & (b - g > 60)).mean())
        if red > self.min_fraction:
            objects.append(("red_object", round(min(red * 3, 0.99), 2)))
        if blue > self.min_fraction:
            objects.append(("blue_object", round(min(blue * 3, 0.99), 2)))
        return Detections(idx=f.idx, objects=objects)


class Display(Flow[Detections, Report]):
    """Print detections to stdout."""

    def step(self, d: Detections) -> Report:
        tag = ", ".join(f"{label} ({conf})" for label, conf in d.objects) or "(nothing)"
        print(f"  frame {d.idx:3d}: {tag}")
        return Report(idx=d.idx)


def build(*, hz: float = 20.0, camera_index: int = 0) -> Pipeline:
    """Build the camera -> detector -> display pipeline (no camera opened yet)."""
    pipe = Pipeline("golden-webcam-demo")
    camera = CameraSource(camera_index=camera_index) @ Rate(hz=hz)
    detector = ColorDetector() @ Trigger("image")
    display = Display() @ Rate(hz=hz)
    pipe.connect(camera, detector, sync=Latest())
    pipe.connect(detector, display, sync=Latest())
    return pipe


def run(*, seconds: float = 5.0, hz: float = 20.0, camera_index: int = 0) -> None:
    """Open the webcam and run the detection graph in-process for `seconds`."""
    pipe = build(hz=hz, camera_index=camera_index)
    print(f"webcam demo: camera @ {hz:g}Hz -> ColorDetector -> Display  (webcam #{camera_index})")
    print("Hold a red or blue object in front of the camera.")
    print(f"Running {seconds:g}s in-process...\n" + "-" * 52)
    dt = 1.0 / hz
    with pipe:
        for _ in range(int(seconds * hz)):
            pipe.step(dt=dt)
    print("-" * 52 + "\nDone.")
