from __future__ import annotations

"""Small PyBullet viewer/bootstrap helpers shared by advanced demos.

These intentionally keep only the narrow reusable pieces mined from older
environment code: connection setup, viewer cleanup, render-disable during
world creation, debug camera control, and light debug labels.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class DebugCameraPose:
    distance: float
    yaw: float
    pitch: float
    target: tuple[float, float, float]


def connect_pybullet(*, gui: bool, time_step_s: float) -> tuple[object, int]:
    try:
        import pybullet as p
    except ImportError as exc:
        raise RuntimeError(
            "PyBullet is required for simulator-backed demos. "
            "Install `pybullet` in the Python environment used to launch the demo."
        ) from exc

    client_id = p.connect(p.GUI if gui else p.DIRECT)
    if client_id < 0:
        raise RuntimeError("Failed to connect to PyBullet.")

    p.setPhysicsEngineParameter(enableFileCaching=0)
    p.setTimeStep(time_step_s)

    if gui:
        configure_gui_viewer(p)

    return p, client_id


def configure_gui_viewer(p: object) -> None:
    # This is the narrow viewer setup worth reusing from the older Ravens path.
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)
    time.sleep(0.15)


def set_rendering_enabled(p: object, enabled: bool) -> None:
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1 if enabled else 0)


@contextmanager
def rendering_disabled(p: object, *, active: bool) -> Iterator[None]:
    if active:
        set_rendering_enabled(p, False)
    try:
        yield
    finally:
        if active:
            set_rendering_enabled(p, True)


def set_debug_camera(p: object, camera: DebugCameraPose) -> None:
    p.resetDebugVisualizerCamera(
        cameraDistance=camera.distance,
        cameraYaw=camera.yaw,
        cameraPitch=camera.pitch,
        cameraTargetPosition=list(camera.target),
    )


def step_gui_frames(p: object, *, frames: int, sleep_s: float) -> None:
    for _ in range(frames):
        p.stepSimulation()
        if sleep_s > 0.0:
            time.sleep(sleep_s)


def add_debug_label(
    p: object,
    text: str,
    position: tuple[float, float, float],
    *,
    color: tuple[float, float, float] = (1.0, 1.0, 1.0),
    size: float = 1.2,
) -> None:
    p.addUserDebugText(
        text=text,
        textPosition=list(position),
        textColorRGB=list(color),
        textSize=size,
    )
