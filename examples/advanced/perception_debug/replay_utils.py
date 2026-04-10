"""Helpers for replaying synthetic perception streams from MCAP."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from examples.advanced.perception_debug.synthetic_color_stepper import SyntheticFrame


def as_synthetic_frame(obj: object) -> SyntheticFrame | None:
    if isinstance(obj, SyntheticFrame):
        if obj.image is None:
            return None
        return SyntheticFrame(
            image=np.asarray(obj.image, dtype=np.uint8),
            frame_id=None if obj.frame_id is None else int(obj.frame_id),
            t_sim=None if obj.t_sim is None else float(obj.t_sim),
        )

    if not isinstance(obj, dict):
        return None

    image = obj.get('image')
    if image is None:
        return None
    frame_id = obj.get('frame_id')
    t_sim = obj.get('t_sim')

    try:
        image_np = np.asarray(image, dtype=np.uint8)
    except Exception:
        return None

    return SyntheticFrame(
        image=image_np,
        frame_id=None if frame_id is None else int(frame_id),
        t_sim=None if t_sim is None else float(t_sim),
    )


def load_synthetic_frame_buffer_from_mcap(path: Path) -> list[tuple[float, SyntheticFrame]]:
    from retriever.lib.mcap import MCAPReader

    with MCAPReader(path) as reader:
        steps = list(reader)

    if not steps:
        raise RuntimeError(f'MCAP recording is empty: {path}')

    camera_key: str | None = None
    for step in steps:
        outputs = step.get('outputs', {}) or {}
        if not isinstance(outputs, dict):
            continue
        for key, val in outputs.items():
            if as_synthetic_frame(val) is not None:
                camera_key = str(key)
                break
        if camera_key is not None:
            break

    if camera_key is None:
        raise RuntimeError(f'Could not locate a SyntheticFrame-like stream in MCAP: {path}')

    buffer: list[tuple[float, SyntheticFrame]] = []
    for step in steps:
        now = step.get('now', 0.0) or 0.0
        try:
            ts = float(now)
        except Exception:
            ts = 0.0

        outputs = step.get('outputs', {}) or {}
        if not isinstance(outputs, dict):
            continue

        frame = as_synthetic_frame(outputs.get(camera_key))
        if frame is None:
            continue
        buffer.append((ts, frame))

    if not buffer:
        raise RuntimeError(f'SyntheticFrame stream extracted but contained no frames: {path}')

    print(f'[Replay] extracted source stream key={camera_key} frames={len(buffer)} from {path}')
    return buffer
