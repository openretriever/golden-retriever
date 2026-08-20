from argparse import Namespace
from math import ceil

import pytest

from examples.advanced.robocasa.app import build_pipeline


def _mock_args(**overrides: object) -> Namespace:
    values = {
        "mode": "mock",
        "task": "TurnOnMicrowave",
        "split": "pretrain",
        "episode": 0,
        "repeat": False,
        "mock_steps": 6,
        "hz": 20.0,
        "visualize": "none",
        "viewer": False,
        "camera": "robot0_agentview_center",
        "width": 320,
        "height": 180,
        "image_hz": 5.0,
        "print_every": 100,
    }
    values.update(overrides)
    return Namespace(**values)


def test_mock_robocasa_replay_reaches_success_without_simulator() -> None:
    args = _mock_args()
    pipeline, simulator = build_pipeline(args)
    try:
        for _ in range(7):
            pipeline.step(dt=1.0 / args.hz)
        latest = simulator.latest
    finally:
        pipeline.close_stepper()

    assert latest is not None
    assert latest.source == "mock"
    assert latest.progress == pytest.approx(1.0)
    assert latest.success is True
    assert simulator.env is None


def test_mock_robocasa_video_is_finalized(tmp_path) -> None:
    import cv2

    video_path = tmp_path / "robocasa-mock.mp4"
    args = _mock_args(video=str(video_path), video_fps=5.0)
    pipeline, _simulator = build_pipeline(args)
    try:
        for _ in range(7):
            pipeline.step(dt=1.0 / args.hz)
    finally:
        pipeline.close_stepper()

    assert video_path.exists()
    assert video_path.stat().st_size > 0
    capture = cv2.VideoCapture(str(video_path))
    try:
        assert capture.isOpened()
        expected_frames = ceil(args.mock_steps / round(args.hz / args.image_hz))
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == expected_frames
    finally:
        capture.release()
