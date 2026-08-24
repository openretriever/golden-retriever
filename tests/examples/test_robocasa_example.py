from argparse import Namespace
from math import ceil

import numpy as np
import pytest
from examples.advanced.robocasa import app
from examples.advanced.robocasa.app import (
    DemoActionSource,
    RoboCasaAction,
    RoboCasaSimulator,
    build_pipeline,
)
from examples.advanced.robocasa.mjviser_bridge import ReplayControls


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


def test_demo_action_source_obeys_browser_replay_controls() -> None:
    controls = ReplayControls(task="TurnOnMicrowave", episode=0)
    source = DemoActionSource(mock_steps=4, controls=controls)
    source.reset()

    first = source.step()
    controls.set_paused(True)
    paused = source.step()
    controls.request_step()
    second = source.step()
    paused_again = source.step()
    controls.request_restart()
    restarted = source.step()

    assert first.active is True
    assert first.episode_step == 0
    assert paused.active is False
    assert second.active is True
    assert second.episode_step == 1
    assert paused_again.active is False
    assert restarted.active is True
    assert restarted.episode_step == 0
    assert restarted.cycle == 1
    assert controls.snapshot().total_steps == 4


def test_real_simulator_restart_restores_complete_episode_state(monkeypatch) -> None:
    class FakeEnvironment:
        def step(self, _action):
            return None, 0.0, False, {}

        def _check_success(self) -> bool:
            return False

    environment = FakeEnvironment()
    initial_state = {
        "states": object(),
        "model": "<mujoco />",
        "ep_meta": '{"task": "PrepareCoffee"}',
    }
    reset_calls = []
    monkeypatch.setattr(
        app,
        "_reset_to_episode",
        lambda env, state: reset_calls.append((env, state)),
    )
    simulator = RoboCasaSimulator(mode="robocasa", hz=1_000.0)
    simulator.env = environment
    simulator._initial_state = initial_state
    simulator._action_count = 2
    simulator._active_cycle = 0

    observation = simulator._step_robocasa(
        RoboCasaAction(
            values=np.zeros(1),
            episode_step=0,
            cycle=1,
            active=True,
        )
    )

    assert reset_calls == [(environment, initial_state)]
    assert simulator._active_cycle == 1
    assert observation.cycle == 1
    assert observation.episode_step == 0


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


def test_run_builds_concise_offline_console_entrypoint(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(app, "_execute", calls.append)

    app.run(task="CoffeeSetupMug", episode=2, open_browser=False)

    args = calls[0]
    assert args.mode == "robocasa"
    assert args.task == "CoffeeSetupMug"
    assert args.episode == 2
    assert args.planner == "offline"
    assert args.visualize == "mjviser"
    assert args.open_browser is False
