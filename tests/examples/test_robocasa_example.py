import sys
from argparse import Namespace
from math import ceil
from types import ModuleType

import numpy as np
import pytest

from examples.advanced.robocasa import app
from examples.advanced.robocasa.app import (
    DemoActionSource,
    RoboCasaAction,
    RoboCasaObservation,
    RoboCasaSimulator,
    TaskVerifier,
    build_pipeline,
)
from examples.advanced.robocasa.embodied import (
    EmbodiedGoal,
    ExecutionState,
    OfflineEmbodiedPlanner,
)
from examples.advanced.robocasa.runtime import ReplayControls


def test_missing_dataset_error_uses_locked_environment(monkeypatch, tmp_path) -> None:
    registry = ModuleType("robocasa.utils.dataset_registry_utils")
    registry.get_ds_meta = lambda **_kwargs: {"path": str(tmp_path / "missing.hdf5")}
    monkeypatch.setitem(sys.modules, registry.__name__, registry)

    with pytest.raises(FileNotFoundError, match=r"pixi run --locked -e robocasa"):
        app._dataset_path("PrepareCoffee", "pretrain")


def _mock_args(**overrides: object) -> Namespace:
    values = {
        "mode": "mock",
        "task": "TurnOnMicrowave",
        "split": "pretrain",
        "episode": 0,
        "execution_mode": "demonstration",
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


def test_demo_action_source_rejects_a_plan_for_another_dataset() -> None:
    source = DemoActionSource(task="TurnOnMicrowave", mock_steps=2)
    source.reset()
    plan = OfflineEmbodiedPlanner().plan(
        EmbodiedGoal(text="Make coffee", task="PrepareCoffee")
    )

    with pytest.raises(ValueError, match="does not match the loaded demonstration"):
        source.step(ExecutionState(plan=plan))


def test_demo_action_source_accepts_its_dispatched_plan() -> None:
    source = DemoActionSource(task="TurnOnMicrowave", mock_steps=2)
    source.reset()
    plan = OfflineEmbodiedPlanner().plan(
        EmbodiedGoal(text="Turn on the microwave", task="TurnOnMicrowave")
    )

    action = source.step(ExecutionState(plan=plan))

    assert action.active is True
    assert action.episode_step == 0


def test_demo_action_source_marks_each_cycle_complete_once(monkeypatch) -> None:
    controls = ReplayControls(task="TurnOnMicrowave", episode=0)
    source = DemoActionSource(mock_steps=2, controls=controls)
    source.reset()
    completions = 0
    original_mark_complete = controls.mark_complete

    def count_completion() -> None:
        nonlocal completions
        completions += 1
        original_mark_complete()

    monkeypatch.setattr(controls, "mark_complete", count_completion)

    assert source.step().episode_step == 0
    assert source.step().episode_step == 1
    first_terminal = source.step()
    repeated_terminal = source.step()

    assert first_terminal.active is False
    assert repeated_terminal.active is False
    assert first_terminal.episode_step == repeated_terminal.episode_step == 1
    assert first_terminal.cycle == repeated_terminal.cycle == 0
    assert completions == 1

    controls.request_restart()
    assert source.step().cycle == 1
    source.step()
    source.step()

    assert completions == 2


def test_repeating_source_advances_controls_and_verifier_together() -> None:
    controls = ReplayControls(task="TurnOnMicrowave", episode=0)
    source = DemoActionSource(mock_steps=2, repeat=True, controls=controls)
    simulator = RoboCasaSimulator(mode="mock", mock_steps=2)
    verifier = TaskVerifier(controls=controls)
    source.reset()
    simulator.reset()

    for _ in range(3):
        verifier.step(simulator.step(source.step()))

    snapshot = controls.snapshot()
    assert snapshot.cycle == 1
    assert snapshot.episode_step == 0
    assert snapshot.status == "Running"
    assert snapshot.success is False


def test_paused_source_still_marks_an_exhausted_replay_complete() -> None:
    controls = ReplayControls(task="TurnOnMicrowave", episode=0)
    source = DemoActionSource(mock_steps=1, controls=controls)
    source.reset()

    assert source.step().active is True
    controls.set_paused(True)
    terminal = source.step()

    assert terminal.active is False
    assert controls.snapshot().status == "Failed"


def test_restart_survives_repeated_total_step_initialization() -> None:
    controls = ReplayControls(task="TurnOnMicrowave", episode=0)
    controls.set_total_steps(4)
    controls.request_restart()

    controls.set_total_steps(4)

    may_advance, restart_cycle = controls.claim_next_action()
    assert may_advance is True
    assert restart_cycle == 1


def test_transient_success_is_latched_until_final_verification() -> None:
    controls = ReplayControls(task="PrepareCoffee", episode=0)
    controls.set_total_steps(3)

    controls.update_observation(
        episode_step=1,
        cycle=0,
        progress=0.5,
        reward=1.0,
        success=True,
        action_norm=0.0,
    )
    controls.update_observation(
        episode_step=2,
        cycle=0,
        progress=1.0,
        reward=0.0,
        success=False,
        action_norm=0.0,
    )
    controls.mark_complete()

    snapshot = controls.snapshot()
    assert snapshot.status == "Success"
    assert snapshot.success is True


def test_simulator_reset_restores_existing_environment(monkeypatch) -> None:
    environment = object()
    initial_state = {"states": np.array([1.0]), "model": "<mujoco />"}
    reset_calls = []
    monkeypatch.setattr(
        app,
        "_reset_to_episode",
        lambda env, state: reset_calls.append((env, state)),
    )
    simulator = RoboCasaSimulator(mode="robocasa")
    simulator.env = environment
    simulator._initial_state = initial_state

    simulator.reset()

    assert reset_calls == [(environment, initial_state)]
    assert simulator._active_cycle is None
    assert simulator._last_episode_step == -1


def test_simulator_unwinds_viewers_when_console_start_fails(monkeypatch) -> None:
    closed: list[str] = []

    class FakeEnvironment:
        sim = object()

        def close(self) -> None:
            closed.append("environment")

    class FakeViewer:
        def update(self, _sim) -> None:
            return None

        def close(self) -> None:
            closed.append("viewer")

    class FailingConsole:
        def start(self) -> None:
            raise RuntimeError("console failed")

        def close(self) -> None:
            closed.append("console")

    lerobot = ModuleType("robocasa.utils.lerobot_utils")
    lerobot.get_env_metadata = lambda _path: {  # type: ignore[attr-defined]
        "env_name": "PrepareCoffee",
        "env_kwargs": {},
    }
    lerobot.get_episode_states = lambda _path, _episode: [np.zeros(1)]  # type: ignore[attr-defined]
    lerobot.get_episode_actions = lambda _path, _episode: [np.zeros(1)]  # type: ignore[attr-defined]
    lerobot.get_episode_model_xml = lambda _path, _episode: "<mujoco />"  # type: ignore[attr-defined]
    lerobot.get_episode_meta = lambda _path, _episode: {}  # type: ignore[attr-defined]
    robocasa = ModuleType("robocasa")
    robocasa_utils = ModuleType("robocasa.utils")
    robocasa_utils.lerobot_utils = lerobot  # type: ignore[attr-defined]
    robosuite = ModuleType("robosuite")
    environment = FakeEnvironment()
    robosuite.make = lambda **_kwargs: environment  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "robocasa", robocasa)
    monkeypatch.setitem(sys.modules, "robocasa.utils", robocasa_utils)
    monkeypatch.setitem(
        sys.modules,
        "robocasa.utils.lerobot_utils",
        lerobot,
    )
    monkeypatch.setitem(sys.modules, "robosuite", robosuite)
    monkeypatch.setattr(app, "_dataset_path", lambda *_args: "dataset")
    monkeypatch.setattr(app, "_reset_to_episode", lambda *_args: None)
    simulator = RoboCasaSimulator(mode="robocasa")
    simulator._web_viewer = FakeViewer()
    simulator._web_console = FailingConsole()

    with pytest.raises(RuntimeError, match="console failed"):
        simulator.reset()

    assert closed == ["console", "viewer", "environment"]
    assert simulator.env is None


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


def test_state_playback_is_deterministic_across_repeated_restarts(
    monkeypatch,
) -> None:
    class FakeEnvironment:
        def __init__(self) -> None:
            self.state = -1.0
            self.action_calls = 0

        def step(self, _action):
            self.action_calls += 1
            self.state += 0.125
            return None, 0.0, False, {}

        def _check_success(self) -> bool:
            return self.state == 2.0

    environment = FakeEnvironment()
    initial_state = {
        "states": np.array([-1.0]),
        "model": "<mujoco />",
        "ep_meta": '{"task": "PrepareCoffee"}',
    }
    restored_states: list[float] = []
    full_resets = 0

    def restore_state(env, state) -> None:
        nonlocal full_resets
        if "model" in state:
            full_resets += 1
        env.state = float(np.asarray(state["states"]).item())
        restored_states.append(env.state)

    monkeypatch.setattr(app, "_reset_to_episode", restore_state)
    simulator = RoboCasaSimulator(
        mode="robocasa",
        hz=1_000_000.0,
    )
    simulator.env = environment
    simulator._initial_state = initial_state
    simulator._action_count = 3

    terminal_results = []
    for cycle in range(3):
        for episode_step, state in enumerate((0.0, 1.0, 2.0)):
            terminal_results.append(
                simulator._step_robocasa(
                    RoboCasaAction(
                        values=np.array([99.0]),
                        recorded_state=np.array([state]),
                        episode_step=episode_step,
                        cycle=cycle,
                        active=True,
                    )
                )
            )

    assert environment.action_calls == 0
    assert full_resets == 2
    assert restored_states == [
        0.0,
        1.0,
        2.0,
        -1.0,
        0.0,
        1.0,
        2.0,
        -1.0,
        0.0,
        1.0,
        2.0,
    ]
    assert [result.success for result in terminal_results[2::3]] == [True] * 3


def test_action_playback_rejects_skipped_actions() -> None:
    class FakeEnvironment:
        def __init__(self) -> None:
            self.action_calls = 0

        def step(self, _action):
            self.action_calls += 1
            return None, 0.0, False, {}

        def _check_success(self) -> bool:
            return False

    environment = FakeEnvironment()
    simulator = RoboCasaSimulator(
        mode="robocasa",
        hz=1_000_000.0,
    )
    simulator.env = environment
    simulator._initial_state = {"states": np.array([0.0])}
    simulator._action_count = 3

    simulator._step_robocasa(
        RoboCasaAction(
            values=np.array([0.0]),
            episode_step=0,
            cycle=0,
            active=True,
        )
    )
    with pytest.raises(RuntimeError, match="skipped a recorded action"):
        simulator._step_robocasa(
            RoboCasaAction(
                values=np.array([0.0]),
                episode_step=2,
                cycle=0,
                active=True,
            )
        )

    assert environment.action_calls == 1


def test_verifier_uses_the_latest_observation_in_each_cycle() -> None:
    verifier = TaskVerifier()

    success = verifier.step(
        RoboCasaObservation(
            episode_step=2,
            cycle=0,
            progress=0.7,
            reward=1.0,
            success=True,
        )
    )
    transient_false = verifier.step(
        RoboCasaObservation(
            episode_step=3,
            cycle=0,
            progress=0.6,
            reward=0.0,
            success=False,
        )
    )
    restarted = verifier.step(
        RoboCasaObservation(
            episode_step=0,
            cycle=1,
            progress=0.0,
            reward=0.0,
            success=False,
        )
    )

    assert success.success is True
    assert transient_false.success is False
    assert transient_false.reward == 0.0
    assert transient_false.progress == 0.7
    assert restarted.success is False


def test_verifier_ignores_delayed_intermediate_success() -> None:
    controls = ReplayControls(task="PrepareCoffee", episode=0)
    controls.set_total_steps(3)
    verifier = TaskVerifier(controls=controls)

    verifier.step(
        RoboCasaObservation(
            episode_step=2,
            cycle=0,
            progress=1.0,
            success=False,
        )
    )
    controls.mark_complete()
    recovered = verifier.step(
        RoboCasaObservation(
            episode_step=1,
            cycle=0,
            progress=0.7,
            reward=1.0,
            success=True,
        )
    )

    assert recovered.episode_step == 2
    assert recovered.success is False
    assert controls.snapshot().status == "Failed"


def test_plan_progression_and_terminal_status_are_canonical() -> None:
    controls = ReplayControls(task="PrepareCoffee", episode=0)
    goal = EmbodiedGoal(text="Make coffee", task="PrepareCoffee")
    plan = OfflineEmbodiedPlanner().plan(goal)
    controls.configure_execution(goal, plan)
    controls.set_total_steps(5)
    verifier = TaskVerifier(controls=controls)

    verifier.step(
        RoboCasaObservation(
            episode_step=0,
            cycle=0,
            progress=0.0,
        )
    )
    assert controls.snapshot().current_step_id == "step-1"

    verifier.step(
        RoboCasaObservation(
            episode_step=2,
            cycle=0,
            progress=0.5,
        )
    )
    assert controls.snapshot().current_step_id == "step-3"

    verifier.step(
        RoboCasaObservation(
            episode_step=4,
            cycle=0,
            progress=1.0,
            reward=1.0,
            success=True,
        )
    )
    controls.mark_complete()
    succeeded = controls.snapshot()
    assert succeeded.status == "Success"
    assert succeeded.current_step_id == ""
    assert "Complete" not in {event.status for event in succeeded.events}

    controls.request_restart()
    restarted = controls.snapshot()
    assert restarted.status == "Restarting"
    assert restarted.current_step_id == "step-1"

    verifier.step(
        RoboCasaObservation(
            episode_step=4,
            cycle=1,
            progress=1.0,
            success=False,
        )
    )
    controls.mark_complete()
    failed = controls.snapshot()
    assert failed.status == "Failed"
    assert failed.current_step_id == ""
    assert "Complete" not in {event.status for event in failed.events}


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
    assert args.execution_mode == "demonstration"
    assert args.visualize == "mjviser"
    assert args.open_browser is False


def test_old_rerun_sdk_gets_plural_scalar_alias(monkeypatch) -> None:
    rerun = ModuleType("rerun")
    archetypes = ModuleType("rerun.archetypes")

    class Scalar:
        pass

    rerun.Scalar = Scalar
    rerun.archetypes = archetypes
    archetypes.Scalar = Scalar
    monkeypatch.setitem(sys.modules, "rerun", rerun)
    monkeypatch.setitem(sys.modules, "rerun.archetypes", archetypes)

    app._prepare_rerun_compat()

    assert rerun.Scalars is Scalar
    assert archetypes.Scalars is Scalar


def test_rerun_execution_uses_runtime_config_fields(monkeypatch) -> None:
    calls = []

    class PipelineStub:
        def run(self, **kwargs) -> None:
            calls.append(kwargs)

    args = _mock_args(
        mode="robocasa",
        visualize="rerun",
        video=None,
        seconds=0.1,
        rerun_mode="connect",
        rerun_address="127.0.0.1:9876",
    )
    monkeypatch.setattr(app.retriever, "init", lambda **_kwargs: None)
    monkeypatch.setattr(app, "build_pipeline", lambda _args: (PipelineStub(), object()))
    monkeypatch.setattr(app, "_prepare_rerun_compat", lambda: None)

    app._execute(args)

    assert calls[0]["backend_config"] == {
        "rerun_config": {
            "mode": "connect",
            "address": "127.0.0.1:9876",
        }
    }
