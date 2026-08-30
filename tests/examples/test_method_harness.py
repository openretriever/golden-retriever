from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from examples.advanced.openpi_policy.common import ActionChunk, PolicyObservation
from examples.advanced.robocasa.method_harness import (
    MethodHarness,
    SafetyEnvelope,
    Transition,
    TrialRequest,
    Verification,
)
from examples.advanced.robocasa.robosuite_lift import LiftEnvFlow


def _observation(step: int = 0) -> PolicyObservation:
    return PolicyObservation(
        image=np.zeros((4, 4, 3), dtype=np.uint8),
        state=np.array([step], dtype=np.float32),
        prompt="lift the cube",
    )


class FakeEnvironment:
    def __init__(self, *, success: bool = True) -> None:
        self.success = success
        self.steps = 0
        self.closed = False

    def reset(self, request: TrialRequest) -> PolicyObservation:
        assert request.seed == 7
        return _observation()

    def step(self, action: np.ndarray) -> Transition:
        self.steps += 1
        return Transition(
            observation=_observation(self.steps),
            reward=float(action[0]),
            terminated=self.steps == 3,
        )

    def verify(self) -> Verification:
        return Verification(
            success=self.success,
            reward=float(self.success),
            message="native task signal",
        )

    def close(self) -> None:
        self.closed = True


class ChunkMethod:
    def predict(self, observation: PolicyObservation) -> ActionChunk:
        del observation
        return ActionChunk(
            actions=np.full((2, 1), 0.25, dtype=np.float32),
            horizon=2,
            dof=1,
            source="scripted",
        )


def test_harness_runs_chunks_and_reports_native_verification() -> None:
    environment = FakeEnvironment()
    report = MethodHarness(
        safety=SafetyEnvelope(allowed_sources=frozenset({"scripted"}))
    ).run(
        TrialRequest(
            method_id="privileged-lift",
            task="Lift",
            seed=7,
            max_steps=8,
        ),
        environment=environment,
        method=ChunkMethod(),
    )

    assert report.status == "success"
    assert report.success is True
    assert report.steps == 3
    assert report.total_reward == 0.75
    assert report.verification.message == "native task signal"
    assert environment.closed is True
    assert [event.sequence for event in report.events] == list(
        range(len(report.events))
    )
    assert report.events[0].kind == "trial_started"
    assert report.events[-2].kind == "verification"
    assert report.events[-1].kind == "trial_completed"


def test_safety_envelope_rejects_invalid_chunks() -> None:
    safety = SafetyEnvelope(max_horizon=2, max_abs_action=1.0)
    invalid = ActionChunk(
        actions=np.array([[np.nan]], dtype=np.float32),
        horizon=1,
        dof=1,
        source="test",
    )

    with pytest.raises(ValueError, match="non-finite"):
        safety.validate(invalid)


def test_safety_envelope_requires_an_explicitly_allowed_source() -> None:
    chunk = ActionChunk(
        actions=np.zeros((1, 1), dtype=np.float32),
        horizon=1,
        dof=1,
        source="remote",
    )

    with pytest.raises(ValueError, match="source is not allowed"):
        SafetyEnvelope().validate(chunk)

    with pytest.raises(ValueError, match="source is not allowed"):
        SafetyEnvelope(allowed_sources=frozenset()).validate(chunk)


def test_harness_reports_policy_error_and_always_closes_environment() -> None:
    class InvalidMethod:
        def predict(self, observation: PolicyObservation) -> ActionChunk:
            del observation
            return ActionChunk(
                actions=np.ones((3, 1), dtype=np.float32),
                horizon=3,
                dof=1,
                source="scripted",
            )

    environment = FakeEnvironment()
    report = MethodHarness(safety=SafetyEnvelope(max_horizon=2)).run(
        TrialRequest(method_id="invalid", task="Lift", seed=7),
        environment=environment,
        method=InvalidMethod(),
    )

    assert report.status == "error"
    assert "horizon" in report.error
    assert environment.closed is True
    assert report.events[-2].kind == "trial_error"
    assert report.events[-1].kind == "trial_completed"


def test_harness_rejects_invalid_request_before_environment_reset() -> None:
    environment = FakeEnvironment()
    report = MethodHarness().run(
        TrialRequest(method_id="scripted", task="Lift", seed=7, max_steps=0),
        environment=environment,
        method=ChunkMethod(),
    )

    assert report.status == "error"
    assert environment.steps == 0
    assert environment.closed is True


def test_event_sink_failure_does_not_abort_trial() -> None:
    def broken_sink(_event) -> None:
        raise RuntimeError("observer failed")

    report = MethodHarness(event_sink=broken_sink).run(
        TrialRequest(method_id="scripted", task="Lift", seed=7, max_steps=8),
        environment=FakeEnvironment(),
        method=ChunkMethod(),
    )

    assert report.status == "success"


def test_lift_environment_forwards_reproduction_seed(monkeypatch) -> None:
    received: dict[str, object] = {}

    class FakeEnvironment:
        def reset(self) -> dict[str, object]:
            return {}

        def close(self) -> None:
            return None

    def make(**kwargs):
        received.update(kwargs)
        return FakeEnvironment()

    monkeypatch.setitem(sys.modules, "robosuite", SimpleNamespace(make=make))
    environment = LiftEnvFlow(
        mode="robosuite",
        env_name="Lift",
        robot="Panda",
        has_renderer=False,
        seed=17,
    )

    environment.init()
    environment.finalize()

    assert received["seed"] == 17
