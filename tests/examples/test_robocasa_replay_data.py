from __future__ import annotations

import os

import pytest

from examples.advanced.robocasa.app import DemoActionSource, RoboCasaSimulator


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_ROBOCASA_DATA_TESTS") != "1",
    reason="real RoboCasa assets and demonstrations are opt-in",
)


@pytest.mark.parametrize("task", ["CoffeeSetupMug", "PrepareCoffee"])
def test_recorded_episode_reaches_robocasa_success(task: str) -> None:
    source = DemoActionSource(mode="robocasa", task=task, episode=0)
    simulator = RoboCasaSimulator(mode="robocasa", task=task, episode=0, hz=1e9)

    try:
        source.reset()
        simulator.reset()
        assert source.actions is not None

        observation = None
        for _ in range(len(source.actions)):
            observation = simulator.step(source.step())

        assert observation is not None
        assert observation.progress == pytest.approx(1.0)
        assert observation.success is True
    finally:
        simulator.finalize()
