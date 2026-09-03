import asyncio

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")
pytest.importorskip("mujoco")
pytest.importorskip("onnxruntime")

from examples.advanced.twist2_web_simulation.app import (
    BrowserTwist2State,
    WebProprioFlow,
    constants,
    index,
)


def test_twist2_web_ui_and_constants_are_available() -> None:
    html = asyncio.run(index())
    controller = asyncio.run(constants())

    assert "TWIST2 Web" in html
    assert "Loading MuJoCo WASM" in html
    assert "/ws/policy" in html
    assert controller["num_actions"] == 29
    assert len(controller["default_dof_pos"]) == 29


def test_browser_state_converts_to_policy_proprioception() -> None:
    qpos = [0.0] * 36
    qpos[2] = 0.8
    qpos[3] = 1.0
    state = BrowserTwist2State(
        time=0.0,
        qpos=qpos,
        qvel=[0.0] * 35,
        last_action=[0.0] * 29,
    )
    flow = WebProprioFlow()
    flow.reset()

    output = flow.step(state)

    assert output.proprio.shape == (92,)
    assert np.isfinite(output.proprio).all()
