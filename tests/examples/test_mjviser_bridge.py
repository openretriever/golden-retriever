from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from examples.advanced.robocasa.mjviser_bridge import (
    MjviserBridge,
    ReplayControls,
    _graph_html,
)


def test_replay_controls_pause_step_restart_and_speed() -> None:
    controls = ReplayControls(task="PrepareCoffee", episode=2)
    controls.set_total_steps(749)

    assert controls.claim_next_action() == (True, False)

    controls.set_paused(True)
    assert controls.claim_next_action() == (False, False)

    controls.request_step()
    assert controls.claim_next_action() == (True, False)
    assert controls.claim_next_action() == (False, False)

    controls.set_speed(0.5)
    controls.update_observation(
        episode_step=17,
        cycle=0,
        progress=18 / 749,
        reward=0.25,
        success=False,
        action_norm=1.5,
    )
    snapshot = controls.snapshot()
    assert snapshot.status == "Paused"
    assert snapshot.speed == 0.5
    assert snapshot.episode_step == 17
    assert snapshot.total_steps == 749

    controls.request_restart()
    assert controls.claim_next_action() == (True, True)
    snapshot = controls.snapshot()
    assert snapshot.status == "Restarting"
    assert snapshot.paused is False
    assert snapshot.progress == 0.0

    with pytest.raises(ValueError, match="positive"):
        controls.set_speed(0.0)


def test_live_graph_html_contains_typed_flow_and_escapes_task() -> None:
    controls = ReplayControls(task="Coffee <script>", episode=0)
    controls.set_total_steps(100)
    controls.update_observation(
        episode_step=24,
        cycle=0,
        progress=0.25,
        reward=1.0,
        success=False,
        action_norm=0.75,
    )

    html = _graph_html(controls.snapshot(), "Running")

    assert "DemoActionSource" in html
    assert "RoboCasaAction" in html
    assert "RoboCasaSimulator" in html
    assert "RoboCasaObservation" in html
    assert "ObservationPrinter" in html
    assert "action 25 of 100" in html
    assert "25.0% complete" in html
    assert "Coffee &lt;script&gt;" in html
    assert "Coffee <script>" not in html


def test_bridge_streams_native_robosuite_state(monkeypatch) -> None:
    servers = []
    scenes = []

    class FakeServer:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.stopped = False
            servers.append(self)

        def stop(self) -> None:
            self.stopped = True

    class FakeScene:
        def __init__(self, server, model, *, num_envs) -> None:
            self.server = server
            self.model = model
            self.num_envs = num_envs
            self.geom_groups_visible = [True] * 6
            self.synced = False
            self.gui_created = False
            self.updates = []
            scenes.append(self)

        def _sync_visibilities(self) -> None:
            self.synced = True

        def create_visualization_gui(self) -> None:
            self.gui_created = True

        def update_from_mjdata(self, data) -> None:
            self.updates.append(data)

    monkeypatch.setitem(sys.modules, "viser", SimpleNamespace(ViserServer=FakeServer))
    monkeypatch.setitem(
        sys.modules,
        "mjviser",
        SimpleNamespace(ViserMujocoScene=FakeScene),
    )

    model = object()
    data = object()
    sim = SimpleNamespace(
        model=SimpleNamespace(_model=model),
        data=SimpleNamespace(_data=data),
    )
    bridge = MjviserBridge(port=0, label="test")

    bridge.start(sim)
    bridge.update(sim)

    assert servers[0].kwargs == {"host": "127.0.0.1", "port": 0, "label": "test"}
    assert scenes[0].model is model
    assert scenes[0].num_envs == 1
    assert scenes[0].geom_groups_visible[0] is False
    assert scenes[0].synced is True
    assert scenes[0].gui_created is True
    assert scenes[0].updates == [data, data]

    bridge.close()
    assert servers[0].stopped is True
