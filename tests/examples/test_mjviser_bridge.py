from __future__ import annotations

import sys
from types import SimpleNamespace

from examples.advanced.robocasa.mjviser_bridge import MjviserBridge


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
