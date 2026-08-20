"""Optional browser visualization for robosuite-backed environments."""

from __future__ import annotations

from typing import Any


class MjviserBridge:
    """Publish an existing robosuite MuJoCo state through mjviser."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8085,
        label: str = "Retriever simulation",
    ) -> None:
        self.host = host
        self.port = int(port)
        self.label = label
        self._server: Any | None = None
        self._scene: Any | None = None

    def start(self, sim: Any) -> None:
        if self._scene is not None:
            return

        try:
            import viser
            from mjviser import ViserMujocoScene
        except ImportError as exc:
            raise RuntimeError(
                "mjviser is not installed. Install the optional simulation dependencies "
                'with `python -m pip install -e ".[robosuite]"`.'
            ) from exc

        model, data = _native_mujoco_state(sim)
        self._server = viser.ViserServer(
            host=self.host,
            port=self.port,
            label=self.label,
        )
        self._scene = ViserMujocoScene(self._server, model, num_envs=1)

        # robosuite uses group 0 for collision proxies and group 1 for visual geoms.
        # Keep collisions available in the Groups tab, but do not overlay them by default.
        self._scene.geom_groups_visible[0] = False
        self._scene._sync_visibilities()
        self._scene.create_visualization_gui()
        self._scene.update_from_mjdata(data)

        display_host = (
            "localhost" if self.host in {"0.0.0.0", "127.0.0.1"} else self.host
        )
        print(f"Retriever mjviser: http://{display_host}:{self.port}")

    def update(self, sim: Any) -> None:
        self.start(sim)
        if self._scene is None:
            return
        _, data = _native_mujoco_state(sim)
        self._scene.update_from_mjdata(data)

    def restart(self, sim: Any) -> None:
        """Rebuild the browser scene after a simulator model reset."""

        self.close()
        self.start(sim)

    def close(self) -> None:
        if self._server is not None:
            self._server.stop()
        self._scene = None
        self._server = None


def _native_mujoco_state(sim: Any) -> tuple[Any, Any]:
    """Return native MuJoCo objects from robosuite wrappers or direct objects."""

    model = getattr(sim.model, "_model", sim.model)
    data = getattr(sim.data, "_data", sim.data)
    return model, data
