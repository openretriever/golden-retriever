"""Shared flows and payloads for the OpenPI pi0.5 policy example lane.

Three policy flows share the same typed contract
(`PolicyObservation -> ActionChunk`):

- `MockPi05Policy`: deterministic, dependency-free; what CI runs.
- `Pi05RemotePolicy`: thin client for an `openpi` websocket policy server
  (requires `openpi-client`; the model runs elsewhere, e.g. a GPU box).
- The hub path (`retriever.hub.use("openretriever/pi05-policy:...")`) loads
  the same contract from a published module — see README.md in this folder.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from retriever.flow import Flow, io


@io
@dataclass
class PolicyObservation:
    """One observation frame for a manipulation policy."""

    image: np.ndarray  # (H, W, 3) uint8 base camera
    state: np.ndarray  # (dof,) float32 proprioception
    prompt: str        # natural-language task instruction


@io
@dataclass
class ActionChunk:
    """A pi0-style action chunk: `horizon` future actions, planned at once."""

    actions: np.ndarray  # (horizon, dof) float32
    horizon: int
    dof: int
    source: str  # which policy produced it ("mock", "remote", "hub")


class SyntheticManipObservation(Flow[None, PolicyObservation]):
    """Deterministic observation source: a synthetic scene plus a slow ramp state."""

    def __init__(self, *, prompt: str = "pick up the cup", dof: int = 7) -> None:
        super().__init__()
        self.prompt = prompt
        self.dof = int(dof)

    def init_config(self) -> dict:
        return {"prompt": self.prompt, "dof": self.dof}

    def init(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.frame_index = 0

    def step(self, _):  # type: ignore[override]
        self.frame_index += 1
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[24:40, 24:40, 0] = 200  # a "cup"
        state = (
            0.1 * np.sin(self.frame_index * 0.2 + np.arange(self.dof))
        ).astype(np.float32)
        return PolicyObservation(image=image, state=state, prompt=self.prompt)


class MockPi05Policy(Flow[PolicyObservation, ActionChunk]):
    """Deterministic stand-in with pi0.5-shaped output (action chunks).

    Real pi0.5 plans a chunk of future actions per inference call; the mock
    reproduces that interface with a smooth, prompt-independent trajectory so
    downstream wiring can be developed and tested without model weights.
    """

    def __init__(self, *, horizon: int = 10, dof: int = 7) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.dof = int(dof)

    def init_config(self) -> dict:
        return {"horizon": self.horizon, "dof": self.dof}

    def step(self, obs: PolicyObservation) -> ActionChunk:
        base = obs.state if obs.state is not None else np.zeros(self.dof, dtype=np.float32)
        t = np.linspace(0.0, 1.0, self.horizon, dtype=np.float32)[:, None]
        actions = base[None, : self.dof] * (1.0 - t) + 0.05 * t
        return ActionChunk(
            actions=actions.astype(np.float32),
            horizon=self.horizon,
            dof=self.dof,
            source="mock",
        )


class Pi05RemotePolicy(Flow[PolicyObservation, ActionChunk]):
    """Client flow for an openpi websocket policy server.

    Serve pi0.5 on a GPU machine with openpi's `scripts/serve_policy.py`,
    then point this flow at it. Only `openpi-client` is needed locally.

    Note: the observation dict layout must match the served config
    (e.g. DROID configs expect `observation/*` keys). Adjust `_pack` for
    your deployment; this default follows the DROID convention.
    """

    def __init__(self, *, host: str = "localhost", port: int = 8000) -> None:
        super().__init__()
        self.host = host
        self.port = int(port)
        self._client = None

    def init_config(self) -> dict:
        return {"host": self.host, "port": self.port}

    def init(self) -> None:
        try:
            from openpi_client import websocket_client_policy
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Pi05RemotePolicy requires the `openpi-client` package: "
                "pip install openpi-client (see examples/advanced/openpi_policy/README.md)"
            ) from exc
        self._client = websocket_client_policy.WebsocketClientPolicy(
            host=self.host, port=self.port
        )

    def _pack(self, obs: PolicyObservation) -> dict:
        return {
            "observation/image": obs.image,
            "observation/state": obs.state,
            "prompt": obs.prompt,
        }

    def step(self, obs: PolicyObservation) -> ActionChunk:
        result = self._client.infer(self._pack(obs))
        actions = np.asarray(result["actions"], dtype=np.float32)
        return ActionChunk(
            actions=actions,
            horizon=int(actions.shape[0]),
            dof=int(actions.shape[-1]),
            source="remote",
        )


class ActionChunkPrinter(Flow[ActionChunk, None]):
    """Print a compact summary of each received chunk."""

    def step(self, chunk: ActionChunk) -> None:
        if chunk.actions is None:
            return None
        first = np.array2string(chunk.actions[0], precision=3, suppress_small=True)
        print(
            f"[{chunk.source}] chunk horizon={chunk.horizon} dof={chunk.dof} "
            f"first_action={first}"
        )
        return None
