"""Run a Retriever `Flow` as a study method.

Until this existed, every arm in this study was a plain dataclass with a
`predict` method, so the study measured policies that happened to live in this
repository rather than policies written against Retriever's own contract. The
framework's policy shape is an ordinary `Flow[PolicyObservation, ActionChunk]`
— that is what `examples/advanced/openpi_policy` declares for pi0.5 in all
three of its sourcing modes, and what `robosuite_lift.ScriptedLiftMethod`
already bridges for the Lift heuristic. This generalises that bridge.

**Chunk truncation is the benchmark's, not ours.** The RoboCasa365 evaluation
client (`robocasa-benchmark/openpi`, `examples/robocasa/main.py`) keeps only
the leading actions of each predicted chunk:

    replan_steps: int = 5
    action_plan.extend(action_chunk[: replan_steps])

`take` implements exactly that, for any Flow. The harness then dispatches those
rows and calls `predict` again, which re-queries the policy on a fresh
observation — so the re-planning cadence is expressed by the adapter rather
than hidden inside each policy's own queue.

**Why `init_config` is recorded.** A cell reproducible only from live Python
objects is not reproducible. Retriever's `Flow` declares `init_config()` and
`from_init_config()` precisely so a graph can be rebuilt from serialised form,
and the sibling drawer harness learned the same lesson the hard way: a run
whose IR cannot name which policy produced it will happily report numbers for
one policy under the name of another. So the adapter records both the
reference string and the config, and refuses a policy that cannot describe
itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from examples.advanced.openpi_policy.common import ActionChunk, PolicyObservation


class FlowContractError(TypeError):
    """The wrapped object is not a usable `PolicyObservation -> ActionChunk` Flow."""


def policy_reference(policy) -> str:
    """A `module:ClassName` reference for the policy, as bklva's registry uses."""
    cls = type(policy)
    return f"{cls.__module__}:{cls.__qualname__}"


def _config_of(policy) -> dict:
    config = getattr(policy, "init_config", None)
    if not callable(config):
        raise FlowContractError(
            f"{policy_reference(policy)} has no init_config(); a policy that cannot "
            "describe itself cannot be reproduced from a run record")
    value = config()
    if not isinstance(value, dict):
        raise FlowContractError("init_config() must return a dict")
    return value


@dataclass
class FlowMethod:
    """Adapt a `Flow[PolicyObservation, ActionChunk]` to the study's methods.

    `take` truncates each predicted chunk to its leading rows, which is the
    official evaluation client's `replan_steps`. Leave it `None` to execute
    whatever horizon the policy proposes.
    """

    policy: object
    take: int | None = None
    #: Overrides the chunk's own `source`. Left empty, the policy's label is
    #: kept, so a method cannot be quietly relabelled by the adapter.
    source: str = ""
    calls: int = 0
    rows_dispatched: int = 0
    rows_discarded: int = 0
    reference: str = ""
    config: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.take is not None and (type(self.take) is not int or not 1 <= self.take <= 64):
            raise ValueError("take must be None or an int from 1 through 64")
        if not isinstance(self.source, str):
            raise TypeError("source must be a string")
        step = getattr(self.policy, "step", None)
        if not callable(step):
            raise FlowContractError(
                f"{policy_reference(self.policy)} has no step(); a Retriever policy is a "
                "Flow whose step maps an observation to an action chunk")
        self.reference = policy_reference(self.policy)
        self.config = _config_of(self.policy)
        # `reset` is the current name; `init` is its deprecated alias, and the
        # openpi lane still calls the old one, so accept either.
        for name in ("reset", "init"):
            hook = getattr(self.policy, name, None)
            if callable(hook):
                hook()
                break

    def reset(self, seed: int | None = None) -> None:
        """Re-seed the wrapped policy for one episode, when it accepts a seed.

        A sampling policy — SmolVLA integrates from sampled noise — returns a
        different chunk for the same observation on every call, so without this
        an episode is not reproducible and a port cannot be validated against
        the run it is meant to reproduce. Policies that take no seed are left
        alone rather than reset with an argument they never declared.
        """
        hook = getattr(self.policy, "reset", None)
        if not callable(hook):
            return
        try:
            hook(seed)
        except TypeError:
            hook()

    # --- the harness contract ------------------------------------------

    def predict(self, observation: PolicyObservation) -> ActionChunk:
        self.calls += 1
        chunk = self.policy.step(observation)
        if chunk is None:
            raise FlowContractError(f"{self.reference} returned no action chunk")
        actions = np.asarray(getattr(chunk, "actions", None), dtype=np.float32)
        if actions.ndim != 2 or not actions.size:
            raise FlowContractError(
                f"{self.reference} returned actions of shape {actions.shape}; "
                "a chunk must be (horizon, dof)")
        proposed = int(actions.shape[0])
        if self.take is not None:
            actions = actions[: self.take]
        kept = int(actions.shape[0])
        self.rows_dispatched += kept
        self.rows_discarded += proposed - kept
        return ActionChunk(actions=np.ascontiguousarray(actions), horizon=kept,
                           dof=int(actions.shape[1]),
                           source=self.source or getattr(chunk, "source", "") or "model")

    def describe(self) -> dict:
        """What ran, in terms a run record can be rebuilt from."""
        described = {}
        inner = getattr(self.policy, "describe", None)
        if callable(inner):
            value = inner()
            if isinstance(value, dict):
                described = dict(value)
        described.update({
            "method": "flow",
            "policy_ref": self.reference,
            "policy_config": dict(self.config),
            "take": self.take,
            "calls": self.calls,
            "rows_dispatched": self.rows_dispatched,
            "rows_discarded": self.rows_discarded,
        })
        return described
