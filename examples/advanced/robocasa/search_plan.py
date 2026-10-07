"""The search plan: where the robot intends to look, in order.

A search plan cannot be a fixed script. Its length is unknown until the target
turns up, and every observation can change what the rest of it should be — so
this is an open-ended payload rather than `embodied.SkillPlan`, whose
`validate()` requires contiguous progress fractions covering `[0, 1]` and
therefore assumes the step count is known up front. Fabricating fractions for
steps that may never run would make the contract lie; weakening `SkillPlan`
would break the lanes that rely on it. `SkillPlan` remains the right shape for
*displaying* a committed plan, and projecting onto it is a display concern.

The plan is a prediction, not a commitment, and recording both makes the gap
visible: `nearest-first` chooses by distance from wherever the robot currently
stands, so acting on the plan moves the robot and can reorder everything after
the first step. Each time the remaining order stops matching the prediction,
that is a **replan** — which the catalog names as a system metric worth
reporting alongside task success.
"""

from __future__ import annotations

from dataclasses import dataclass

from retriever.flow import io

from .search_belief import InspectionResult, SearchBelief
from .search_scenes import SearchScene

__all__ = ["SearchPlan", "plan_ahead"]


@io
@dataclass(frozen=True)
class SearchPlan:
    """The order the policy intends to inspect containers, given what it knows."""

    policy: str = ""
    revision: int = 0
    container_ids: tuple[str, ...] = ()
    basis: str = ""

    @property
    def next_container(self) -> str:
        return self.container_ids[0] if self.container_ids else ""

    def as_dict(self) -> dict:
        return {
            "policy": self.policy,
            "revision": self.revision,
            "container_ids": list(self.container_ids),
            "basis": self.basis,
        }


def plan_ahead(
    policy,
    belief: SearchBelief,
    scene: SearchScene,
    position: tuple[float, float, float] | None = None,
    *,
    revision: int = 0,
) -> SearchPlan:
    """Roll the policy forward over a hypothetical belief to get its intended order.

    Pure lookahead: it never touches the world. Each candidate is assumed to
    come back empty, which is what the policy would have to assume anyway — the
    alternative (the target being there) ends the search, so the plan beyond
    that point is irrelevant.
    """
    hypothetical = belief
    here = position if position is not None else scene.robot_home
    order: list[str] = []

    # A plan can never be longer than the number of containers.
    for _ in range(len(scene.containers)):
        decision = policy.decide(hypothetical, scene, here)
        if decision.stops or not decision.container_id:
            break
        order.append(decision.container_id)
        here = scene.container(decision.container_id).position
        hypothetical = hypothetical.update(
            InspectionResult(container_id=decision.container_id, found=False)
        )

    return SearchPlan(
        policy=getattr(policy, "name", "unknown"),
        revision=revision,
        container_ids=tuple(order),
        basis=(
            f"rolled {getattr(policy, 'name', 'policy')} forward over "
            f"{len(belief.candidates)} candidate(s)"
        ),
    )
