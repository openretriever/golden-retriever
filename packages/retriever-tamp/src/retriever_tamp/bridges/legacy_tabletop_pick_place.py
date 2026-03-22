"""Migration manifest for the current tabletop TAMP MVP.

This module keeps the package's migration story concrete without pretending that
all of the destination implementations already exist.
"""

from __future__ import annotations

from dataclasses import dataclass

LEGACY_EXAMPLE_PATH = "examples/advanced/tamp_tabletop_pick_place"


@dataclass(frozen=True)
class LegacyModuleMapping:
    legacy_file: str
    package_zone: str
    eventual_module: str | None
    note: str


LEGACY_MODULE_MAPPINGS: tuple[LegacyModuleMapping, ...] = (
    LegacyModuleMapping(
        legacy_file="domain.py",
        package_zone="retriever_tamp.symbolic + retriever_tamp.problems.tabletop_pick_place",
        eventual_module=None,
        note="Split reusable symbolic structs/operators from tabletop-specific problem setup.",
    ),
    LegacyModuleMapping(
        legacy_file="task_planner.py",
        package_zone="retriever_tamp.symbolic.planners",
        eventual_module="retriever_tamp.symbolic.planners.astar",
        note="First concrete reusable port; keep the planner independent from scene glue.",
    ),
    LegacyModuleMapping(
        legacy_file="motion_refiner.py",
        package_zone="retriever_tamp.refinement.providers",
        eventual_module="retriever_tamp.refinement.providers.tabletop_candidates",
        note="Preserve the lazy next-step refinement contract while swapping implementations later.",
    ),
    LegacyModuleMapping(
        legacy_file="scene.py",
        package_zone="retriever_tamp.problems.tabletop_pick_place",
        eventual_module="retriever_tamp.problems.tabletop_pick_place.scene_spec",
        note="Keep world/problem definition out of the generic refinement interfaces.",
    ),
    LegacyModuleMapping(
        legacy_file="app.py",
        package_zone="retriever_tamp.execution + retriever_tamp.bridges",
        eventual_module="retriever_tamp.bridges.legacy_tabletop_pick_place",
        note="The controller logic should migrate into the package; the example should become a thin adapter.",
    ),
)

PHASED_MIGRATION_STEPS: tuple[str, ...] = (
    "Keep the current example runnable as the low-friction debugging surface.",
    "Port reusable planner/refinement pieces into the reserved package landing zones.",
    "Rewrite the example as a thin bridge around retriever_tamp.execution.TAMPController.",
    "Only then add GoldenRetriever-specific runtime and RoboPlan adapters.",
)
