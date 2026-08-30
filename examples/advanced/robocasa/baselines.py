"""Typed benchmark manifests for embodied-policy reference baselines."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass

from retriever.flow import io


@io
@dataclass(frozen=True)
class BaselineSpec:
    """One inspectable baseline configuration exposed by the demo launcher."""

    baseline_id: str = ""
    label: str = ""
    family: str = "Code-as-Policy"
    tier: str = "S1"
    task: str = "Lift"
    environment: str = "robosuite"
    interaction: str = "single-turn"
    grounding: str = "privileged"
    abstraction: str = "full-api"
    feedback: str = "none"
    examples: str = "included"
    runner: str = "reference"
    reference_url: str = "https://github.com/capgym/cap-x"
    license: str = "MIT"
    description: str = ""
    available: bool = False
    unavailable_reason: str | None = None


def discover_method_baselines(
    module_available: Callable[[str], bool] | None = None,
) -> tuple[BaselineSpec, ...]:
    """Describe code-as-policy methods without importing reference runtimes."""

    available = module_available or _module_available
    local_runtime = available("robosuite") and available("mjviser")
    missing_local = (
        None
        if local_runtime
        else "Install robosuite and mjviser in the simulator environment."
    )
    return (
        BaselineSpec(
            baseline_id="method-s1-privileged-lift",
            label="Scripted privileged cube lift",
            tier="S1",
            task="Lift",
            interaction="single-turn",
            grounding="privileged state",
            abstraction="full API",
            feedback="task reward",
            runner="robosuite_lift",
            description=(
                "A real MuJoCo/robosuite smoke run for the privileged tier. "
                "The controller is scripted and is not reported as a "
                "successful learned-policy trial."
            ),
            available=local_runtime,
            unavailable_reason=missing_local,
        ),
        BaselineSpec(
            baseline_id="method-s2-reduced-vision-stack",
            label="Reduced-API cube stack",
            tier="S2",
            task="Stack",
            interaction="single-turn",
            grounding="RGB-D",
            abstraction="reduced API",
            feedback="task reward",
            description=(
                "Planned adapter for RGB-D grounding and a reduced, "
                "allow-listed manipulation API."
            ),
            unavailable_reason="Requires the Linux vision and motion-planning worker.",
        ),
        BaselineSpec(
            baseline_id="method-m3-visual-feedback-stack",
            label="Visual-feedback cube stack",
            tier="M3",
            task="Stack",
            interaction="multi-turn",
            grounding="RGB-D",
            abstraction="reduced API",
            feedback="image differencing",
            examples="withheld",
            description=(
                "Planned multi-turn baseline that replans from bounded visual "
                "feedback while preserving Retriever's typed skill boundary."
            ),
            unavailable_reason="Requires the Linux vision and motion-planning worker.",
        ),
    )


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


__all__ = ["BaselineSpec", "discover_method_baselines"]
