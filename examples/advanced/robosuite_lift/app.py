"""Backward-compatible entry point for the consolidated RoboSuite Lift demo."""

from examples.advanced.robocasa.robosuite_lift import (
    HeuristicLiftPolicy as _HeuristicLiftPolicy,
)
from examples.advanced.robocasa.robosuite_lift import (
    LiftAction,
    LiftEnvFlow,
    LiftPrinter,
    LiftState,
    build_pipeline,
    main,
    parse_args,
)


class HeuristicLiftPolicy(_HeuristicLiftPolicy):
    """Compatibility policy that is ready for direct, non-Pipeline use."""

    def __init__(self, *, target_height: float) -> None:
        super().__init__(target_height=target_height)
        self.init()


__all__ = [
    "HeuristicLiftPolicy",
    "LiftAction",
    "LiftEnvFlow",
    "LiftPrinter",
    "LiftState",
    "build_pipeline",
    "main",
    "parse_args",
]


if __name__ == "__main__":
    main()
