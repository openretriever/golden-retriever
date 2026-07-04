"""OpenPI pi0.5 policy lane: observation -> pi0.5-style policy -> action chunks.

Run (mock, no dependencies beyond the default env):
  pixi run demo-pi05-mock
  pixi run python -m examples.advanced.openpi_policy.app --mode mock --steps 8 --dt 0.1

Run against a live openpi policy server (requires `openpi-client` and a
served pi0.5 checkpoint; see README.md):
  pixi run python -m examples.advanced.openpi_policy.app --mode remote --host <gpu-box> --port 8000

Load the policy flow from the Retriever Hub once the module is published
(design in README.md):
  pixi run python -m examples.advanced.openpi_policy.app --mode hub
"""

from __future__ import annotations

import argparse

from retriever.flow import Latest, Pipeline, Rate, Trigger

from examples.advanced.openpi_policy.common import (
    ActionChunkPrinter,
    MockPi05Policy,
    Pi05RemotePolicy,
    SyntheticManipObservation,
)

HUB_REF = "openretriever/pi05-policy:Pi05Policy"


def make_policy(args: argparse.Namespace):
    if args.mode == "mock":
        return MockPi05Policy(horizon=args.horizon, dof=args.dof)
    if args.mode == "remote":
        return Pi05RemotePolicy(host=args.host, port=args.port)
    if args.mode == "hub":
        from retriever import hub
        from retriever.error import HubError

        try:
            Policy = hub.use(HUB_REF)
        except HubError as exc:
            raise SystemExit(
                f"Hub module '{HUB_REF}' is not available yet ({exc}).\n"
                "This mode is the target integration; the packaging design lives in "
                "examples/advanced/openpi_policy/README.md. Use --mode mock meanwhile."
            ) from exc
        return Policy()
    raise ValueError(f"Unknown mode: {args.mode}")


def build_pipeline(args: argparse.Namespace) -> Pipeline:
    hz = 1.0 / max(args.dt, 1e-6)
    pipe = Pipeline("openpi_policy_lane")
    with pipe:
        source = SyntheticManipObservation(prompt=args.prompt, dof=args.dof) @ Rate(hz=hz)
        policy = make_policy(args) @ Trigger("image")
        printer = ActionChunkPrinter() @ Trigger("actions")
        pipe.connect(source, policy, sync=Latest())
        pipe.connect(policy, printer, sync=Latest())
    return pipe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="pi0.5-style policy over a synthetic manipulation scene.")
    parser.add_argument("--mode", choices=["mock", "remote", "hub"], default="mock")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--prompt", type=str, default="pick up the cup")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--dof", type=int, default=7)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipe = build_pipeline(args)
    try:
        for _ in range(args.steps):
            pipe.step(dt=args.dt)
    finally:
        pipe.close_stepper()


if __name__ == "__main__":
    main()
