from argparse import Namespace

import pytest

from examples.advanced.robocasa.app import build_pipeline


def test_mock_robocasa_replay_reaches_success_without_simulator() -> None:
    args = Namespace(
        mode="mock",
        task="TurnOnMicrowave",
        split="pretrain",
        episode=0,
        repeat=False,
        mock_steps=6,
        hz=20.0,
        visualize="none",
        viewer=False,
        camera="robot0_agentview_center",
        width=320,
        height=180,
        image_hz=5.0,
        print_every=100,
    )
    pipeline, simulator = build_pipeline(args)
    try:
        for _ in range(7):
            pipeline.step(dt=1.0 / args.hz)
        latest = simulator.latest
    finally:
        pipeline.close_stepper()

    assert latest is not None
    assert latest.source == "mock"
    assert latest.progress == pytest.approx(1.0)
    assert latest.success is True
    assert simulator.env is None
