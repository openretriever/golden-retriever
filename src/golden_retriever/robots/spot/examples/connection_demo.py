#!/usr/bin/env python3
"""Minimal demo for SpotConnectionManager and its mock.

Usage:
  python -m retriever.robots.spot.examples.connection_demo --mock
  python -m retriever.robots.spot.examples.connection_demo --host 192.168.80.3 --user admin --password secret
"""
from __future__ import annotations

import argparse
from retriever.robots.spot.connection import (
    SpotConnectionConfig,
    SpotConnectionManager,
    MockSpotConnectionManager,
)
from retriever.types.core_types import Action, Command


def as_command(action_type: str, **params) -> Command:
    return Command(action=Action(type=action_type, parameters=params))


def run_demo(args) -> None:
    if args.mock:
        mgr = MockSpotConnectionManager(SpotConnectionConfig(host="mock"))
    else:
        cfg = SpotConnectionConfig(
            host=args.host,
            username=args.user,
            password=args.password,
            app_token=args.token,
        )
        mgr = SpotConnectionManager(cfg)

    print("Status:", mgr.status())

    for cmd in [
        as_command("move_to", x=1.0, y=0.5, yaw=1.57),
        as_command("open_gripper"),
        as_command("close_gripper"),
        as_command("stop"),
    ]:
        result = mgr.execute({"type": cmd.action.type, "parameters": cmd.action.parameters})
        print(cmd.action.type, "->", result)
    print("Final status:", mgr.status())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--user", type=str, default=None)
    parser.add_argument("--password", type=str, default=None)
    parser.add_argument("--token", type=str, default=None)
    run_demo(parser.parse_args())
