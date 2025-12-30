#!/usr/bin/env python3
"""
Remote Connection Example - Distributed Execution

Demonstrates how to run parts of a pipeline on different machines
using the `deploy()` API with the Dora backend.

Scenario:
- Machine A (local): Runs the Robot Controller (hardware interface).
- Machine B (remote): Runs the Policy (heavy compute / GPU).
"""

import argparse
import logging

from retriever.flow import Pipeline, Rate, Trigger
from retriever.flow.adapter import Latest

from .components import Controller, Policy, RobotAction, RobotState


def main():
    parser = argparse.ArgumentParser(description="Remote Connection Demo")
    parser.add_argument(
        "--machine-a", 
        default="A", 
        help="Name of machine A (Controller) in dora config"
    )
    parser.add_argument(
        "--machine-b", 
        default="A", # Default to same machine for easy testing, user should change to B
        help="Name of machine B (Policy) in dora config"
    )
    parser.add_argument(
        "--backend",
        default="dora",
        help="Backend to use (must use 'dora' for distributed deployment)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Remote Connection Demo")
    print(f"  Controller Machine: {args.machine_a}")
    print(f"  Policy Machine:     {args.machine_b}")
    print("=" * 60)

    # 1. Define Flows
    # Controller runs at 10Hz
    controller = Controller() @ Rate(hz=10)
    
    # Policy runs when it receives state (or we could rate limit it)
    policy = Policy(compute_time=0.05) @ Trigger("state")

    # Option 1: Deploy using API (static)
    # controller.deploy(args.machine_a)
    # policy.deploy(args.machine_b)

    # Option 2: Deploy at runtime (see pipe.run below)


    # 3. Build Pipeline
    pipe = Pipeline("remote_demo")
    
    # Connect: Controller -> Policy (State)
    pipe.connect(controller, policy, map={"*": "state"})
    
    # Connect: Policy -> Controller (Action)
    # Use Latest() adapter so controller doesn't block waiting for policy
    pipe.connect(policy, controller, map={"*": "action"}, sync=Latest())

    # 4. Run
    # Note: For strict distributed execution, you might run this script to GENERATE the yaml,
    # and then use `dora start` remotely. 
    # But `pipe.run(backend="dora")` will try to spawn the coordinator if not running,
    # or connect to it.
    print(f"\nRunning with backend: {args.backend}")
    pipe.run(
        backend=args.backend,
        blocking=True,
        # Runtime deployment overrides
        deploy={
            controller: args.machine_a,
            policy: args.machine_b,
        }
        # duration=10.0 # Run for 10s then exit
    )



if __name__ == "__main__":
    main()
