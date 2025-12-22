import argparse

from retriever.flow import Pipeline, Rate, Trigger
from .components import (
    RobotEnv, Commander, SkillRouter, ApproachSkill, ManipulateSkill, ActionArbiter
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=15.0)
    args = parser.parse_args()

    env = RobotEnv() @ Rate(hz=10.0)
    # Commander runs open-loop on time
    commander = Commander() @ Rate(hz=10.0)
    # Router runs on Rate to ensuring checking inputs
    router = SkillRouter() @ Rate(hz=20.0)
    # Skills input field is 'flow'.
    approach = ApproachSkill() @ Trigger("flow")
    manipulate = ManipulateSkill() @ Trigger("flow")
    arbiter = ActionArbiter() @ Rate(hz=20.0) # Check frequently

    pipe = Pipeline("skill_switching_demo")

    # Wire: Env & Commander -> Router
    pipe.connect(env, router, map={
        "x": "x",
        "y": "y",
        "gripper_open": "gripper_open",
        "holding_object": "holding_object"
    })
    
    pipe.connect(commander, router, map={
        "mode": "mode",
        "target_x": "target_x",
        "target_y": "target_y"
    })

    # Wire: Router -> Skills
    pipe.connect(router, approach, map={"approach_flow": "flow"})
    pipe.connect(router, manipulate, map={"manipulate_flow": "flow"})

    # Wire: Skills -> Arbiter
    pipe.connect(approach, arbiter, map={"packet": "approach_packet"})
    pipe.connect(manipulate, arbiter, map={"packet": "manipulate_packet"})

    # Wire: Arbiter -> Env
    # Now both use ActionSignal (containing 'packet')
    # map packet -> packet
    pipe.connect(arbiter, env, map={"packet": "packet"})

    print("Starting Skill Switching Demo (15s)...")
    pipe.run(duration=args.duration, blocking=True)


if __name__ == "__main__":
    main()
