"""
TWIST2 Humanoid Simulation - Retriever Port

Demonstrates:
- Frequency decoupling (1000 Hz physics vs 50 Hz policy)
- @gui_flow for native MuJoCo viewer
- Dora backend for high-performance dataflow
"""
import argparse
import retriever
from retriever.flow import Latest
from flows import Twist2EnvFlow, Twist2PolicyFlow, MotionPlayerFlow, Twist2VisFlow


def main():
    parser = argparse.ArgumentParser(description="Retriever TWIST2 Port")
    parser.add_argument("--xml", type=str, default="TWIST2/assets/g1/g1_sim2sim_29dof.xml")
    parser.add_argument("--policy", type=str, default="TWIST2/assets/ckpts/twist2_1017_20k.onnx")
    parser.add_argument("--motion", type=str, default="TWIST2/assets/example_motions/0807_yanjie_walk_001.pkl")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--backend", type=str, default="dora")
    parser.add_argument("--no-viewer", action="store_true", help="Disable native viewer")
    args = parser.parse_args()

    # Define Flows with Rate Decoupling
    motion = MotionPlayerFlow(motion_file=args.motion) @ retriever.Rate(50)
    policy = Twist2PolicyFlow(policy_path=args.policy, device=args.device) @ retriever.Rate(50)
    env = Twist2EnvFlow(xml_path=args.xml) @ retriever.Rate(500)
    vis = Twist2VisFlow(xml_path=args.xml) @ retriever.Rate(30)

    # Build pipeline with .then() chaining
    motion.then(env, sync=Latest())
    motion.then(policy, sync=Latest())
    env.then(policy, sync=Latest())
    policy.then(env, sync=Latest())
    env.then(vis, map={"vis": "vis"}, sync=Latest())

    # Get pipeline from any handle
    pipe = motion.pipeline
    pipe.name = "twist2_demo"

    print("\n=== TWIST2 Simulation (Retriever Port) ===")
    print(f"  Physics: 500 Hz")
    print(f"  Policy:   50 Hz")
    print(f"  Backend: {args.backend}")
    print("==========================================")

    # Run with Rerun visualization enabled
    pipe.run(backend=args.backend, duration=10.0, visualize="rerun")


if __name__ == "__main__":
    main()
