import retriever
from retriever.flow import Latest
from flows import MujocoEnvFlow, ControllerFlow, RerunLoggerFlow


def main():
    # 1. Instantiate Flows with distinct Rates
    # Physics runs at 200 Hz (sufficient for this demo)
    env = MujocoEnvFlow() @ retriever.Rate(200)

    # Controller runs slower (50 Hz), typical for RL/inference
    ctrl = ControllerFlow() @ retriever.Rate(50)

    # Visualization runs at 30 Hz (human-viewable)
    logger = RerunLoggerFlow() @ retriever.Rate(30)

    # 2. Wire the Closed Loop
    # Env -> Controller: Controller sees the LATEST state
    retriever.connect(env, ctrl, sync=Latest(), qsize=100)

    # Controller -> Env: Env sees the LATEST control
    retriever.connect(ctrl, env, sync=Latest(), qsize=100)

    # Env -> Logger: Logger sees the LATEST state (for visualization)
    retriever.connect(env, logger, sync=Latest(), qsize=100)

    # 3. Run
    print("Running MuJoCo Simulation with Rerun Visualization...")
    print("  Physics Rate: 1000 Hz")
    print("  Control Rate: 50 Hz")
    print("  Render Rate:  30 Hz (Rerun)")
    print("  Sync Policy:  Latest (Async/Decoupled)")

    # Run for 5 seconds to see the arm converge
    # Using 'dora' backend to demonstrate high-performance Rust runtime
    retriever.run(backend="dora", duration=5.0)


if __name__ == "__main__":
    main()
