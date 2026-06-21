"""
Main Application for Code as Policies Example.
"""

from .flows import TabletopEnvFlow, CodePolicyFlow, EnvAction

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Put the block in the bowl", help="Task description")
    parser.add_argument("--model", type=str, default="gemini-robotics-er-1.5-preview", help="Gemini Model ID")
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=logging.INFO)

    print(f"Starting Code as Policies Demo with task: '{args.task}'")
    
    # 1. Define Flows
    env_flow = TabletopEnvFlow()
    policy_flow = CodePolicyFlow(instruction=args.task, model=args.model, env_flow=env_flow)

    # 2. Define Pipeline
    # Topology:
    # Env (run initially) -> Obs
    # Obs -> Policy -> Action
    # Action -> Env (loop)
    
    # Note: For this demo, we run a manual loop, so we don't need declarative wiring
    # unless we were using the properties of the Pipeline runtime.
    
    # Standard Execution Loop
    env_flow.init()
    policy_flow.init()
    
    print("Pipeline started. Press Ctrl+C to stop.")
    action = EnvAction() # Init empty action

    import time
    try:
        while True:
            start = time.time()
            
            # 1. Env Step
            obs = env_flow.run(action)
            
            # 2. Policy Step
            action = policy_flow.run(obs)
            
            # 3. Visualization / Logging (Console mostly)
            # (Optional: Add Rerun logging here if needed)
            
            # Rate limit (10 Hz)
            elapsed = time.time() - start
            if elapsed < 0.1:
                time.sleep(0.1 - elapsed)
                
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        policy_flow.finalize()

if __name__ == "__main__":
    main()
