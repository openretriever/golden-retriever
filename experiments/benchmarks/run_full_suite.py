
import subprocess
import os
import sys

# Define commands
commands = [
    # Verify clean slate (though we just did it)
    "rm -f experiments/benchmarks/results/*.csv",
    
    # Run Benchmarks
    "pixi run benchmark-retriever-dora",
    "pixi run benchmark-retriever-mp",
    "pixi run benchmark-retriever-in-process",
    
    # Dora Native (requires daemon restart usually handled by the task, but let's be safe)
    "pixi run benchmark-dora-suite",
    
    # Plot
    "pixi run benchmark-plot"
]

def run():
    print("Starting full benchmark suite...")
    for cmd in commands:
        print(f"\n>>> Running: {cmd}")
        # explicit shell=True for 'rm' glob expansion and pixi aliases
        ret = subprocess.run(cmd, shell=True)
        if ret.returncode != 0:
            print(f"Command failed: {cmd}")
            # We fail hard to notice issues
            sys.exit(ret.returncode)
    print("\n>>> All benchmarks completed successfully.")

if __name__ == "__main__":
    run()
