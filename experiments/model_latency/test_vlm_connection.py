import os
import sys
import time
from PIL import Image

# Add project root to path
sys.path.append(os.getcwd())

from experiments.closed_loop_planning.pipelines.vlm_utils import VLMPlanner


def main():
    print("Initializing VLMPlanner...")
    planner = VLMPlanner(model_name="gemini-robotics-er-1.5-preview")
    print(f"Model: {planner.model_name}")

    # Create dummy image
    img = Image.new("RGB", (128, 128), color="red")

    print("Sending native request...")
    start = time.time()
    try:
        res = planner.plan(
            img, "Describe this image in one sentence.", mode="perception"
        )
        print(f"Response: {res}")
    except Exception as e:
        print(f"VLM Error: {e}")
        import traceback

        traceback.print_exc()

    print(f"Latency: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
