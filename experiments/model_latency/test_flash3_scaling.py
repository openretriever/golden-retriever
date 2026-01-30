import os
import sys
import time
from PIL import Image
import random

# Add project root to path
sys.path.append(os.getcwd())

from experiments.closed_loop_planning.pipelines.vlm_utils import VLMPlanner
from experiments.closed_loop_planning.flows_test.planner_vlm import VLMTaskPlannerFlow


def create_dummy_image(size=(640, 480)):
    return Image.effect_noise(size, 10).convert("RGB")


def main():
    model = "gemini-3-flash-preview"
    planner = VLMPlanner(model_name=model)

    # Real-ish instruction for Planner
    instruction = """
    Task: remove all objects from the container
    Current Goal: at(green_apple, container) = no
    Past Action: pick(green_apple)
    Belief: holding(robot, green_apple) = yes
    """

    print(f"--- Deep Dive: {model} ---")

    results = []

    for count in [1, 3, 5, 10]:
        imgs = [create_dummy_image() for _ in range(count)]
        start = time.time()
        res = planner.plan_multi(imgs, instruction, mode="general")
        lat = time.time() - start
        results.append((count, lat))
        print(f"Planner ({count} imgs): {lat:.2f}s | Status: {res.get('status')}")

    print("\nScaling Analysis:")
    for count, lat in results:
        print(f"{count} images: {lat:.2f}s")


if __name__ == "__main__":
    main()
