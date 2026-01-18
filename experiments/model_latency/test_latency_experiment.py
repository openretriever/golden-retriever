import os
import sys
import time
from PIL import Image
import random

# Add project root to path
sys.path.append(os.getcwd())

from experiments.closed_loop_planning.pipelines.vlm_utils import VLMPlanner


def create_dummy_image(size=(640, 480)):
    # Create random noise image to simulate real data size
    return Image.effect_noise(size, 10).convert("RGB")


def test_model(model_name):
    print(f"\n==================================================")
    print(f"Testing Model: {model_name}")
    print(f"==================================================")

    try:
        planner = VLMPlanner(model_name=model_name)

        results = {}

        # Test 1: Perception (1 Image)
        img = create_dummy_image()
        start = time.time()
        try:
            planner.plan(img, "Identify salient objects.", mode="perception")
            lat = time.time() - start
            print(f"[Perception] 1 Image: {lat:.2f}s")
            results["Perception"] = lat
        except Exception as e:
            print(f"[Perception] Failed: {e}")
            results["Perception"] = "Fail"

        # Test 2: Belief (5 Images)
        images = [create_dummy_image() for _ in range(5)]
        start = time.time()
        try:
            planner.plan_multi(
                images, "Update belief state based on these frames.", mode="general"
            )
            lat = time.time() - start
            print(f"[Belief]     5 Images: {lat:.2f}s")
            results["Belief"] = lat
        except Exception as e:
            print(f"[Belief]     Failed: {e}")
            results["Belief"] = "Fail"

        # Test 3: Planner (1 Image, Complex)
        img = create_dummy_image()
        instruction = "Task: Remove all objects. Belief: [object at table]. Plan a sequence of actions."
        start = time.time()
        try:
            planner.plan(img, instruction, mode="general")
            lat = time.time() - start
            print(f"[Planner]    1 Image:  {lat:.2f}s")
            results["Planner"] = lat
        except Exception as e:
            print(f"[Planner]    Failed: {e}")
            results["Planner"] = "Fail"

        return results

    except Exception as e:
        print(f"Failed to initialize planner for {model_name}: {e}")
        return None


def main():
    models = [
        "gemini-robotics-er-1.5-preview",
        "gemini-2.5-flash-lite",
        "gemini-3-flash-preview",
        "gemini-2.5-flash",
    ]

    summary = {}

    for model in models:
        res = test_model(model)
        if res:
            summary[model] = res

    print("\n\n==================================================")
    print("BENCHMARK SUMMARY")
    print("==================================================")
    print(
        f"{'Model':<35} | {'Perception':<10} | {'Belief (5img)':<15} | {'Planner':<10}"
    )
    print("-" * 80)

    for model in models:
        if model in summary:
            res = summary[model]
            p = (
                f"{res['Perception']:.2f}s"
                if isinstance(res["Perception"], float)
                else str(res["Perception"])
            )
            b = (
                f"{res['Belief']:.2f}s"
                if isinstance(res["Belief"], float)
                else str(res["Belief"])
            )
            pl = (
                f"{res['Planner']:.2f}s"
                if isinstance(res["Planner"], float)
                else str(res["Planner"])
            )
            print(f"{model:<35} | {p:<10} | {b:<15} | {pl:<10}")
        else:
            print(f"{model:<35} | {'FAIL':<10} | {'FAIL':<15} | {'FAIL':<10}")


if __name__ == "__main__":
    main()
