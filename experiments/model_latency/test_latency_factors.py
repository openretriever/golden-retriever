import os
import sys
import time
from PIL import Image
import random

# Add project root to path
sys.path.append(os.getcwd())

from experiments.closed_loop_planning.pipelines.vlm_utils import VLMPlanner
from experiments.closed_loop_planning.flows.planner_vlm import VLMTaskPlannerFlow


def create_dummy_image(size=(640, 480)):
    return Image.effect_noise(size, 10).convert("RGB")


def test_scenario(model_name, scenario_name, func):
    try:
        start = time.time()
        func()
        lat = time.time() - start
        return lat
    except Exception as e:
        print(f"[{model_name}] {scenario_name} Failed: {e}")
        return None


def main():
    models = [
        "gemini-robotics-er-1.5-preview",
        "gemini-2.5-flash-lite",
        "gemini-3-flash-preview",
        "gemini-2.5-flash",
    ]

    # Image counts to test
    image_counts = [1, 3, 5, 10]

    results = {m: {} for m in models}

    print("Initializing Planners...")
    planners = {}
    for m in models:
        try:
            planners[m] = VLMPlanner(model_name=m)
        except:
            print(f"Could not init {m}")

    print("\nStarting Benchmark...")

    for model in models:
        if model not in planners:
            continue
        planner = planners[model]

        print(f"\n--- Testing {model} ---")

        # 1. Perception (Scaling Images)
        for count in [1, 3, 5]:
            imgs = [create_dummy_image() for _ in range(count)]
            lat = test_scenario(
                model,
                f"Perception ({count} imgs)",
                lambda: planner.plan_multi(
                    imgs, "Identify salient objects.", mode="perception"
                ),
            )
            if lat:
                results[model][f"Perc_{count}"] = lat
            print(f"Perception ({count} imgs): {lat:.2f}s")

        # 2. Belief Updater (Scaling Images)
        for count in image_counts:
            imgs = [create_dummy_image() for _ in range(count)]
            lat = test_scenario(
                model,
                f"Belief ({count} imgs)",
                lambda: planner.plan_multi(
                    imgs, "Update belief state.", mode="general"
                ),
            )
            if lat:
                results[model][f"Belief_{count}"] = lat
            print(f"Belief ({count} imgs):   {lat:.2f}s")

        # 3. Planner (Scaling Prompt + Images)
        flow = VLMTaskPlannerFlow(model=model)
        flow.planner = planner
        flow.init()

        instruction = """
        Task: remove all objects from the container
        Current Goal: at(green_apple, container) = no
        Past Action: pick(green_apple)
        Belief: holding(robot, green_apple) = yes
        """

        for count in [1, 3, 5]:
            imgs = [create_dummy_image() for _ in range(count)]
            lat = test_scenario(
                model,
                f"Planner ({count} imgs)",
                lambda: planner.plan_multi(imgs, instruction, mode="general"),
            )
            if lat:
                results[model][f"Plan_{count}"] = lat
            print(f"Planner ({count} imgs):    {lat:.2f}s")

    print(
        "\n\n==========================================================================================================="
    )
    print(
        f"{'Model':<35} | {'Perc(1)':<8} | {'Perc(5)':<8} | {'Bel(1)':<8} | {'Bel(10)':<8} | {'Plan(1)':<8} | {'Plan(5)':<8}"
    )
    print(
        "-----------------------------------------------------------------------------------------------------------"
    )
    for model in models:
        r = results[model]
        if not r:
            continue

        def fmt(k):
            return f"{r.get(k, 0):.2f}s" if k in r else "-"

        print(
            f"{model:<35} | {fmt('Perc_1'):<8} | {fmt('Perc_5'):<8} | {fmt('Belief_1'):<8} | {fmt('Belief_10'):<8} | {fmt('Plan_1'):<8} | {fmt('Plan_5'):<8}"
        )


if __name__ == "__main__":
    main()
