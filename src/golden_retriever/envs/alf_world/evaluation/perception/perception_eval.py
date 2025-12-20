# this is a script to evaluate the performance of the perception module
# it will take in a trajectory and evaluate the performance of the perception module
# it will output the performance of the perception module

# First, we need to load the trajectory
# Then, we need to evaluate the performance of the perception module
# Finally, we need to output the performance of the perception module
import argparse
import json
import os

from agents.agent_utils.observer import SingleImageObserver, TwoImageObserver
from evaluation.perception.evaluator import Evaluator
from PIL import Image
from tqdm import tqdm


def load_trial_data(trial_dir, perception_type):
    assert os.path.exists(
        f"{trial_dir}/traj.json"
    ), f"Trajectory file not found at {trial_dir}/traj.json"
    with open(f"{trial_dir}/traj.json", "r") as f:
        traj = json.load(f)
        task_info = traj["initial_info"]
        # load the observations in png format
        obs_dir = f"{trial_dir}/{perception_type}"
        # load the observations in the order of the name of the files 0, 1, 2, ...
        # Sort observation files numerically (0.png, 1.png, 2.png etc)
        obs_files = sorted(
            os.listdir(obs_dir), key=lambda x: int(os.path.splitext(x)[0])
        )
        observations = []
        for obs_file in obs_files:
            # load a list of observations in png format
            # Construct full path to observation file
            obs_path = os.path.join(obs_dir, obs_file)

            # Load image using PIL
            try:
                observation = Image.open(obs_path)
                # Add to list of observations
                observations.append(observation)
            except Exception as e:
                print(f"Error loading image {obs_path}: {e}")

        actions = [step["action"] for step in traj["steps"]]
        object_bindings = [step["object_bindings"] for step in traj["steps"]]
        frame_descriptions = [step["frame_description"] for step in traj["steps"]]
        inventory = [step["inventory"] for step in traj["steps"]]

        # remove the first action and frame description and truncate the length to 10 to reduce repetitions
        actions = actions[1:]
        object_bindings = object_bindings[1:]
        frame_descriptions = frame_descriptions[1:]
        observations = observations[1:]
        inventory = inventory[1:]
        truncated_length = min(len(actions), 15)
        actions = actions[:truncated_length]
        object_bindings = object_bindings[:truncated_length]
        frame_descriptions = frame_descriptions[:truncated_length]
        observations = observations[:truncated_length]
        inventory = inventory[:truncated_length]

        return (
            observations,
            actions,
            object_bindings,
            frame_descriptions,
            inventory,
            task_info,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_num", type=int, default=20)
    parser.add_argument("--perception_type", type=str, default="bbox")
    parser.add_argument("--vlm", type=str, default="gpt-4o")
    parser.add_argument("--llm", type=str, default="gpt-4o")
    parser.add_argument("--is-double-image", type=bool, default=False)
    args = parser.parse_args()
    if args.is_double_image:
        observer = TwoImageObserver(args.vlm, args.perception_type)
    else:
        observer = SingleImageObserver(args.vlm, args.perception_type)
    evaluator = Evaluator(args.llm)
    # specify the path to the trajectory json file
    eval_num = args.eval_num
    # perception_type can either be "segs" or "bbox"
    perception_type = observer.perception_type
    traj_dir = "./data/rule_based_expert/trajectory"
    # load traj_dir/task_type_dir/{trial_id}/traj.json
    # get task_type and trial_id under the traj_dir
    trial_dirs = []
    task_types = [d for d in os.listdir(traj_dir) if d != ".DS_Store"]
    for task_type in task_types:
        trial_ids = os.listdir(f"{traj_dir}/{task_type}")
        for trial_id in trial_ids:
            trial_dir = f"{traj_dir}/{task_type}/{trial_id}"
            trial_dirs.append(trial_dir)

    print(f"Overall we have {len(trial_dirs)} trials...")
    trial_dirs = trial_dirs[:eval_num]
    print(f"We will evaluate {len(trial_dirs)} trials...")
    # load the trajectory json file
    for trial_dir in tqdm(trial_dirs, desc="Evaluating trials"):
        (
            observations,
            actions,
            object_bindings,
            frame_descriptions,
            inventorys,
            task_info,
        ) = load_trial_data(trial_dir, perception_type)
        evaluator.start_trial(trial_name=trial_dir, task_info=task_info)
        observer.add_task_info(task_info)
        observer.set_logger(trial_dir)
        evaluator.set_logger(trial_dir, observer.name)
        # Evaluate perception for each observation and action pair
        for i, (obs, action, frame_desc, object_binding, inventory) in enumerate(
            zip(
                observations,
                actions,
                frame_descriptions,
                object_bindings,
                inventorys,
                strict=False,
            )
        ):
            print(f"\nStep {i+1}:")
            print(f"Action: {action}")
            print(f"Ground Truth Description: {frame_desc}")
            observer.info(f"\nStep {i+1}:")
            observer.info(f"Action: {action}")
            observer.info(f"Ground Truth Description: {frame_desc}")
            cur_obs = {
                "image": obs,
                "object_bindings": object_binding,
                "inventory": inventory,
            }

            observer.add_action(action)
            observer.observe(cur_obs)
            textual_obs = observer.get_observation_text()
            print(f"{textual_obs}")
            observer.info(f"{textual_obs}")
            evaluation = evaluator.evaluate(textual_obs, frame_desc)
            evaluator.info(f"\nStep {i+1}: \n {evaluation}")
            scores = evaluator.get_scores(evaluation)
            evaluator.log_step_scores(scores)

        observer.reset()
        trial_metrics = evaluator.get_trial_metrics()
        evaluator.info(f"\nTrial Metrics:\n\n {trial_metrics}")
        evaluator.close_logger()
        print(trial_metrics)
        overall_score = evaluator.get_overall_score()
        print(overall_score)


if __name__ == "__main__":
    main()
