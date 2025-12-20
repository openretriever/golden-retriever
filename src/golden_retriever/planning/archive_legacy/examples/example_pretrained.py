import os
import pathlib
import random

import numpy as np
import openai
import ray
import torch
from PIL import Image

from retriever.envs.ravens import tasks
from retriever.envs.ravens.envs.environment import Environment
from retriever.models import utils
from retriever.models.segmentation.langsam_actor import LangSAM
from retriever.planning.examples.planning_helper import (
    format_and_simplify_plan,
    process_image_and_question,
)

openai.api_key = os.getenv("OPENAI_API_KEY")
assert openai.api_key.startswith("sk-"), "OpenAI API key should start with 'sk-'"

# Variable to control GPU usage
# Set to True to use GPU, False to not use GPU
use_gpu = torch.cuda.is_available()
num_gpus = torch.cuda.device_count()
print(f"Number of GPUs: {num_gpus}")

# Note: Replace with your Ray cluster's address
ray.init(num_gpus=0 if not use_gpu else num_gpus)
print(ray.available_resources())

# Options dictionary to dynamically set num_gpus
actor_options = {"num_gpus": 0.1} if use_gpu else {}

# Create an actor instance with dynamic GPU allocation
# clip_actor = CLIPActor.options(**actor_options).remote(use_gpu=use_gpu)
# owl_actor = OWLV2Actor.options(**actor_options).remote(use_gpu=use_gpu)
langsam_actor = LangSAM.options(**actor_options).remote(use_gpu=use_gpu)

save_video = True
root_dir = pathlib.Path.cwd().parent.parent
assets_root = os.path.join(root_dir, "envs/ravens/envs/assets/")

# Initialize the environment
record_cfg = {
    "save_video": save_video,
    "save_video_path": "./tmp/images",
    "add_text": True,
    "fps": 20,
    "video_height": 640,
    "video_width": 720,
}
env = Environment(
    assets_root, disp=True, shared_memory=False, hz=480, record_cfg=record_cfg
)

# Initialize task
task_list = [
    # from ViLa: Blocks & Bowls
    "stack-blocks",
    "put-blocks-on-corner-side",
    "put-blocks-matching-colors",
    "put-blocks-mismatched-colors",
    "put-blocks-different-corners",
    "stack-blocks-cool-colors",
    "stack-blocks-warm-colors",
    "sort-primary-color-blocks",
    # from ViLa: Letters tasks
    "put-letters-alphabetical-order",
    "spell-word",
    "separate-vowels",
    "put-letters-reverse-alphabetical-order",
    "spell-sport",
    "sort-symmetrical-letters",
    "separate-consonants",
    "sort-letters-less-than-d",
]
# task_name = "stack-blocks"
task_name = task_list[0]
task = tasks.names[task_name]()
task.mode = "test"
print(f"Task Name: {task_name}")
print(f"Task Lang Goal: {task.get_lang_goal()}")


# Initialize scripted oracle agents
agent = task.step_oracle(env)
# FIXME replace with pretrained version, use trained CLIPort


n_eval = 3
seed = 8888
success_times = 0

for i in range(n_eval):
    print(f"\nEvaluation Instance: {i + 1}/{n_eval}")

    # Set seeds.
    seed += 2
    np.random.seed(seed)
    random.seed(seed)
    env.seed(seed)

    env.set_task(task)
    obs = env.reset()
    success = False

    task_goal = env.info["high_level_lang_goal"]
    print(f"High Level Goal: {task_goal}")

    front_obs = obs["color"][0]  # front camera, 480 x 640 x 3
    front_obs_img = Image.fromarray(front_obs)
    # front_obs_bgr = cv2.cvtColor(front_obs, cv2.COLOR_RGB2BGR)
    utils.display_image(front_obs_img, "Front Observation")
    utils.save_image(front_obs_img, "visualization-front-obs.png")

    topdown_obs = obs["color"][3]
    topdown_obs_img = Image.fromarray(topdown_obs)
    # topdown_obs_bgr = cv2.cvtColor(topdown_obs, cv2.COLOR_RGB2BGR)
    utils.display_image(topdown_obs_img, "Topdown Observation")
    utils.save_image(topdown_obs_img, "visualization-topdown-obs.png")

    # print(process_image_and_question("image_langsam.png",question=complete_prompt))
    # formateplan =format_plan_from_response(process_image_and_question("image_langsam.png",question=complete_prompt))
    # print(formateplan)
    # cv2.imshow(f'Observation Image - Instance {i}', front_obs_bgr )
    # cv2.waitKey(0)  # Wait for a key press to close the window

    # TODO test using LangSAM - next: use masked image as input
    text_prompt = "blocks"
    # masks, boxes, phrases, logits = ray.get(
    #     langsam_actor.predict.remote(image_pil, text_prompt)
    # )
    # ray.get(langsam_actor.save.remote(masks, boxes, phrases, logits, image_pil))

    print(f"Observation color 0 shape: {obs['color'][0].shape}")

    keys_list = list(env.info["blockbowl_affordance"].keys())
    block_keys = [key for key in keys_list if "block" in key]
    bowl_keys = [key for key in keys_list if "bowl" in key]
    print(f"Affordance key list: {keys_list}")

    # Generate the keys_description string with only the filtered keys
    block_keys_desc = ", ".join(block_keys)
    bowl_keys_desc = ", ".join(bowl_keys)

    question_prompt = f"""
    Given the task described, first, observe the environment where specific objects, are placed within boxes for clearer identification. Using the provided color definitions, accurately identify the colors of these objects based on their keys. Then, list all available objects in the environment, including their colors, ensuring accuracy in color identification by considering the color definitions and the visual cues from the boxes. Following this, generate a plan using the specified skills, ensuring the plan utilizes only the listed objects and explicitly excludes any interactions with the "table" object. The plan must strictly follow the provided template.

    Task Description: {task_goal}

    Current Environment State:
    - The environment contains objects placed within boxes, which should aid in clearer identification.
    - Objects and their colors are defined as follows (please adjust based on actual environment observation and the color definitions provided):
        'blue': [078.0 / 255.0, 121.0 / 255.0, 167.0 / 255.0],
        'red': [255.0 / 255.0, 087.0 / 255.0, 089.0 / 255.0],
        'green': [089.0 / 255.0, 169.0 / 255.0, 079.0 / 255.0],
        'orange': [242.0 / 255.0, 142.0 / 255.0, 043.0 / 255.0],
        'yellow': [237.0 / 255.0, 201.0 / 255.0, 072.0 / 255.0],
        'purple': [176.0 / 255.0, 122.0 / 255.0, 161.0 / 255.0],
        'pink': [255.0 / 255.0, 157.0 / 255.0, 167.0 / 255.0],
        'cyan': [118.0 / 255.0, 183.0 / 255.0, 178.0 / 255.0],
        'brown': [156.0 / 255.0, 117.0 / 255.0, 095.0 / 255.0],
        'white': [255.0 / 255.0, 255.0 / 255.0, 255.0 / 255.0],
        'gray': [186.0 / 255.0, 176.0 / 255.0, 172.0 / 255.0],

    Objective: To complete the task using only two skills: [pick up] and [place it], without involving the "table" in any of the steps.

    Instructions:
    1. Carefully observe the objects within boxes and use the color definitions to accurately identify and list all available objects along with their corresponding colors. The available keys for identification include: {block_keys_desc}.
    2. With the accurately identified objects and colors, generate a step-by-step plan that follows the specified format and excludes any interaction with the "table".

    Plan Generation Format:
    - Begin with a clear identification of the objects and colors, stating "Available objects are: "color objectname","color objectname""
    - Use the identified objects to construct the plan. For each step, use square brackets and state "pick up the [color/object]" followed by "and place it on the [color/object]."
    - Ensure no steps involve placing objects on the "table".
    - Keep descriptions concise and adhere to the guidelines provided.

    Ensure the plan:
    - Is based on the accurately identified objects and colors.
    - Follows the specified format and instructions.
    - Does not involve interaction with the "table".
    This structured approach is crucial for the successful execution of the task.

    Begin List of Objects (You must only choose to operate blocks in these colors):
    Blocks: {block_keys_desc}
    Bowls: {bowl_keys_desc}

    """

    # # Placeholder for the end of object listing
    # end_of_object_list = """
    #
    # START_PLAN
    #
    # """
    #
    # # Placeholder for the end of the plan
    # end_of_plan = """
    # END_PLAN
    # """

    # Combine the parts to form the complete prompt
    # complete_prompt = question_prompt + end_of_object_list + end_of_plan
    complete_prompt = question_prompt
    # TODO this prompt was not even used

    # Predict masks and scores
    response = process_image_and_question(
        complete_prompt,  # TODO add prompt
        "prompt1.png",
        "prompt2.png",
        "prompt3.png",
        "visualization-topdown-obs.png",
        question=task_goal,
    )
    print(f"Response: {response['choices'][0]['message']['content']}")

    task_lang_plan = format_and_simplify_plan(response)
    print(f"Formatted GPT-4V Plan: {task_lang_plan}")

    if save_video:
        env.start_rec(f"GPT-4V-{task_name}-seed{seed}")

    try:
        for j in range(len(task_lang_plan)):
            # TODO add replanning with current observation
            mid_level_instruction = task_lang_plan[j]
            act = agent.act(obs, mid_level_instruction)
            obs, _, _, info = env.step(act)
            success = info["success"]
            print(f"GPT-4V plan: {mid_level_instruction} | Success: {success}")
            if success:
                success_times += 1

            if save_video:
                env.end_rec()

    except Exception as e:
        # print(Exception.args)
        print(f"Error: {e}")

print(f"\nSuccess Rate: {success_times / n_eval:.3f} ({success_times}/{n_eval})")
