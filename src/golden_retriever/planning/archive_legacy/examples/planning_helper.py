import base64

import openai
import requests


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def format_and_simplify_plan(response_json):
    """
    Extracts the plan from the given response JSON and simplifies the format according to specified requirements.

    Parameters:
    - response_json (dict): The response JSON structure from which the plan is to be extracted.

    Returns:
    - list: A list of simplified plan steps.
    """
    try:
        # Extracting the plan content from the response JSON
        content = response_json["choices"][0]["message"]["content"]

        # Lists to store objects and plan steps
        objects_list = []
        plan_steps = []

        # Splitting content into lines
        lines = content.split("\n")

        # Flags to identify listing objects and generating plan
        is_listing_objects = False
        is_generating_plan = False

        # Iterating through each line
        for line in lines:
            if line.startswith("Blocks:") or line.startswith("Bowls:"):
                # Adding objects to the objects_list
                objects_list.append(line.strip().split(": ")[1])
            elif line.strip("*").startswith("START_PLAN"):
                # Setting flag to indicate plan generation has started
                is_generating_plan = True
            elif line.strip("*").startswith("END_PLAN"):
                # Setting flag to indicate plan generation has ended
                is_generating_plan = False
            elif is_generating_plan and line.startswith("-"):
                # Adding plan steps, excluding the 'done' step
                # NOTE: original version, for format "Step i: xxx"
                # step = line.split(": ")[1].strip().lower()
                # NOTE: format: "- pick xxx and place xxx"
                step = line[2:].strip().lower()
                if step.lower() != "done":
                    plan_steps.append(step)

        # Return simplified plan steps
        return plan_steps

    except Exception as e:
        print(f"An error occurred while processing the plan: {e}")
        return []


def simplify_plan_format(plan_steps):
    """
    Simplifies a list of plan steps by ensuring each step only mentions the color of the blocks being interacted with.

    Parameters:
    - plan_steps (list): A list of plan steps in a specific format.

    Returns:
    - list: A simplified list of plan steps.
    """
    simplified_plan = []
    for step in plan_steps:
        # Assume step format: 'pick up the [color block] and place it on the [color block].'
        parts = step.split(" and ")
        pick_up_part = parts[0]  # E.g., "pick up the [color block]"
        place_on_part = parts[1]

        # Simplify the description by removing unnecessary details and focusing on the color only
        pick_up_color = pick_up_part.split("[")[1].split(" ")[0]  # Extracts color
        place_on_color = place_on_part.split("[")[1].split(" ")[0]  # Extracts color

        simplified_step = f"pick up the {pick_up_color} block and place it on the {place_on_color} block."
        simplified_plan.append(simplified_step)

    return simplified_plan


def process_image_and_question(
    prompt_text, prompt1, prompt2, prompt3, image_path, question
):
    # Path to your image
    prompt1img = encode_image(prompt1)
    prompt2img = encode_image(prompt2)
    prompt3img = encode_image(prompt3)

    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai.api_key}",
    }

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """You excel at counting and identifying colors and objects in images, as well as strategizing for robotic tabletop rearrangement tasks.
                                    I will provide three examples with corresponding tasks and robot action plans. Please first itemize all objects in each image and then
                                    detail the plan. Ensure you use the command "pick up the [color] block and place it on the [place]", and you can only pick up one block
                                    at a time, not multiple stacked blocks. Stick to the color palette: ['blue', 'red', 'green', 'yellow', 'brown', 'cyan', 'orange', 'purple', 'pink',
                                    'white']. (Please note that your plan can only contain the instructions of each step, can not have any superfluous explanations with
                                    notes and parentheses)""",
                    },
                    {
                        # TODO add prompt; may need to merge with original one
                        "type": "text",
                        "text": prompt_text,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{prompt1img}"},
                    },
                    {
                        "type": "text",
                        "text": """This is the first example.
                                    Task: Stack all the blocks
                                    Blocks: red block, cyan block, orange block, pink block, brown block
                                    Bowls: cyan bowl, red bowl
                                    START_PLAN
                                    - pick up the brown block and place it on the pink block
                                    - pick up the cyan block and place it on the brown block
                                    - pick up the orange block and place it on the cyan block
                                    - pick up the red block and place it on the orange block
                                    END_PLAN""",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{prompt2img}"},
                    },
                    {
                        "type": "text",
                        "text": """This is the second example.
                                    Task: Put all the blocks on the bottom left corner
                                    Blocks: blue block, purple block, green block, yellow block, white block
                                    Bowls: yellow bowl, cyan bowl
                                    START_PLAN
                                    - pick up the white block and place it on the bottom left corner
                                    - pick up the yellow block and place it on the bottom left corner
                                    - pick up the green block and place it on the bottom left corner
                                    - pick up the blue block and place it on the bottom left corner
                                    - pick up the purple block and place it on the bottom left corner
                                    END_PLAN""",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{prompt3img}"},
                    },
                    {
                        "type": "text",
                        "text": """This is the third example.
                                Task: Put all the blocks on the bowls with matching colors
                                Blocks: purple block, pink block, brown block, orange block, red block, cyan block, white block
                                Bowls: purple bowl, brown bowl, pink bowl, cyan bowl
                                START_PLAN
                                - pick up the cyan block and place it on the cyan bowl
                                - pick up the purple block and place it on the purple bowl
                                - pick up the brown block and place it on the brown bowl
                                - pick up the pink block and place it on the pink bowl
                                END_PLAN
                                """,
                    },
                    {
                        "type": "text",
                        "text": f"""
                        There is a new task, your job is to solve this task.
                        Complete prompt that contains useful information will be provided afterwards.

                        Task: {question}
                        Current observation:
                        """,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        "max_tokens": 600,
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )

    return response.json()


if __name__ == "__main__":
    # Example usage:
    response_json_1 = {
        "choices": [
            {
                "message": {
                    "content": "The image provided contains the following objects:\n\n"
                    "Blocks: blue block, yellow block, green block, red block\n"
                    "Bowls: brown bowl, white bowl\n\n"
                    "Given Task: Stack all the blocks\n"
                    "START_PLAN\n"
                    "Step 1: pick up the green block and place it on the yellow block\n"
                    "Step 2: pick up the blue block and place it on the green block\n"
                    "Step 3: pick up the red block and place it on the blue block\n"
                    "Step 4: done\n"
                    "END_PLAN"
                }
            }
        ]
    }

    response_json_2 = {
        "choices": [
            {
                "message": {
                    "content": "Blocks: blue block, red block, yellow block\n"
                    "Bowls: brown bowl, white bowl\n\n"
                    "START_PLAN\n"
                    "Step 1: pick up the blue block and place it on the red block\n"
                    "Step 2: pick up the yellow block and place it on the blue block\n"
                    "Step 3: done\n"
                    "END_PLAN"
                }
            }
        ]
    }

    print(format_and_simplify_plan(response_json_1))
    print(format_and_simplify_plan(response_json_2))
