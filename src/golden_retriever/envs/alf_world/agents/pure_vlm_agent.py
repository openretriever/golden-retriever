import json
import os
from typing import Any, Dict, List

from pydantic import BaseModel

from agents.agent_utils.helper import VLM_MODEL_LIST, tensor_to_base64_image
from agents.agent_utils.inference_engine import engine_factory
from agents.base_agent import BaseAgent

FOLDER = "./prompt"
PROMPT_FILE = "pure_vlm_prompts.json"
with open(os.path.join(FOLDER, PROMPT_FILE), "r") as f:
    d = json.load(f)


class Response(BaseModel):
    observation: str
    think: str
    action: str

    def __str__(self) -> str:
        s = ""
        s += f"observation:{self.observation} "
        s += f"think:{self.think} "
        s += f"action:{self.action} "
        return s


class PureVLMAgent(BaseAgent):
    def __init__(self, vlm_model, num_eval: int):
        """
        Initializes the Agent.

        """
        super().__init__(num_eval=num_eval)
        assert vlm_model in VLM_MODEL_LIST, "Specified VLM model is not supported."

        self.vlm_model = vlm_model
        self.vlm_engine = engine_factory(model=vlm_model)
        self.name = "PureVLMAgent"
        self.logger = self.set_logger(name="PureVLMAgent")

    def observe(self, observation):
        self.obs_memory.append(observation)

    def add_to_memory(self, label: str, value: str | Dict):
        """
        Adds a message to the agent's memory.
        Child classes must implement how to store this message.

        Args:
            message (Any): The message to store in memory.
        """
        assert label in ["response"], "label is incorrect."
        self.response_memory += [
            {
                "label": label,
                "value": value,
            }
        ]

    def init_prompts(self):
        """
        Generates a prompt based on the agent's memory and user input.
        Child classes must implement how the prompt is created.

        Args:
            user_input (str): The latest input from the user.
            context_length (int): The number of recent memory entries to include in the prompt.

        Returns:
            str: The generated prompt.
        """
        self.prompts = {
            "vlm_sys_prompt": (
                "You are an household agent designed to interact with a simulated household environment to solve household tasks step by step. "
                "In this environment, you can interact with objects and receptacles to solve the task."
                "At each stage, you can see the simulated househould environment by a first-person segmented image and know the previous observations, thoughts and actions of yours."
                "Your goal is to complete tasks by analyzing segmented, labeled first-person images of the environment and generate observation, thoughts and actions you will take to accomplish the task based on past observations, thoughts, actions and task information.\n\n"
                "You will stop receiving observations from the environment once the task is finished, if you think you have already completed the task, but still receive observations from the environment, you haven't completed the task."
                "Instructions:\n\n"
                "1. **Task Goals**: Each task includes a clear objective and a list of relevant objects and receptacles.\n\n"
                "2. **Segmented Visual Input**: "
                "You will receive images showing a first-person view of the environment, with objects and receptacles segmented and labeled by numbers. "
                "These labels correspond to names in the object binding information, such as '3--countertop 2' or '4--sinkbasin 1.' Use these labels to identify objects.\n\n"
                "    - Objects and receptacles in the environment are uniquely identified by combining their type and a numerical ID. Examples:\n"
                "      - 'countertop 1' refers to a receptacle of type 'countertop' with ID 1.\n"
                "      - 'apple 2' refers to an object of type 'apple' with ID 2.\n\n"
                "    - Object bindings will be presented as mappings, for example:\n"
                "      - 'Object Bindings: 1--countertop 1, 3--apple 2' means that segment 1 corresponds to 'countertop 1' and segment 3 corresponds to 'apple 2.'\n\n"
                "    Always refer to objects and receptacles by their full names (e.g., 'countertop 1,' 'apple 2'), avoiding segmentation numbers in your descriptions.\n\n"
                "Here are two examples.\n"
                + d["seereact_" + f"{self.task_type}_0"]
                + d["seereact_" + f"{self.task_type}_1"]
            ),
            "vlm_obs_prompt": (
                'Always refer to objects and receptacles by their exact names (e.g., "countertop 1," "apple 2") from the object bindings, avoiding segmentation numbers. '
                "You will generate textual description centered around task-relevant objects and receptacles.\n\n"
                "Hence you should infer what are task-relevant objects, you should infer task-relevant objects and receptacles based on:\n"
                "- Visual observations\n"
                "- Agent's thought and action history\n"
                "- Task objectives\n\n"
                "Your responsibilities include:\n"
                "1. **Identify the Held Item**: Determine and describe the object the agent is holding. The held item will:\n"
                "- Always appear at the bottom center of the view, closer to you than other objects, and floating in the air.\n"
                '- Be held only if previous actions include "take [object] from [receptacle]."\n'
                '- No longer be held if the previous action includes "put [object] in/on [receptacle]."\n'
                "Think step by step to verify these conditions before concluding whether the agent is holding an object.\n\n"
                "2. **Previous Action Effect**: Explain the impact of the agent's most recent action on the environment or objects. You should respond according to previous action by the following way:\n"
                "- If your previous action is go to [recep id], you should respond if you have arrived at [recep id] based on current observation.\n"
                "- If your previous action is take [obj id] from [recep id], you should respond if you pick up or take [obj id] based on current observation.\n"
                "- If your previous action is put [obj id] in/on [recep id], you should respond if you put [obj id] in/on [recep id] based on current observation.\n"
                "- If your previous action is open [recep id], you should respond if [recep id] is opened based on current observation.\n"
                "- If your previous action is close [recep id], you should respond if [recep id] is closed based on current observation.\n"
                "- If your previous action is use [obj id], you should respond if [obj id] is toggled on based on current observation. For example, if use desklamp, desklamp will be on.\n"
                '- If your previous action is heat/cool/clean [obj id] with [recep id], respond as if these actions are all successful. For example, if previous action is "heat apple 1 with microwave 1", you should respond "I heat the apple 1 using microwave 1."\n'
                "3. **Describe Spatial Relationships**: Identify and describe spatial relationships between task-relevant objects and receptacles in the environment. It is possible that you could see task-relevant objects in your current observations but not in/on current receptacle that you are interacting with.\n\n"
                "4. **Indicate Receptacle Status**: Specify the status (opened or closed) of task-relevant receptacles such as fridges, safes, drawers, cabinets, or microwaves. If these receptacles are not visible, do not comment on their status.\n\n"
                'Combine all observations into a single concise sentence, addressing the agent directly using "I.", generate your reponse starting with "Observation: " For example:\n'
                '- "Observation: I am holding an apple (2). I have arrived at countertop 1. On countertop 1, I see microwave 1. Microwave 1 is closed."\n'
                '- "Observation: I am holding saltshaker 1. I have arrived at safe 1. Safe 1 is closed."\n'
                'If no objects or receptacles are visible, output: "Observation: I see nothing."\n'
                "Following is the first-person segmented image along with object bindings information, generate your response carefully according to the guidance above.\n\n"
            ),
            "vlm_think_prompt": (
                "Let us think step by step to generate your thoughts of what plan could lead to task accomplish based on the observation and past history. "
                "If you think you have completed the task but still receiving observations, you haven't completed the task, reflect on you history and think step by step what could go wrong. "
                'Your response should start with "Think: "'
                "For example:\n"
                "Think: Now I have taken apple 1, the task is to put apple 1 on countertop 1, I need to go to countertop 1 and then put in/on countertop to accomplish the task.\n"
            ),
            "vlm_action_prompt": (
                "Specify the next action the agent should take to progress toward the task goal, following these guidelines:\n\n"
                '1. Think Step-by-Step: Reflect on the previous thoughts ("Think:") and determine the action that logically advances the task.\n'
                "2. Object and Receptacle References: Use specific identifiers:\n"
                "   - [obj id] for objects (e.g., apple 1).\n"
                "   - [recep id] for receptacles (e.g., countertop 1).\n"
                "3. Action Validity: Follow the exact format below. Any deviation renders the action invalid:\n\n"
                "Action Formats:\n"
                "- go to [recep id]\n"
                "- take [obj id] from [recep id]\n"
                "- put [obj id] in/on [recep id]\n"
                "- open [recep id]\n"
                "- close [recep id]\n"
                "- use [obj id]\n"
                "- heat [obj id] with [recep id]\n"
                "- cool [obj id] with [recep id]\n"
                "- clean [obj id] with [recep id]\n\n"
                "Response Format:\n"
                '- Start your response with "Action:".\n'
                "- Example: Action: take apple 1 from countertop 2\n\n"
                "Additional Rules:\n"
                "- Generate only one action per response.\n"
                "- Avoid repeating the same action as the previous step.\n"
            ),
        }

    def generate_obs_query_for_vlm(self) -> List[Any]:
        vlm_query = []
        vlm_query.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": self.prompts["vlm_sys_prompt"]}],
            }
        )
        s = f"\nHere is the task and agent's history of observations, thoughts and actions:\n{self.task_info}\n"
        for i, item in enumerate(self.response_memory):
            if item["label"] == "response":
                s += f'{item["value"]}'
            if i != len(self.response_memory) - 1:
                s += "\n"
        vlm_query.append({"role": "user", "content": [{"type": "text", "text": s}]})
        if len(self.obs_memory) > 0:
            cur_obs = self.obs_memory[-1]
            vlm_query.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompts["vlm_obs_prompt"]},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{tensor_to_base64_image(cur_obs['image'])}"
                            },
                        },
                        {"type": "text", "text": cur_obs["object_bindings"]},
                    ],
                }
            )
        return vlm_query

    def generate_think_query_for_vlm(self, gen_obs: str) -> List[Any]:
        vlm_query = []
        vlm_query.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": self.prompts["vlm_sys_prompt"]}],
            }
        )
        s = f"\nHere is the task and agent's history of observations, thoughts and actions:\n{self.task_info}\n"
        for i, item in enumerate(self.response_memory):
            if item["label"] == "response":
                s += f'{item["value"]}'
            if i != len(self.response_memory) - 1:
                s += "\n"
        vlm_query.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": s + gen_obs},
                    {"type": "text", "text": self.prompts["vlm_think_prompt"]},
                ],
            }
        )
        if len(self.obs_memory) > 0:
            cur_obs = self.obs_memory[-1]
            vlm_query.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Following is the first-person segmented image along with object bindings information, generate your response carefully\n",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{tensor_to_base64_image(cur_obs['image'])}"
                            },
                        },
                        {"type": "text", "text": cur_obs["object_bindings"]},
                    ],
                }
            )
        return vlm_query

    def generate_action_query_for_vlm(self, gen_obs: str, gen_think: str) -> List[Any]:
        vlm_query = []
        vlm_query.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": self.prompts["vlm_sys_prompt"]}],
            }
        )
        s = f"\nHere is the task and agent's history of observations, thoughts and actions:\n{self.task_info}\n"
        for i, item in enumerate(self.response_memory):
            if item["label"] == "response":
                s += f'{item["value"]}'
            if i != len(self.response_memory) - 1:
                s += "\n"
        vlm_query.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": s + gen_obs + gen_think},
                    {"type": "text", "text": self.prompts["vlm_action_prompt"]},
                ],
            }
        )

        return vlm_query

    def predict(self) -> str:
        obs = ""
        if len(self.obs_memory) > 0:
            vlm_obs_query = self.generate_obs_query_for_vlm()
            obs = self.vlm_engine.generate(conversation=vlm_obs_query)

        vlm_think_query = self.generate_think_query_for_vlm(gen_obs=obs)
        think = self.vlm_engine.generate(
            conversation=vlm_think_query, stop=["Action:", "\n", "\nAction:"]
        )

        vlm_action_query = self.generate_action_query_for_vlm(
            gen_obs=obs, gen_think=think
        )
        action = self.vlm_engine.generate(conversation=vlm_action_query)
        response = obs + " " + think + " " + action
        if response == self.last_response:
            self.is_exhausted = True
        self.last_response = response

        return response
