"""
This agent is used to collect trajectory level data for evaluation of perception.
"""
import json
import os
from typing import Any, Dict, List

from agents.agent_utils.inference_engine import engine_factory

PREFIXES = {
    "pick_and_place": "put",
    "pick_clean_then_place": "clean",
    "pick_heat_then_place": "heat",
    "pick_cool_then_place": "cool",
    "look_at_obj": "examine",
    "pick_two_obj": "puttwo",
}
FOLDER = "./prompt"
PROMPT_FILE = "alfworld_3prompts_revised.json"
with open(os.path.join(FOLDER, PROMPT_FILE), "r") as f:
    d = json.load(f)


class Planner:
    def __init__(self, *args):
        pass

    def add_to_memory(self, *args):
        pass

    def get_task_info(self, *args):
        pass

    def init_prompts(self):
        pass

    def predict(self):
        pass

    def reset(self):
        pass


class ReActPlanner(Planner):
    """
    ReAct agent that on each step, takes in the textual feedback from the ALFWorld, and outputs thought and action.
    """

    def __init__(self, llm_model):
        self.llm_model = llm_model
        self.llm_engine = engine_factory(llm_model)
        self.response_memory: List[Any] = []

    def add_to_memory(self, label: str, value: str | Dict):
        assert label in ["observation", "response"], "Invalid label"
        self.response_memory.append(
            {
                "label": label,
                "value": value,
            }
        )

    def get_task_info(self, task_name: str, task_info: str):
        self.task_name = task_name
        self.task_info = task_info
        if self.task_name.startswith("find"):
            self.task_type = "find"
        else:
            for _, (k, v) in enumerate(PREFIXES.items()):
                if self.task_name.startswith(k):
                    self.task_type = v

        self.init_prompts()

    def init_prompts(self):
        self.prompts = {
            "llm_system_prompt": (
                "You are an household agent designed to interact with a simulated household environment to solve household tasks step by step. "
                "In this environment, you can interact with objects and receptacles to solve the task."
                "After you execute an action, you will receive a textual feedback from the environment."
            ),
            "llm_think_prompt": (
                "Let us think step by step to generate your thoughts of what plan could lead to task accomplish based on the observation and past history. "
                'Your response should start with "think: "'
                "For example:\n"
                "think: Now I have taken apple 1, the task is to put apple 1 on countertop 1, I need to go to countertop 1 and then put in/on countertop to accomplish the task.\n"
            ),
            "llm_action_prompt": (
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
                '- Start your response with "action:".\n'
                "- Example: action: take apple 1 from countertop 2\n\n"
                "Additional Rules:\n"
                "- Generate only one action per response.\n"
                "- Avoid repeating the same action as the previous step.\n"
            ),
            "llm_two_shot_prompt": ""
            if not (
                f"react_{self.task_type}_0" in d and f"react_{self.task_type}_1" in d
            )
            else "Here are two examples.\n"
            + d[f"react_{self.task_type}_0"]
            + d[f"react_{self.task_type}_1"],
        }

    def generate_think_prompt_for_llm(self):
        prompts = (
            self.prompts["llm_system_prompt"] + self.prompts["llm_two_shot_prompt"]
        )
        llm_query = [
            {"role": "system", "content": prompts},
        ]
        history = self.get_history()
        llm_query.append({"role": "user", "content": history})
        llm_query.append({"role": "user", "content": self.prompts["llm_think_prompt"]})

        return llm_query

    def generate_action_prompt_for_llm(self):
        prompts = (
            self.prompts["llm_system_prompt"] + self.prompts["llm_two_shot_prompt"]
        )
        llm_query = [
            {"role": "system", "content": prompts},
        ]
        history = self.get_history()
        llm_query.append({"role": "user", "content": history})
        llm_query.append({"role": "user", "content": self.prompts["llm_action_prompt"]})
        return llm_query

    def get_history(self):
        history = ""
        history += f"\nHere is the task:\n{self.task_info}"
        # generate history text using response_memory
        for i in range(len(self.response_memory)):
            history += self.response_memory[i]["value"]

        return history

    def predict(self):
        llm_think_query = self.generate_think_prompt_for_llm()
        llm_action_query = self.generate_action_prompt_for_llm()

        llm_think_response = self.llm_engine.generate(llm_think_query)
        llm_action_response = self.llm_engine.generate(llm_action_query)
        response = llm_think_response + llm_action_response

        self.add_to_memory("response", response)
        action = self.process_action(llm_action_response)
        return action

    def process_action(self, response: str):
        """
        Processes the agent's response to an action.
        """
        parts = response.lower().split("action:")
        if len(parts) > 1:
            # Get the substring after 'Action:' and strip any leading/trailing whitespace
            action = parts[1].strip()
        else:
            action = " "

        return action

    def reset(self):
        self.response_memory = []
        self.task_name = None
        self.task_info = None
        self.task_type = None
