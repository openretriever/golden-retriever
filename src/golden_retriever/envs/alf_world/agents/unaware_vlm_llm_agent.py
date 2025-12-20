import json
import os
from typing import Any, List

from pydantic import BaseModel

from agents.agent_utils.helper import VLM_MODEL_LIST, tensor_to_base64_image
from agents.agent_utils.inference_engine import engine_factory
from agents.base_agent import BaseAgent


class Response(BaseModel):
    think: str
    action: str

    def __str__(self) -> str:
        s = ""
        s += f"think:{self.think} "
        s += f"action:{self.action} "
        return s


FOLDER = "./prompt"
PROMPT_FILE = "alfworld_3prompts_revised.json"
with open(os.path.join(FOLDER, PROMPT_FILE), "r") as f:
    d = json.load(f)


class UnawareVLMAgent(BaseAgent):
    def __init__(self, llm_model, vlm_model, num_eval: int):
        super().__init__(num_eval=num_eval)
        assert vlm_model in VLM_MODEL_LIST, "Specified VLM model is not supported."
        self.llm_model = llm_model
        self.vlm_model = vlm_model
        self.llm_engine = engine_factory(model=llm_model)
        self.vlm_engine = engine_factory(model=vlm_model)
        self.name = "UnawareVLMAgent"
        self.logger = self.set_logger(name=self.name)

    def observe(self, observation) -> str:
        self.obs_memory.append(observation)
        # conversation = self.generate_query_for_vlm()
        # vlm_obs = self.vlm_engine.generate(conversation=conversation)

        # self.add_to_memory(label="observation", value=vlm_obs)
        # return vlm_obs

    def add_to_memory(self, label, value):
        assert label in ["observation", "response"], "label is incorrect."
        self.response_memory += [
            {
                "label": label,
                "value": value,
            }
        ]

    def init_prompts(self):
        self.prompts = {
            "vlm_prompt": (
                "You are a visual assistant helping an agent perceive their environment to complete household tasks. "
                "The environment consists of receptacles and objects, each uniquely named by combining their type and a numerical ID. For example:\n"
                '- "Countertop 1" refers to a receptacle of type "countertop" with the ID 1.\n'
                '- "Apple 2" refers to an object of type "apple" with the ID 2.\n\n'
                "This naming convention ensures all receptacles and objects are uniquely identifiable. You will receive images showing the agent's first-person view, with objects segmented and labeled by numbers. For instance:\n"
                '- "Object Bindings: 1--countertop 1, 3--apple 2" means that segment 1 corresponds to "countertop 1" and segment 3 corresponds to "apple 2."\n\n'
                'Always refer to objects and receptacles using their names (e.g., "countertop 1," "apple 2") rather than the segmentation numbers. Use only the names provided in the object bindings.\n\n'
                "Your tasks are as follows:\n"
                "1. **Identify the Held Item**: Think step by step to describe the object the agent is holding. The item held by the agent will always appear at the bottom center of the view and will appear closer to the agent than any other objects on the scene, it will appear as if it is floating in the air. \n"
                "2. **Describe Spatial Relationships**: Identify the spatial relationships between objects and receptacles in the environment.\n"
                "3. **Indicate Receptacle Status**: Note whether any of the following receptacles are open or closed: fridges, safes, drawers, cabinets, or microwaves. If such receptacles are not visible, do not comment on their status.\n\n"
                'Combine all observations into one concise sentence and address the agent directly in your response. Use "you" when describing observations. For example:\n'
                '- "you are holding an apple (2). on countertop 1, you see microwave 1. microwave 1 is closed."\n'
                'If there is nothing in the image, output "you see nothing."\n'
            ),
            "llm_prompt": "Interact with a household to solve a task. Here are two examples.\n"
            + d[f"react_{self.task_type}_1"]
            + d[f"react_{self.task_type}_0"],
        }

    def generate_query_for_vlm(self) -> List[Any]:
        cur_obs = self.obs_memory[-1]
        vlm_query = []
        vlm_query.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": self.prompts["vlm_prompt"]}],
            }
        )
        vlm_query.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{tensor_to_base64_image(cur_obs['image'])}"
                        },
                    },
                    {
                        "type": "text",
                        "text": cur_obs["object_bindings"]
                        + "Generate a concise observation based on the provided image and object bindings.",
                    },
                ],
            }
        )
        return vlm_query

    def generate_query_for_llm(self) -> str:
        llm_query = self.prompts["llm_prompt"] + "\n"
        llm_query += f"\nHere is the task:\n{self.task_info}"
        for i, item in enumerate(self.response_memory):
            if item["label"] == "response":
                llm_query += f'> {item["value"]}'
            elif item["label"] == "observation":
                llm_query += item["value"]
            if i != len(self.response_memory) - 1:
                llm_query += "\n"
        return llm_query

    def predict(self) -> str:
        obs = ""
        if len(self.obs_memory) > 0:
            vlm_obs_query = self.generate_query_for_vlm()
            obs = self.vlm_engine.generate(conversation=vlm_obs_query)
            self.add_to_memory(label="observation", value=obs)
        llm_query = self.generate_query_for_llm()
        messages = [{"role": "user", "content": llm_query + ">"}]
        # response = self.llm_engine.generate(conversation=messages, stop=['\n'])
        response = self.llm_engine.generate_format(
            conversation=messages, response_format=Response
        )
        response = obs + " " + response
        if response == self.last_response:
            self.is_exhausted = True
        self.last_response = response
        return response
