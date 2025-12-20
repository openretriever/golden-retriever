import logging
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, List

import numpy as np

PREFIXES = {
    "pick_and_place": "put",
    "pick_clean_then_place": "clean",
    "pick_heat_then_place": "heat",
    "pick_cool_then_place": "cool",
    "look_at_obj": "examine",
    "pick_two_obj": "puttwo",
}


class BaseAgent(ABC):
    """
    Abstract Base Class for an agent with memory and prompt generation functionality.
    """

    def __init__(self, num_eval: int):
        """
        Initializes the BaseAgent with a name and memory.
        """
        self.prompts: str = ""
        self.task_info: str = ""
        self.response_memory: List[Any] = []
        self.obs_memory: List[Any] = []
        self.is_exhausted: bool = False
        self.last_response: str = ""
        self.num_eval = num_eval
        self.set_metrics()
        self.metric_mapping = {
            "pick_and_place": "episode_succ_rate_pick_and_place",
            "pick_two_obj_and_place": "episode_succ_rate_pick_two_obj_and_place",
            "look_at_obj_in_light": "episode_succ_rate_look_at_obj_in_light",
            "pick_heat_then_place_in_recep": "episode_succ_rate_pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep": "episode_succ_rate_pick_cool_then_place_in_recep",
            "pick_clean_then_place_in_recep": "episode_succ_rate_pick_clean_then_place_in_recep",
        }

    def reset(self):
        self.clear_memory()
        self.is_exhausted: bool = False
        self.last_response: str = ""

    def get_task_info(self, task_name: str, task_info: str):
        self.task_name = task_name
        self.task_info = task_info
        for _, (k, v) in enumerate(PREFIXES.items()):
            if self.task_name.startswith(k):
                self.task_type = v
        self.init_prompts()

    @abstractmethod
    def observe(self):
        """ """
        pass

    @abstractmethod
    def add_to_memory(self):
        """
        Adds a message to the agent's memory.
        Child classes must implement how to store this message.

        Args:
            message (Any): The message to store in memory.
        """
        pass

    def get_memory(self):
        """
        Retrieves the agent's memory.
        """
        return self.obs_memory, self.response_memory

    def clear_memory(self):
        """
        Clears the agent's memory.
        """
        self.obs_memory = []
        self.response_memory = []

    @abstractmethod
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
        pass

    @abstractmethod
    def predict(self):
        pass

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

    def set_logger(self, name):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        # create a stream handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # create a logging format
        console_formatter = logging.Formatter("%(message)s")

        # add the formatter to the handler
        ch.setFormatter(console_formatter)

        # add the handler to the logger
        logger.addHandler(ch)

        return logger

    def set_metrics(self):
        # Metrics tracking
        self.metrics = {
            "episode_success_rate": deque(maxlen=self.num_eval),
            "episode_gc_success_rate": deque(maxlen=self.num_eval),
            "episode_succ_rate_pick_and_place": deque(maxlen=self.num_eval),
            "episode_succ_rate_pick_two_obj_and_place": deque(maxlen=self.num_eval),
            "episode_succ_rate_look_at_obj_in_light": deque(maxlen=self.num_eval),
            "episode_succ_rate_pick_heat_then_place_in_recep": deque(
                maxlen=self.num_eval
            ),
            "episode_succ_rate_pick_cool_then_place_in_recep": deque(
                maxlen=self.num_eval
            ),
            "episode_succ_rate_pick_clean_then_place_in_recep": deque(
                maxlen=self.num_eval
            ),
        }

    def track_metrics(self, infos):
        if not infos["won"][0]:
            self.logger.info("Task failed.")
        if infos["extra.gamefile"][0] is not None:
            for key, metric_name in self.metric_mapping.items():
                if key in infos["extra.gamefile"][0]:
                    self.metrics[metric_name].append(float(infos["won"][0]))
                    break
            self.metrics["episode_success_rate"].append(float(infos["won"][0]))
            self.metrics["episode_gc_success_rate"].append(
                float(infos["goal_condition_success_rate"][0])
            )

    def get_metrics_summary(self, iteration):
        summary = {}
        summary["iteration"] = iteration
        for key, value in self.metrics.items():
            summary[f"{key}.mean"] = np.mean(value)

        return summary
