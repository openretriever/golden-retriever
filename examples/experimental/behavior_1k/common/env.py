import logging
from typing import Optional, Any
import dataclasses
import numpy as np

from retriever.flow import Flow, io

# import omnigibson as og  # type: ignore


@io
class Observation:
    rgb: Any
    depth: Any
    proprio: Any
    instruction: str


@io
class Action:
    delta_pose: Any
    gripper: float


class OmniGibsonEnv(Flow[Optional[Action], Observation]):
    """
    Wraps the BEHAVIOR-1K (OmniGibson) environment as a Retriever Flow.
    Acts as the source of truth for time/state (Output: Observation).
    """

    def __init__(self, task_name: str, headless: bool = True):
        self.task_name = task_name
        self.headless = headless
        self.env = None
        self.step_count = 0

    def _setup_env(self):
        logging.info(f"Setting up OmniGibson environment for task: {self.task_name}")
        try:
            # import omnigibson as og
            # cfg = { ... }
            # self.env = og.Environment(configs=cfg)
            pass
        except ImportError:
            logging.warning("OmniGibson not found, using mock environment.")

    def run(self, action: Optional[Action]) -> Observation:
        if self.env is None:
            self._setup_env()

        # Simulate step
        if action is not None:
            # self.env.step(action.delta_pose) # Simplified
            pass

        self.step_count += 1

        # MOCK OBSERVATION matching B1KPolicyWrapper expectations
        # Keys from eval_b1k_wrapper.py process_obs expects:
        # - robot_r1::proprio (16,)
        # - robot_r1::robot_r1:zed_link:Camera:0::rgb (H, W, 3)
        # - robot_r1::robot_r1:left_realsense_link:Camera:0::rgb (H, W, 3)
        # - robot_r1::robot_r1:right_realsense_link:Camera:0::rgb (H, W, 3)

        return Observation(
            rgb={
                "robot_r1::robot_r1:zed_link:Camera:0::rgb": np.zeros(
                    (224, 224, 3), dtype=np.uint8
                ),
                "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb": np.zeros(
                    (224, 224, 3), dtype=np.uint8
                ),
                "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb": np.zeros(
                    (224, 224, 3), dtype=np.uint8
                ),
            },
            depth={},
            proprio=np.zeros(16, dtype=np.float64),
            instruction=f"Task: {self.task_name}",
        )
