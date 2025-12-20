"This script is used to evaluate the performance of the perception module in the ALF World environment."
import json
import logging
import os

from agents.agent_utils.inference_engine import engine_factory
from agents.agent_utils.prompts import generate_prompts_for_evaluator
from pydantic import BaseModel


class FinalScore(BaseModel):
    # three scores corresponding to three aspects:
    # 2. correctness of identifying the spatial relationships between the objects and receptacles
    # 3. correctness of identifying the statuses of the receptacles
    spatial_relationship_score: float
    receptacle_status_score: float


class Evaluator:
    def __init__(self, llm_model):
        self.llm_model = llm_model
        self.engine = engine_factory(llm_model)
        # Dictionary to store trial metrics
        self.trial_metrics = {}
        # Current trial being evaluated
        self.current_trial = None

    def evaluate(self, vlm_obs, ground_truth_obs):
        sys_prompt, user_prompt = generate_prompts_for_evaluator(
            vlm_obs, ground_truth_obs
        )
        conversation = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = self.engine.generate(conversation)
        return response

    def get_scores(self, response):
        conversation = [
            {
                "role": "system",
                "content": "Your task is to extract the final score from the following analysis. There are two scores corresponding to two aspects: 1. correctness of identifying the spatial relationships between the objects and receptacles, 2. correctness of identifying the statuses of the receptacles. The score is a float between 0 and 1.",
            },
            {"role": "user", "content": "The following is the analysis: \n" + response},
        ]
        scores = self.engine.generate_format(
            conversation=conversation, response_format=FinalScore
        )
        scores_dict = json.loads(scores.choices[0].message.content)
        return scores_dict

    def start_trial(self, trial_name, task_info):
        """Start tracking metrics for a new trial"""
        self.current_trial = trial_name
        self.current_task_info = task_info
        self.trial_metrics[trial_name] = {
            "spatial_relationship_scores": [],
            "receptacle_status_scores": [],
        }

    def log_step_scores(self, scores):
        """Log scores for current timestep"""
        if self.current_trial is None:
            return

        metrics = self.trial_metrics[self.current_trial]
        metrics["spatial_relationship_scores"].append(
            scores["spatial_relationship_score"]
        )
        metrics["receptacle_status_scores"].append(scores["receptacle_status_score"])

    def set_logger(self, traj_dir, observer_name):
        logger_name = "evaluator_logger"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(
            os.path.join(traj_dir, f"evaluator_{observer_name}.log")
        )
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(message)s")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def info(self, message):
        """Log an info message"""
        self.logger.info(message)

    def close_logger(self):
        """Close the logger"""
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)

    def get_trial_metrics(self, trial_name=None):
        """Get average metrics for a trial"""
        if trial_name is None:
            trial_name = self.current_trial

        if trial_name not in self.trial_metrics:
            return None

        metrics = self.trial_metrics[trial_name]
        return {
            "avg_spatial_relationship_score": sum(
                metrics["spatial_relationship_scores"]
            )
            / len(metrics["spatial_relationship_scores"]),
            "avg_receptacle_status_score": sum(metrics["receptacle_status_scores"])
            / len(metrics["receptacle_status_scores"]),
        }

    def get_overall_score(self):
        """Get average scores through timesteps of all trials"""
        if not self.trial_metrics:
            return None
        total_timesteps = 0
        total_spatial_relationship_score = 0
        total_receptacle_status_score = 0
        # iterate through all trials
        for _, metrics in self.trial_metrics.items():
            # sum up all scores
            total_timesteps += len(metrics["spatial_relationship_scores"])
            total_spatial_relationship_score += sum(
                metrics["spatial_relationship_scores"]
            )
            total_receptacle_status_score += sum(metrics["receptacle_status_scores"])

        return {
            "avg_total_spatial_relationship_score": total_spatial_relationship_score
            / total_timesteps,
            "avg_total_receptacle_status_score": total_receptacle_status_score
            / total_timesteps,
        }


def test_evaluator():
    vlm_obs = "Observation: You close drawer 7. You see countertop 3 and cabinet 18. The drawer 7 is closed."
    # ground_truth_obs = "On the countertop 3, you see nothing. The cabinet 18 is closed. The drawer 7 is closed. "
    ground_truth_obs = "On the countertop 3, you see nothing. "
    evaluator = Evaluator("gpt-4o")
    evaluation = evaluator.evaluate(vlm_obs, ground_truth_obs)
    print(evaluation)
    scores = evaluator.get_scores(evaluation)
    print(scores)


if __name__ == "__main__":
    test_evaluator()
