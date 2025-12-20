from __future__ import annotations

import re
from typing import List, Optional

import dspy

from retriever.core.skills import GroundedSkill, SkillSignature
from retriever.core.types import Module, NLCommand, ObjectDescriptionDict


class PlannerSignature(dspy.Signature):
    """
    Given the state of the world (perceived objects), a set of available skills,
    and a user's goal, generate a sequence of skills to achieve the goal.
    """

    preamble = (
        "You are a helpful robot assistant. Your task is to generate a plan "
        "to achieve a goal, given the state of the world."
    )
    __doc__ = (preamble or "") + "\n\n" + (__doc__ or "")

    world_state = dspy.InputField(
        desc="A list of perceived objects with their unique ID and description. One per line, prefixed with '-'."
    )
    available_skills = dspy.InputField(
        desc="A list of skills that can be used. Format: skill_name(param1, ...). One per line, prefixed with '-'."
    )
    goal = dspy.InputField(desc="The user's command.")

    plan = dspy.OutputField(
        desc="A sequence of skills, one per line. The output format for each skill is: skill_name(param1=object_id_1, ...). Do not output anything else."
    )


class DSPyLLMPlanner(
    Module[
        tuple[ObjectDescriptionDict, NLCommand, list[SkillSignature]],
        list[GroundedSkill],
    ]
):
    """
    A planner that uses a Language Model via DSPy to generate a sequence of
    grounded skills.
    """

    def __init__(self, lm: Optional[dspy.LM] = None):
        """
        Initializes the planner.

        Args:
            lm: A configured DSPy Language Model. If None, DSPy will attempt
                to use a model configured globally (e.g., from environment
                variables). You can configure one like this:
                For OpenAI:
                turbo = dspy.OpenAI(model='gpt-3.5-turbo', api_key='...')
                dspy.settings.configure(lm=turbo)
                For a local Ollama model:
                ollama_lm = dspy.Ollama(model='llama3', max_tokens=1024)
                dspy.settings.configure(lm=ollama_lm)
        """
        if lm is not None:
            dspy.settings.configure(lm=lm)
        elif not dspy.settings.lm:
            raise ValueError(
                "No DSPy LM is configured. Pass one to the constructor or "
                "configure one globally via dspy.settings.configure(lm=...)."
            )

        self.predictor = dspy.Predict(PlannerSignature)

    def __call__(
        self,
        inp: tuple[ObjectDescriptionDict, NLCommand, list[SkillSignature]],
    ) -> list[GroundedSkill]:
        """
        Takes the current state, goal, and available skills, and returns a plan.
        """
        object_descriptions, goal_command, available_skills = inp

        # Format inputs for the DSPy signature
        world_state_str = "\n".join(
            f"- {obj_id}: {desc}"
            for obj_id, desc in sorted(object_descriptions.descriptions.items())
        )
        skills_str = "\n".join(
            f"- {s.name}({', '.join(s.parameters)})"
            for s in sorted(available_skills, key=lambda s: s.name)
        )

        result = self.predictor(
            world_state=world_state_str,
            available_skills=skills_str,
            goal=goal_command.text,
        )

        plan = self._parse_plan(
            result.plan, available_skills, object_descriptions.descriptions
        )

        return plan

    def _parse_plan(
        self,
        llm_output: str,
        skills: list[SkillSignature],
        perceived_objects: dict[str, str],
    ) -> list[GroundedSkill]:
        """Parses the LLM's string output into a list of GroundedSkills."""
        plan = []
        skill_map = {s.name: s for s in skills}

        if llm_output is None:
            raise ValueError("LLM returned an empty plan.")

        for line in llm_output.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            match = re.match(r"(\w+)\((.*)\)", line)
            if not match:
                raise ValueError(f"Could not parse line: '{line}'")

            skill_name, params_str = match.groups()

            if skill_name not in skill_map:
                raise ValueError(f"Unknown skill name: '{skill_name}'")

            signature = skill_map[skill_name]
            grounded_params = {}
            if params_str:
                for part in params_str.split(","):
                    key_val = part.strip().split("=")
                    if len(key_val) != 2:
                        raise ValueError(f"Malformed parameter in line: '{line}'")
                    key, value = key_val
                    grounded_params[key.strip()] = value.strip()

            skill = GroundedSkill(
                signature=signature, grounded_params=grounded_params
            )
            skill.validate_grounding(perceived_objects)
            plan.append(skill)

        return plan 