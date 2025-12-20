from __future__ import annotations

import re
from typing import Any, Dict, List

from retriever.core.skills import GroundedSkill, SkillSignature
from retriever.core.symbolic_structs import GroundAtom, State
from retriever.core.types import Module, NLCommand, ObjectDescriptionDict
from retriever.models.api_models.clients import LLMClient, create_llm_client


class LLMPlanner(
    Module[
        tuple[ObjectDescriptionDict, NLCommand, list[SkillSignature]],
        list[GroundedSkill],
    ]
):
    """
    A planner that uses a generic LLM client to generate a sequence of
    grounded skills.
    """

    def __init__(
        self,
        client: LLMClient | None = None,
        client_type: str = "openai",
        client_kwargs: Dict[str, Any] | None = None,
    ):
        """
        Initializes the planner.

        The client can be provided in three ways, in order of precedence:
        1. Pass a pre-initialized `client` object.
        2. Pass a `client_type` string (e.g., "openai" or "gemini") and
           optional `client_kwargs` to have the planner create the client.
        3. Do nothing to use the default client ("openai").

        Args:
            client: A pre-initialized object that implements the LLMClient protocol.
            client_type: The type of client to create if a client object is not provided.
            client_kwargs: A dictionary of keyword arguments for the client constructor.
        """
        if client is not None:
            self._client = client
        else:
            client_kwargs = client_kwargs or {}
            if "max_tokens" not in client_kwargs:
                client_kwargs["max_tokens"] = 1024
            self._client = create_llm_client(client_type, **client_kwargs)

    def __call__(
        self,
        inp: tuple[ObjectDescriptionDict, NLCommand, list[SkillSignature]],
    ) -> list[GroundedSkill]:
        """
        Takes the current state, goal, and available skills, and returns a plan.
        """
        object_descriptions, goal_command, available_skills = inp

        prompt = self._construct_prompt(
            object_descriptions, goal_command, available_skills
        )
        llm_output = self._client.predict(prompt)

        if llm_output is None:
            raise ValueError("Received empty response from the LLM client.")

        plan = self._parse_plan(
            llm_output, available_skills, object_descriptions.descriptions
        )

        return plan

    def _construct_prompt(
        self,
        objects: ObjectDescriptionDict,
        goal: NLCommand,
        skills: list[SkillSignature],
    ) -> str:
        """Constructs a detailed prompt for the LLM."""
        prompt_parts = [
            (
                "You are a helpful robot assistant. Your task is to generate a plan "
                "to achieve a goal, given the state of the world."
            ),
            (
                "The plan should be a sequence of skills, one per line. The output "
                "format for each skill is: skill_name(param1=object_id_1, ...)"
            ),
            "## Perceived Objects",
            "Here is a list of objects you see, with their unique ID and description:",
        ]
        for obj_id, desc in sorted(objects.descriptions.items()):
            prompt_parts.append(f"- {obj_id}: {desc}")

        prompt_parts.extend(
            [
                "## Available Skills",
                (
                    "Here are the skills you can use. The parameters must be replaced "
                    "with object IDs from the list above."
                ),
            ]
        )
        for skill in sorted(skills, key=lambda s: s.name):
            prompt_parts.append(f"- {skill.name}({', '.join(skill.parameters)})")

        prompt_parts.extend(
            [
                "## Goal",
                f'The user\'s command is: "{goal.text}"',
                "## Plan",
                "Provide the plan to achieve the goal below. Do not output anything else.",
            ]
        )
        return "\n\n".join(prompt_parts)

    def _parse_plan(
        self,
        llm_output: str,
        skills: list[SkillSignature],
        perceived_objects: dict[str, str],
    ) -> list[GroundedSkill]:
        """Parses the LLM's string output into a list of GroundedSkills."""
        plan = []
        skill_map = {s.name: s for s in skills}

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
                    key, value = part.strip().split("=")
                    grounded_params[key.strip()] = value.strip()

            skill = GroundedSkill(
                signature=signature, grounded_params=grounded_params
            )
            skill.validate_grounding(perceived_objects)
            plan.append(skill)

        return plan 