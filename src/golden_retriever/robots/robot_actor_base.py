from typing import Dict, Optional

from retriever.skills.skill_registry import SkillRegistry


class RobotActorBase:
    def __init__(self, actors: Optional[Dict[str, object]] = None):
        self.actors = actors
        self.skill_queue = []

    def observe_environment(self):
        # Placeholder for observation logic
        observation = {}
        return observation

    async def execute_skill(self, skill_name, skill_params):
        """
        Async Ray function - Execute a skill by name and parameters. It can be called via `.remote()` to schedule to
        execute.

        Args:
            skill_name:
            skill_params

        Returns:

        """
        skill_func = SkillRegistry.get_skill(skill_name)
        if skill_func:
            result = await skill_func(skill_params)
            return result
        return None

    async def push_observation(self):
        """
        Async function for push observation

        Returns:

        """
        while True:
            observation = self.observe_environment()
            await self.memory_actor.store_observation.remote(observation)


class MobileComponent:
    # ... mobile-specific methods ...
    pass


class ArmComponent:
    # ... arm-specific methods ...
    pass


class MobileArmRobotActorBase(RobotActorBase):
    def __init__(
        self,
        mobile_component: MobileComponent = None,
        arm_component: ArmComponent = None,
        actors: Optional[Dict[str, object]] = None,
    ):
        super().__init__(actors)
        self.skill_set = []

    def observe_environment(self):
        # Placeholder for observation logic
        observation = {}
        return observation


class ArmRobotActorBase(RobotActorBase):
    def __init__(
        self,
        arm_component: ArmComponent = None,
        actors: Optional[Dict[str, object]] = None,
    ):
        super().__init__(actors)
        self.skill_set = []

    def observe_environment(self):
        # Placeholder for observation logic
        observation = {}
        return observation
