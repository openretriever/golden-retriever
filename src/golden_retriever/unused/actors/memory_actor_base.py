from src.utils.logic_structs import ContinuousActorBase


# TODO write a memory actor
# 1, it should async obtain observation from robot actor (Q: should it wait robot to push or pull from robot actor?)
# A: it should wait for robot actor to push - `save_observation`
# 2, it should (async) process observations to form short-term state for later use - `process_observation`
# 3, it should async store observation to persistent data storage actor - `store_observation`
# 4, it can wait other actors to pull from it - `get_memory`
class RobotMemoryActorBase(ContinuousActorBase):
    """
    This is short-term memory Ray actor for each robot. For example, it can provide object-centric state for robot
    to use.
    """

    def __init__(self):
        super().__init__()
        self.memory = None

    async def save_observation(self, observation):
        """
        Async function for storing observation

        Args:
            observation:

        Returns:

        """
        pass

    async def get_short_term_state(self):
        """
        Async function for getting short term state

        Returns:

        """
        pass

    async def get_observation(self):
        """
        Async function for getting observation

        Returns:

        """
        pass

    async def process_observation(self, observation):
        """
        Async function for processing observation

        Args:
            observation:

        Returns:

        """
        pass

    async def get_memory(self):
        """
        Async function for getting memory

        Returns:

        """
        pass


# TODO a memory actor for modeling environment and potentially be shared by multiple robots
class EnvMemoryActorBase(ContinuousActorBase):
    """
    This is long-term memory Ray actor for the environment. For example, it can provide map of the environment for
    robots to use.
    """

    def __init__(self):
        super().__init__()
        self.memory = None

    async def save_observation(self, observation):
        """
        Async function for storing observation

        Args:
            observation:

        Returns:

        """
        pass

    async def get_short_term_state(self):
        """
        Async function for getting short term state

        Returns:

        """
        pass

    async def get_observation(self):
        """
        Async function for getting observation

        Returns:

        """
        pass

    async def process_observation(self, observation):
        """
        Async function for processing observation

        Args:
            observation:

        Returns:

        """
        pass

    async def get_memory(self):
        """
        Async function for getting memory

        Returns:

        """
        pass
