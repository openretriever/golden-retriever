# Note: prompt to install pddlgym under `./external/pddlgym` if it is not installed
try:
    import pddlgym
except ImportError:
    pddlgym = None
    print(
        "PDDLGym Package is not installed. Note: install one in the `./external/pddlgym` directory using "
        "`pip install -e .`"
    )


class PDDLGymTask:
    pass


if __name__ == "__main__":
    env = pddlgym.make("PDDLEnvBlocks-v0")
    obs = env.reset()

    print("obs in predicates", obs)
    print("render shape", env.render().shape)

    print("action space", env.action_space)
    print("state/obs space", env.observation_space)
