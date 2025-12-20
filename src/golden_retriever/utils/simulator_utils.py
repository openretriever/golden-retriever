from retriever.envs.habitat.habitat_env import House


def make_env(config):
    """Make Habitat simulator"""
    house = House(
        scene=config["scene_cfg"]["scene_file"],  # scene file
        rnd_seed=config["scene_cfg"]["random_seed"],  # random seed
        allow_sliding=config["scene_cfg"][
            "allow_sliding"
        ],  # If True, agent will slide along the walls
        max_episode_length=config["scene_cfg"][
            "max_episode_length"
        ],  # Maximal episode length
        goal_reach_eps=config["scene_cfg"]["goal_reach_eps"],  # goal reaching threshold
        # observation configuration
        enable_rgb_depth=True,
        sensor_height=config["sensor_cfg"]["sensor_height"],
        obs_width=config["sensor_cfg"]["obs_width"],
        obs_height=config["sensor_cfg"]["obs_height"],
        # map configuration
        map_show_agent=config["map_cfg"]["show_agent"],
        map_show_goal=config["map_cfg"]["show_goal"],
        # agent configuration
        move_forward_amount=config["agent_cfg"]["move_forward"],
        turn_left_amount=config["agent_cfg"]["turn_left"],
        turn_right_amount=config["agent_cfg"]["turn_right"],
    )
    return house
