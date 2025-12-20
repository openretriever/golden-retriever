import math
from collections import OrderedDict

import habitat_sim
import habitat_sim.errors
import magnum as mn
import matplotlib.pyplot as plt
import numpy as np
import quaternion
from habitat.utils.visualizations import maps
from habitat_sim.utils.common import (
    d3_40_colors_rgb,
    quat_from_angle_axis,
    quat_from_coeffs,
    quat_to_angle_axis,
)
from PIL import Image
from skimage.color import rgb2gray
from skimage.transform import resize

from retriever.models.mapping.utils_vlmaps.utils import compute_gps_and_compass_point_goal

""" Notes:
        1. It is weird that the Matterport3D and Gibson does not come with the semantic offline_dataset.
        2. The .navmesh file in the same folder will be loaded simultaneously with the .ply or .glb file.
        3. The environment provides single RGB, RGB-D, Depth, and Semantic observations.
        4. The environment provides panoramic RGB, RGB-D, Depth, and Semantic observations.
        5. The environment provides top down observation (agent).
        6. The agent is randomly spawn. The location is randomly chosen from a continuous space and orientation
           is randomly chosen from [east, north, west, south]
        7. The output depth can be clipped to a fixed range see depth_clip_vmax
        8. The back observation is not flipped anymore. (changed in NeurIPs 22 release)

    Important:
        1. In math, a quaternion is represented as q = q_0 + q_1 * i + q_2 * j + q_3 * k, where q_0 is the real
           component and q_1 - q_3 are imaginary components. In general, a quaternion is represented as
           (q_0, q_1, q_2, q_3).
        2. 3-D rotation to quaternion, suppose a rotation is defined as (x, y, z, \theta) where (x, y, z) is a vector
           representing the rotation axis and \theta is the rotation amount (+ for counter clock-wise). We can represent
           the quaternion q = (q_0, q_1, q_2, q_3) as follows:
                - q_0 = np.cos(\theta / 2)
                - q_1 = x * np.sin(\theta / 2)
                - q_2 = y * np.sin(\theta / 2)
                - q_3 = z * np.sin(\theta / 2)
        3. In habitat, the rotation stored in the agent_state is np.quaternion (i.e., q_0, q_1, q_2, q_3). However,
           when a quaternion is represented in lists or numpy arrays, it is [q_1, q_2, q_3, q_0]. Use the quat_to_coeffs
           and quat_from_coeffs to convert between them.
        4. Key takeaway:
            To save quaternion in (q_0, q_1, q_2, q_3) format, use np.quaternion.as_float_array()
"""


class House(object):
    def __init__(
        self,
        scene=None,
        enable_rgb=False,
        enable_depth=False,
        enable_rgb_depth=False,
        enable_semantic=False,
        enable_panorama=False,
        enable_obs_noise=False,
        noise_intensity=0.1,
        enable_act_noise=False,
        enable_local_map=False,
        top_down_type="boundary",
        sensor_height=1.5,
        rnd_seed=1234,
        map_meters_per_pixel=0.01,
        local_map_size=50,
        enable_ego_local_map=False,
        obs_width=224,
        obs_height=224,
        move_forward_amount=0.25,
        turn_left_amount=10,
        turn_right_amount=10,
        map_show_agent=False,
        map_show_goal=False,
        goal_reach_eps=0.2,  # from active neural slam paper
        depth_clip_vmax=False,
        allow_sliding=False,
        max_episode_length=500,
        plot_trajectories=False,
    ):
        """
        The wrapper class for Facebook Habitat environment with several customized functions for PointGoal navigation
        :param scene: the absolute path of the scene files. For example, name.glb (Gibson); name.ply (Replica); string
        :param enable_rgb: if True, enable the RGB observations; bool
        :param enable_depth: if True, enable the Depth observations; bool
        :param enable_rgb_depth: if True, enable the RGB-D observations; bool
        :param enable_semantic: if True, enable the Semantic observations; bool
        :param enable_panorama: if True, enable the panoramic observations; bool
        :param enable_obs_noise: if True, enable the noisy observations; bool
        :param noise_intensity: we use the "GaussianNoiseModel", value 0 - 1, 0 -> noiser; bool
        :param enable_act_noise: if True, enable the noisy actions; bool
        :param enable_local_map: if True, the local map is plotted; Otherwise,the global map is plotted; bool
        :param enable_physics: related to kinematics; by default: False; bool
        :param top_down_type: binary for binary occupancy map; boundary for RGB occupancy map; string
        :param sensor_height: camera height (m); float
        :param rnd_seed: random seed; by default: 1234; int
        :param map_meters_per_pixel: map roughness parameter, number of meters for one pixel; by default 0.01; float
        :param local_map_size: 1/2 size of the local map; by default 50; int
        :param obs_width: width of the observation; @param {type: "integer"}
        :param obs_height: height of the observation; int
        :param move_forward_amount: number of meters for one forward action (m); float
        :param turn_left_amount: number of degrees for the turning left action (degree); float
        :param turn_right_amount: number of degrees for the turning right action (degree); float
        :param map_show_agent: if True, show agent on the map; bool
        :param map_show_goal: if True, show goal on the map; bool
        :param goal_reach_eps: termination signal, if distance (agent, goal) <= eps, agents reach the goal; float
        :param enable_ego_local_map: if True, plot the map in an egocentric way; bool
        :param depth_clip_vmax: max value to clip the depth observation; float
        :param allow_sliding: if False, the sliding behavior is disabled; @param {type: "boolean"}
        """

        # check the scene validation
        assert (
            scene is not None
        ), f"Invalid scene error: Got {scene}. Please check the path/name of the scene file."

        # scene file in: .ply or .glb
        self.scene = scene

        # epsilon for goal reaching
        self.dist_eps = goal_reach_eps

        # set randomness
        self.rnd_seed = rnd_seed

        """
            Sensor configurations
        """
        self.enable_rgb = enable_rgb
        self.enable_depth = enable_depth
        self.enable_rgb_depth = enable_rgb_depth
        self.enable_semantic = enable_semantic
        self.enable_panorama = enable_panorama
        self.enable_noise_obs = enable_obs_noise
        self.noise_intensity = noise_intensity
        self.depth_clip_vmax = depth_clip_vmax
        panorama_names = ["_front", "_left", "_right", "_back"]
        # create the list containing sensor names
        # for monocular: [color_sensor, depth_sensor]
        # for panoramic: [color_sensor_front, color_sensor_left, color_sensor_right, color_sensor_back]
        # note that: rgb-d observations contain both color_sensors and depth_sensors
        if self.enable_panorama:
            if self.enable_rgb_depth:
                sensor_names = ["color_sensor" + item for item in panorama_names] + [
                    "depth_sensor" + item for item in panorama_names
                ]
            elif self.enable_rgb:
                sensor_names = ["color_sensor" + item for item in panorama_names]
            elif self.enable_depth:
                sensor_names = ["depth_sensor" + item for item in panorama_names]
            elif self.enable_semantic:
                sensor_names = ["semantic_sensor" + item for item in panorama_names]
            else:
                raise Exception("Invalid observation")
        else:
            if self.enable_rgb:
                sensor_names = ["color_sensor"]
            elif self.enable_depth:
                sensor_names = ["depth_sensor"]
            elif self.enable_rgb_depth:
                sensor_names = ["color_sensor", "depth_sensor"]
            else:
                raise Exception("Invalid observation")

        # set the sensor names
        self.sensor_names = sensor_names
        # add top-down observation
        self.sensor_names.append("top_down")

        """
            Habitat settings
        """
        self.habitat_settings = {
            "scene": scene,  # path the scene
            "default_agent": 0,  # index of the default agent
            "sensor_height": sensor_height,  # height of sensor in meters
            "width": obs_width,  # observation image width
            "height": obs_height,  # observation image height
            "color_sensor": enable_rgb,  # use RGB sensor
            "depth_sensor": enable_depth,  # use Depth sensor
            "semantic_sensor": enable_semantic,  # use semantic sensor
            "seed": rnd_seed,  # used in random navigation
            "enable_physics": False,  # for kinematic only
        }

        # define the action space
        # The action space could be customized (https://aihabitat.org/docs/habitat-sim/new-actions.html)
        self.enable_act_noise = enable_act_noise
        # Reorder the action order to align with the pre-trained PPO (Following the official order)
        self.action_space = ["stop", "move_forward", "turn_left", "turn_right"]
        self.move_forward_amount = move_forward_amount
        self.turn_left_amount = turn_left_amount
        self.turn_right_amount = turn_right_amount

        # simulator initial configurations
        self.sim_cfg = None
        # agent initial configurations
        self.agent_cfg = None
        # control the sliding behavior
        self.allow_sliding = allow_sliding
        # create simulator, agent, and optimal path follower
        self.simulator, self.agent, self.follower = self.create_simulator_and_agent()

        """
            Map configurations
        """
        self.enable_local_map = enable_local_map
        self.meters_per_pixel = map_meters_per_pixel
        self.local_map_size = local_map_size
        self.map_show_agent = map_show_agent
        self.map_show_goal = map_show_goal
        self.enable_ego_local_map = enable_ego_local_map
        if top_down_type == "binary":
            self.top_down_map = (
                self.render_top_down_view_binary()
            )  # occupancy map: binary representation
        elif top_down_type == "boundary":
            self.top_down_map = (
                self.render_top_down_view_habitat()
            )  # occupancy map: RGB representation
            self.top_down_gray_map = self.rgb2gray(
                self.top_down_map.copy()
            )  # convert to gray
        else:
            raise Exception("Invalid top down map mode.")
        # plot the trajectories on the map
        self.top_down_map_traj = self.top_down_map.copy()
        self.top_down_map_binary = 1 - self.render_top_down_view_binary()

        """
            Environment parameters
        """
        boundary = self.simulator.pathfinder.get_bounds()  # Environment parameters
        self.min_x = boundary[0][0]
        self.max_x = boundary[1][0]  # height?
        self.min_y = boundary[0][2]
        self.max_y = boundary[1][2]  # width?

        """
            Plot the figures
        """
        self.fig = None
        self.arr = None
        self.artists = []
        self.plot_traj = plot_trajectories
        self.top_down_visualization = None

        """
            Episode parameters
        """
        # sample the start location, the goal location, and the geometric distance
        self.start_loc, self.goal_loc, self.geo_dist = None, None, None
        # record the shortest path
        self.shortest_path = None
        # record the optimal actions
        self.optimal_actions = []
        # track the geometric distance between the agent and the goal locations
        self.agent_goal_dist = np.inf
        # maximal episode length
        self.max_episode_length = max_episode_length
        # time step counter
        self.step_counter = 0

        """
            Task evaluation metrics
        """
        self.metrics = OrderedDict()
        self.metrics["DISTANCE_TO_GOAL"] = -1.0
        self.metrics["SUCCESS"] = False
        self.metrics["SPL"] = 0.0
        self.metrics["ACTUAL_PATH"] = 0.0

    # render binary top down observation
    def render_top_down_view_binary(self):
        height = self.simulator.pathfinder.get_bounds()[0][1]
        sim_top_down_map = self.simulator.pathfinder.get_topdown_view(
            self.meters_per_pixel, height
        )
        return sim_top_down_map

    # render boundary top down observation
    def render_top_down_view_habitat(self):
        height = self.simulator.pathfinder.get_bounds()[0][1]
        top_down_map = maps.get_topdown_map(
            self.simulator.pathfinder, height, meters_per_pixel=self.meters_per_pixel
        )
        recolor_map = np.array(
            [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
        )
        top_down_map = recolor_map[top_down_map]
        return top_down_map

    # sensor specifications
    def embedded_agent_sensor_specs(self):
        # sensors
        sensor_specs = []

        # default observation resolution
        img_height = self.habitat_settings["height"]
        img_width = self.habitat_settings["width"]
        sensor_height = self.habitat_settings["sensor_height"]

        # function to get specifications
        def obtain_sensor_spec(sensor_type, sensor_uuid, orientation):
            # get standard sensor specification
            # TODO Note: for after v0.1.7, use this version
            # _spec = habitat_sim.CameraSensorSpec()  # For Habitat-sim v0.1.7 later
            _spec = habitat_sim.SensorSpec()

            # set sensor uuid (i.e., indicator)
            _spec.uuid = sensor_uuid
            # set sensor type
            if sensor_type == "color":
                _spec.sensor_type = habitat_sim.SensorType.COLOR
            elif sensor_type == "depth":
                _spec.sensor_type = habitat_sim.SensorType.DEPTH
            elif sensor_type == "semantic":
                _spec.sensor_type = habitat_sim.SensorType.SEMANTIC
            else:
                raise Exception(
                    f"Invalid sensor type. Expect color, depth, or semantic, but get {sensor_type}"
                )
            # set sensor model
            _spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            # set sensor resolution, position and orientation
            _spec.resolution = [img_height, img_width]
            _spec.position = [0.0, sensor_height, 0.0]  # [x, y, z], meters
            _spec.orientation = (
                orientation  # [pitch, yaw, roll], radius, counter-clock wise
            )

            # enable observation noise
            if self.enable_noise_obs:
                if sensor_type == "color":
                    _spec.noise_model = "GaussianNoiseModel"
                    _spec.noise_model_kwargs = dict(
                        intensity_constant=self.noise_intensity
                    )
                elif sensor_type == "depth":
                    _spec.noise_model = "RedwoodDepthNoiseModel"
                    _spec.noise_model_kwargs = dict(noise_multiplier=1.0)
                else:
                    raise Exception("Wrong sensor type")
            return _spec

        if not self.enable_panorama:
            # set the RGB camera
            if self.enable_rgb:
                # get rgb camera
                rgb_sensor_spec = obtain_sensor_spec(
                    "color", "color_sensor", [0.0, 0.0, 0.0]
                )
                # add the rgb camera
                sensor_specs.append(rgb_sensor_spec)

            # set the depth camera
            if self.enable_depth:
                # get depth camera
                depth_sensor_spec = obtain_sensor_spec(
                    "depth", "depth_sensor", [0.0, 0.0, 0.0]
                )
                # add the depth camera
                sensor_specs.append(depth_sensor_spec)

            # set the semantic camera
            if self.enable_semantic:
                # get semantic camera
                semantic_sensor_spec = obtain_sensor_spec(
                    "semantic", "semantic_sensor", [0.0, 0.0, 0.0]
                )
                # add the semantic sensor
                sensor_specs.append(semantic_sensor_spec)

            # set the RGB-D camera
            if self.enable_rgb_depth:
                # get the RGB camera
                rgb_sensor_spec = obtain_sensor_spec(
                    "color", "color_sensor", [0.0, 0.0, 0.0]
                )
                # get the depth camera
                depth_sensor_spec = obtain_sensor_spec(
                    "depth", "depth_sensor", [0.0, 0.0, 0.0]
                )
                # add the RGB and depth cameras
                sensor_specs.append(rgb_sensor_spec)
                sensor_specs.append(depth_sensor_spec)
        else:
            spec_orientations = [
                [0.0, 0.0, 0.0],
                [0.0, np.pi, 0.0],
                [0.0, np.pi / 2, 0.0],
                [0.0, -1 * np.pi / 2, 0.0],
            ]
            spec_uuids = ["sensor_front", "sensor_back", "sensor_left", "sensor_right"]
            if self.enable_rgb:
                for spec_ori, spec_uuid in zip(
                    spec_orientations, spec_uuids, strict=False
                ):
                    sensor_specs.append(
                        obtain_sensor_spec("color", "color_" + spec_uuid, spec_ori)
                    )

            if self.enable_depth:
                for spec_ori, spec_uuid in zip(
                    spec_orientations, spec_uuids, strict=False
                ):
                    sensor_specs.append(
                        obtain_sensor_spec("depth", "depth_" + spec_uuid, spec_ori)
                    )

            if self.enable_semantic:
                for spec_ori, spec_uuid in zip(
                    spec_orientations, spec_uuids, strict=False
                ):
                    sensor_specs.append(
                        obtain_sensor_spec(
                            "semantic", "semantic_" + spec_uuid, spec_ori
                        )
                    )

            if self.enable_rgb_depth:
                for spec_ori, spec_uuid in zip(
                    spec_orientations, spec_uuids, strict=False
                ):
                    sensor_specs.append(
                        obtain_sensor_spec("color", "color_" + spec_uuid, spec_ori)
                    )
                    sensor_specs.append(
                        obtain_sensor_spec("depth", "depth_" + spec_uuid, spec_ori)
                    )

        # set the sensor specifications
        self.agent_cfg.sensor_specifications = sensor_specs

    def embedded_agent_action_specs(self):
        if self.enable_act_noise:
            self.agent_cfg.action_space = dict(
                move_forward=habitat_sim.ActionSpec(
                    "move_forward",
                    habitat_sim.PyRobotNoisyActuationSpec(
                        amount=self.move_forward_amount,
                        robot="LoCoBot",
                        controller="Proportional",
                        noise_multiplier=1.0,
                    ),
                ),
                turn_left=habitat_sim.ActionSpec(
                    "turn_left",
                    habitat_sim.PyRobotNoisyActuationSpec(
                        amount=self.turn_left_amount,
                        robot="LoCoBot",
                        controller="Proportional",
                        noise_multiplier=0.5,
                    ),
                ),
                turn_right=habitat_sim.ActionSpec(
                    "turn_right",
                    habitat_sim.PyRobotNoisyActuationSpec(
                        amount=self.turn_right_amount,
                        robot="LoCoBot",
                        controller="Proportional",
                        noise_multiplier=0.5,
                    ),
                ),
            )
        else:
            self.agent_cfg.action_space = {
                "move_forward": habitat_sim.agent.ActionSpec(
                    "move_forward",
                    habitat_sim.agent.ActuationSpec(amount=self.move_forward_amount),
                ),
                "turn_left": habitat_sim.agent.ActionSpec(
                    "turn_left",
                    habitat_sim.agent.ActuationSpec(amount=self.turn_left_amount),
                ),
                "turn_right": habitat_sim.agent.ActionSpec(
                    "turn_right",
                    habitat_sim.agent.ActuationSpec(amount=self.turn_right_amount),
                ),
                "stop": habitat_sim.agent.ActionSpec(
                    "stop", habitat_sim.agent.ActuationSpec(amount=0)
                ),
            }

    # create the simulator and agent
    def create_simulator_and_agent(self):
        # create the simulator configurations
        self.sim_cfg = (
            habitat_sim.SimulatorConfiguration()
        )  # get the configuration object for the simulator
        self.sim_cfg.gpu_device_id = 0  # assign a GPU to the simulator
        self.sim_cfg.scene_id = self.habitat_settings[
            "scene"
        ]  # define the scene name with the ply files
        self.sim_cfg.enable_physics = self.habitat_settings[
            "enable_physics"
        ]  # enable the physics

        # create the agent configurations
        self.agent_cfg = (
            habitat_sim.agent.AgentConfiguration()
        )  # get the configuration object for the agent

        # create the sensor configurations
        self.embedded_agent_sensor_specs()

        # create the action configurations
        self.embedded_agent_action_specs()

        # create the simulator
        self.sim_cfg = habitat_sim.Configuration(self.sim_cfg, [self.agent_cfg])
        simulator = habitat_sim.Simulator(self.sim_cfg)

        # disable the sliding behavior
        simulator.config.sim_cfg.allow_sliding = self.allow_sliding

        # create the agent
        agent = simulator.initialize_agent(0)
        # set the agent state
        agent_state = habitat_sim.AgentState()
        agent_state.position = np.array(
            [0.0, 0.0, 0.0]
        )  # the agent is set at [0, 0, 0] by default
        agent.set_state(agent_state)

        # create follower for optimal path planning
        follower = habitat_sim.nav.GreedyGeodesicFollower(
            pathfinder=simulator.pathfinder,
            agent=agent,
            goal_radius=self.dist_eps,
            forward_key="move_forward",
            left_key="turn_left",
            right_key="turn_right",
            stop_key="stop",
        )

        return simulator, agent, follower

    def is_on_map(self, loc):
        c = int((loc[0] - self.min_x) // self.meters_per_pixel)
        r = int((loc[2] - self.min_y) // self.meters_per_pixel)

        return not self.top_down_gray_map[r, c] == 1.0

    # sample navigable start and goal locations
    def sample_start_goal_locations(self):
        while True:
            # sample start and goal locations
            s_loc = self.simulator.pathfinder.get_random_navigable_point()
            g_loc = self.simulator.pathfinder.get_random_navigable_point()

            # exam whether the path exists
            path = habitat_sim.ShortestPath()
            path.requested_start = s_loc
            path.requested_end = g_loc
            found_path = self.simulator.pathfinder.find_path(path)
            geodesic_distance = path.geodesic_distance
            if found_path is True and self.is_on_map(s_loc) and self.is_on_map(g_loc):
                # return start, goal locations and geometry distance
                return s_loc, g_loc, geodesic_distance

    # compute the geometric ditance
    def compute_geo_distance(self, start_loc, goal_loc):
        path = habitat_sim.ShortestPath()
        path.requested_start = start_loc
        path.requested_end = goal_loc
        self.simulator.pathfinder.find_path(path)
        return path.geodesic_distance

    # set agent state
    def set_agent_state(self, loc, ori):
        # Create the agent state
        agent_state = habitat_sim.AgentState()

        # Set the position
        agent_state.position = loc

        # Set the rotation: convert the Euler angle to quaternion
        # This should follow the world coordinate frame of Habitat simulator
        # right-handed, +x right (pitch), +y up (yaw), +z (roll) out of the screen
        # The agent locates on the x-z plane
        # The axis seems to accept [x, y, z]
        if isinstance(ori, float):
            agent_state.rotation = quat_from_angle_axis(ori, np.array([0, 1, 0]))
        elif isinstance(ori, list):
            agent_state.rotation = quat_from_coeffs(ori)
        else:
            raise NotImplementedError

        # Assign the state
        self.agent.set_state(agent_state)

    # render agent observations
    def render_agent_observations(self):
        # render from the simulator
        observations = self.simulator.get_sensor_observations()

        # save the observation in a dict
        return_obs = {}
        for name in self.sensor_names:
            if name == "top_down":  # render top down observation
                return_obs[name] = self.draw_agent_on_top_down_map()
            else:  # render sensor observations
                if (
                    "depth" in name and self.depth_clip_vmax
                ):  # clip the depth image if depth_clip_vmax != 0
                    observations[name] = np.clip(
                        observations[name], a_min=0, a_max=self.depth_clip_vmax
                    )

                return_obs[name] = observations[name]

        return return_obs

    # gym reset function
    def reset(self, start_pose=None, goal_pose=None):
        # Reset the step counter
        self.step_counter = 0

        if start_pose is None and goal_pose is None:
            # Randomly sample the start and goal locations
            (
                self.start_loc,
                self.goal_loc,
                self.geo_dist,
            ) = self.sample_start_goal_locations()
        else:
            # Set start and goal poses
            self.start_loc = start_pose[0:3]
            self.goal_loc = goal_pose
            self.geo_dist = self.query_shortest_path(self.start_loc, self.goal_loc)

        # Reset the top-down map
        self.top_down_map_traj = self.top_down_map.copy()

        # Randomly sample the agent's orientation
        # For set the orientation: [0, PI / 2, PI, PI * 3 / 2] = [North, West, South, East]
        # For render the orientation: [PI, PI * 3 / 2, 0, PI / 2] = [North, West, South, East]
        # ori = random.sample([0, np.pi/2, np.pi, np.pi * 3 / 2], 1)[0]
        if start_pose is None:
            ori = 0.0  # Fixed the orientation to be north
        else:
            ori = start_pose[3:]
        # Set the agent's orientation
        self.set_agent_state(loc=self.start_loc, ori=ori)

        # Compute the optimal actions
        try:
            self.optimal_actions = self.follower.find_path(self.goal_loc)
        except habitat_sim.errors.GreedyFollowerError:
            self.optimal_actions = [None]

        # Render agent observation
        obs = self.render_agent_observations()

        # Render agent state: position [x, y, z] and quaternion
        agent_state = self.agent.get_state()
        obs["position"] = agent_state.position.tolist()  # save the agent's position
        # Note the rotation is represented as a quaternion (i.e., [q_0, q_1, q_2, q_3] or [a, b, c, d] or [w, x, y, z])
        # where q_0, a  or w is the real component
        obs["quaternion"] = quaternion.as_float_array(agent_state.rotation).tolist()

        # Add GPS + Compass goal
        # Return the goal relative to agent's current location in the polar coordinate
        obs["pointgoal_with_gps_compass_sensor"] = compute_gps_and_compass_point_goal(
            agent_state.position, agent_state.rotation, np.array(self.goal_loc)
        )

        # Reset the metrics
        self.metrics["DISTANCE_TO_GOAL"] = obs["pointgoal_with_gps_compass_sensor"][0]
        self.metrics["SUCCESS"] = False
        self.metrics["SPL"] = 0.0
        self.metrics["ACTUAL_PATH"] = 0.0
        self.metrics["OPTIMAL_PATH"] = self.geo_dist

        return obs

    # gym step function
    def step(self, act):
        # Step
        self.simulator.step(act)

        # Update the actual path
        self.metrics["ACTUAL_PATH"] += (
            self.move_forward_amount if act == "move_forward" else 0.0
        )

        # Render observation
        obs = self.render_agent_observations()

        # Render state
        agent_state = self.agent.get_state()

        # Split the position and quaternion
        obs["position"] = agent_state.position.tolist()
        obs["quaternion"] = quaternion.as_float_array(agent_state.rotation).tolist()

        # Add GPS + Compass goal
        obs["pointgoal_with_gps_compass_sensor"] = compute_gps_and_compass_point_goal(
            agent_state.position, agent_state.rotation, np.array(self.goal_loc)
        )

        # check the termination
        done, timeout = self.is_terminal(obs["position"])

        # Update the metrics
        self.update_metrics(done)

        # increase the time step
        self.step_counter += 1

        return obs, done, timeout

    def update_metrics(self, done):
        # Find the shortest path
        path = habitat_sim.ShortestPath()
        path.requested_start = self.agent.state.position.tolist()
        path.requested_end = self.goal_loc
        self.simulator.pathfinder.find_path(path)

        # Geometric distance
        self.metrics["DISTANCE_TO_GOAL"] = path.geodesic_distance
        # Compute the SUCCESS
        self.metrics["SUCCESS"] = 1.0 if done else 0.0
        # Compute the SPL
        self.metrics["SPL"] = (
            self.metrics["SUCCESS"]
            * self.metrics["OPTIMAL_PATH"]
            / max(self.metrics["ACTUAL_PATH"], self.metrics["OPTIMAL_PATH"])
        )

    # gym render function
    def render(self, observation, time_step):
        if self.enable_panorama:
            if self.enable_rgb:
                self.display_panorama(observation, time_step, "color_sensor")

            if self.enable_depth:
                self.display_panorama(observation, time_step, "depth_sensor")

            if self.enable_semantic:
                self.display_panorama(observation, time_step, "semantic_sensor")

            if self.enable_rgb_depth:
                self.display_panorama(observation, time_step, "color_sensor")
        else:
            self.display_observations(observation, time_step)

    def query_shortest_path(self, start_loc, goal_loc):
        # Use ShortestPath module to compute path between samples.
        self.shortest_path = habitat_sim.ShortestPath()
        # set the start and goal locations
        self.shortest_path.requested_start = start_loc
        self.shortest_path.requested_end = goal_loc
        # find the path use the built-in path finder
        self.simulator.pathfinder.find_path(self.shortest_path)
        return self.shortest_path.geodesic_distance

    def display_shortest_path(self):
        # render top down map
        top_down_map = self.top_down_map.copy()
        grid_dimensions = (top_down_map.shape[0], top_down_map.shape[1])
        # convert world trajectory points to maps module grid points
        trajectory = [
            maps.to_grid(
                path_point[2],
                path_point[0],
                grid_dimensions,
                pathfinder=self.simulator.pathfinder,
            )
            for path_point in self.shortest_path.points
        ]
        grid_tangent = mn.Vector2(
            trajectory[1][1] - trajectory[0][1], trajectory[1][0] - trajectory[0][0]
        )
        path_initial_tangent = grid_tangent / grid_tangent.length()
        initial_angle = math.atan2(path_initial_tangent[0], path_initial_tangent[1])
        # draw the agent and trajectory on the map
        maps.draw_path(top_down_map, trajectory)
        maps.draw_agent(top_down_map, trajectory[0], initial_angle, agent_radius_px=8)
        return top_down_map

    @staticmethod
    def crop_local_map(agent_loc, global_map, size):
        # get the size of the global map
        r, c, _ = global_map.shape

        # compute the boundary
        crop_from_r = agent_loc[0] - size if agent_loc[0] - size > 0 else 0
        crop_from_c = agent_loc[1] - size if agent_loc[1] - size > 0 else 0

        crop_to_r = agent_loc[0] + size if agent_loc[0] + size < r else r
        crop_to_c = agent_loc[1] + size if agent_loc[1] + size < c else c

        # crop the map
        local_map = global_map[crop_from_r:crop_to_r, crop_from_c:crop_to_c, :]

        return local_map

    # Customize the maps.to_grid_func()
    def map_2d_to_grid_func(self, position, grid_size):
        """
        Grid coords: top left origin: (0, 0) -> (r, c)
        Real world coords: lower left origin:
        (position[0]_min (horizontal), position[2]_min (vertical)) -->
        (position[0]_max (horizontal), position[2]_max (vertical))
        """
        # Obtain the real world range
        lower_bound, upper_bound = self.simulator.pathfinder.get_bounds()

        # Compute the real world dimension in meters
        world_height = abs(upper_bound[2] - lower_bound[2])
        world_width = abs(upper_bound[0] - lower_bound[0])

        # Compute the cell size in the grid
        grid_height = grid_size[0]
        grid_width = grid_size[1]
        grid_cell_size = (
            abs(world_height) / grid_height,
            abs(world_width) / grid_width,
        )

        # Agent position in the real world
        real_world_row = position[2]
        real_world_col = position[0]

        grid_row = int((real_world_row - lower_bound[2]) / grid_cell_size[0])
        grid_col = int((real_world_col - lower_bound[0]) / grid_cell_size[1])

        return grid_row, grid_col

    def draw_waypoint_on_top_down_map(self, waypoint, top_down_map):
        # get the boundary of the environment
        grid_dimensions = (top_down_map.shape[0], top_down_map.shape[1])

        # convert the world positions to map locations
        waypoint_map_pose = maps.to_grid(
            waypoint[2],
            waypoint[0],
            grid_dimensions,
            pathfinder=self.simulator.pathfinder,
        )

        # plot results
        plot_waypoint = top_down_map.copy()
        goal_off_set = 10
        plot_waypoint[
            waypoint_map_pose[0] - goal_off_set : waypoint_map_pose[0] + goal_off_set,
            waypoint_map_pose[1] - goal_off_set : waypoint_map_pose[1] + goal_off_set,
            :,
        ] = 0.7

        return plot_waypoint

    def draw_all_waypoints_on_top_down_map(self, waypoints_list, top_down_map):
        # Get the boundary of the environment
        grid_dimensions = (top_down_map.shape[0], top_down_map.shape[1])

        # Plot the results
        plot_waypoint = top_down_map.copy()
        # Plot all waypoints
        for i in range(1, len(waypoints_list) - 1, 1):
            # Next waypoint
            waypoint = waypoints_list[i]

            # Convert the world positions to map locations
            waypoint_map_pose = maps.to_grid(
                waypoint[2],
                waypoint[0],
                grid_dimensions,
                pathfinder=self.simulator.pathfinder,
            )

            # Plot results
            goal_off_set = 15
            plot_waypoint[
                waypoint_map_pose[0]
                - goal_off_set : waypoint_map_pose[0]
                + goal_off_set,
                waypoint_map_pose[1]
                - goal_off_set : waypoint_map_pose[1]
                + goal_off_set,
                :,
            ] = 0.7

        return plot_waypoint

    def draw_agent_on_top_down_map(self):
        # get the boundary of the environment
        top_down_map = self.top_down_map_traj.copy()
        grid_dimensions = (top_down_map.shape[0], top_down_map.shape[1])

        # convert the world positions to map locations
        agent_map_pos = maps.to_grid(
            self.agent.state.position[2],
            self.agent.state.position[0],
            grid_dimensions,
            pathfinder=self.simulator.pathfinder,
        )

        # plot the agent's path in dashed red color
        if self.plot_traj:
            top_down_map[
                agent_map_pos[0] - 5 : agent_map_pos[0] + 5,
                agent_map_pos[1] - 5 : agent_map_pos[1] + 5,
                0,
            ] = 255
            top_down_map[
                agent_map_pos[0] - 5 : agent_map_pos[0] + 5,
                agent_map_pos[1] - 5 : agent_map_pos[1] + 5,
                1,
            ] = 100
            top_down_map[
                agent_map_pos[0] - 5 : agent_map_pos[0] + 5,
                agent_map_pos[1] - 5 : agent_map_pos[1] + 5,
                2,
            ] = 0

        # save the current trajectory
        self.top_down_map_traj = top_down_map.copy()

        # convert goal positions to map locations
        goal_map_pos = maps.to_grid(
            self.goal_loc[2],
            self.goal_loc[0],
            grid_dimensions,
            pathfinder=self.simulator.pathfinder,
        )

        # get the orientation and reference axis
        agent_angle, refer_axis = quat_to_angle_axis(self.agent.state.rotation)

        # compute the direction vector
        head_vector = [np.cos(agent_angle), 0, np.sin(agent_angle)]

        if np.array_equal(refer_axis, np.array([0, 1, 0])):
            # rotate along x axis
            rot_mat_x = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(-np.pi), -np.sin(-np.pi)],
                    [0, np.sin(-np.pi), np.cos(-np.pi)],
                ]
            )
            head_vector = np.matmul(rot_mat_x, head_vector)

        # rotate along z axis
        rot_mat_z = np.array(
            [
                [np.cos(np.pi), -np.sin(np.pi), 0],
                [np.sin(np.pi), np.cos(np.pi), 0],
                [0, 0, 1],
            ]
        )
        head_vector_map = np.matmul(rot_mat_z, head_vector)

        # compute the angle
        map_angle = np.arctan2(head_vector_map[2], head_vector_map[0])

        if self.map_show_agent:
            agent_radius = 15
            # draw the agent on the map
            maps.draw_agent(
                top_down_map, agent_map_pos, map_angle, agent_radius_px=agent_radius
            )  # the agent will be resize a region with radius = 5 pixelsex

            # draw the goal on the map
            if self.map_show_goal:
                goal_off_set = 25
                top_down_map[
                    goal_map_pos[0] - goal_off_set : goal_map_pos[0] + goal_off_set,
                    goal_map_pos[1] - goal_off_set : goal_map_pos[1] + goal_off_set,
                    :,
                ] = 0.7

        # ============================================================
        # todo: for debug only
        # record the top-down view
        self.top_down_visualization = top_down_map.copy()
        agent_radius = 25
        # draw the agent on the map
        maps.draw_agent(
            self.top_down_visualization,
            agent_map_pos,
            map_angle,
            agent_radius_px=agent_radius,
        )  # the agent will be resize a region with radius = 5 pixelsex

        goal_off_set = 25
        self.top_down_visualization[
            goal_map_pos[0] - goal_off_set : goal_map_pos[0] + goal_off_set,
            goal_map_pos[1] - goal_off_set : goal_map_pos[1] + goal_off_set,
            :,
        ] = 0.7
        # ============================================================

        # crop the local map out if the local map flag is True
        if self.enable_local_map:
            top_down_map = self.crop_local_map(
                agent_loc=agent_map_pos,
                global_map=top_down_map,
                size=self.local_map_size,
            )

            if (
                self.enable_ego_local_map
            ):  # if True, rotate the local map based on agent orientation
                # rotate the local map into egocentric view
                rot_control = int(np.round(map_angle / (np.pi / 2)))
                if rot_control == 0:
                    top_down_map = np.rot90(top_down_map, k=2)
                elif rot_control == 1:
                    top_down_map = np.rot90(top_down_map, k=1)
                elif rot_control == -1:
                    top_down_map = np.rot90(top_down_map, k=3)
                else:
                    pass

                # check the size of the top-down map
                target_size = (self.local_map_size * 2, self.local_map_size * 2, 3)
                if top_down_map.shape != target_size:
                    top_down_map = resize(top_down_map, target_size)

        return top_down_map

    def display_observations(self, obs, t):
        if t == 0:  # create display artists for all sensors
            view_num = len(self.sensor_names)
            self.fig, self.arr = plt.subplots(1, view_num, figsize=(12, 6))
            self.arr = [self.arr] if view_num == 1 else self.arr
            # create the plot artists
            for i, name in enumerate(self.sensor_names):
                # remove the axis labels
                self.arr[i].axis("off")
                # add title
                self.arr[i].set_title(name)
                if name == "color_sensor":
                    self.artists.append(self.arr[i].imshow(obs[name][:, :, 0:3]))
                elif name == "depth_sensor":
                    if self.depth_clip_vmax:
                        self.artists.append(self.arr[i].imshow(obs[name]))
                    else:
                        self.artists.append(self.arr[i].imshow(obs[name]))
                elif name == "top_down":
                    self.artists.append(self.arr[i].imshow(obs[name]))
                elif name == "semantic_sensor":
                    semantic_img = Image.new(
                        "P", (obs[name].shape[1], obs[name].shape[0])
                    )
                    semantic_img.putpalette(d3_40_colors_rgb.flatten())
                    semantic_img.putdata((obs[name].flatten() % 40).astype(np.uint8))
                    semantic_img = semantic_img.convert("RGBA")
                    self.artists.append(self.arr[i].imshow(semantic_img))
                else:
                    raise Exception(f"Invalid observation name {name}")
        else:
            for i, name in enumerate(self.sensor_names):
                if name == "color_sensor":
                    self.artists[i].set_data(obs[name][:, :, 0:3])
                elif name == "depth_sensor":
                    self.artists[i].set_data(obs[name])
                elif name == "top_down":
                    self.artists[i].set_data(obs[name])
                elif name == "semantic_sensor":
                    semantic_img = Image.new(
                        "P", (obs[name].shape[1], obs[name].shape[0])
                    )
                    semantic_img.putpalette(d3_40_colors_rgb.flatten())
                    semantic_img.putdata((obs[name].flatten() % 40).astype(np.uint8))
                    semantic_img = semantic_img.convert("RGBA")
                    self.artists[i].set_data(semantic_img)

        self.fig.canvas.draw()
        plt.pause(0.1)

    def display_panorama(self, obs, t, obs_type):
        if t == 0:
            if not self.enable_rgb_depth:
                # create the figure
                self.fig, self.arr = plt.subplots(3, 3, figsize=(12, 12))
                # remove the axis labels
                for m in range(3):
                    for n in range(3):
                        self.arr[m, n].axis("off")

                # built-in function
                def create_artist(plot_arr, r, c, obs_name):
                    # set title of the plot
                    plot_arr[r, c].set_title(obs_name)
                    # create the artist for subplot
                    if obs_type == "color_sensor":
                        return plot_arr[r, c].imshow(obs[obs_name][:, :, 0:3])
                    elif obs_type == "depth_sensor":
                        if self.depth_clip_vmax:
                            return plot_arr[r, c].imshow(obs[obs_name])
                        else:
                            return plot_arr[r, c].imshow(obs[obs_name])
                    elif obs_type == "semantic_sensor":
                        _img = Image.new(
                            "P", (obs[obs_name].shape[1], obs[obs_name].shape[0])
                        )
                        _img.putpalette(d3_40_colors_rgb.flatten())
                        _img.putdata((obs[obs_name].flatten() % 40).astype(np.uint8))
                        _img = _img.convert("RGBA")
                        return plot_arr[r, c].imshow(_img)
                    else:
                        raise Exception(f"Invalid observation name {obs_type}")

                # add title
                self.artists.append(create_artist(self.arr, 0, 1, obs_type + "_front"))
                self.artists.append(create_artist(self.arr, 1, 0, obs_type + "_left"))
                self.artists.append(create_artist(self.arr, 1, 2, obs_type + "_right"))
                self.artists.append(create_artist(self.arr, 2, 1, obs_type + "_back"))

                # add top down observation
                self.artists.append(self.arr[1, 1].imshow(obs["top_down"]))
            else:
                # create the figure
                self.fig, self.arr = plt.subplots(4, 3, figsize=(12, 12))
                # remove the axis labels
                for m in range(4):
                    for n in range(3):
                        self.arr[m, n].axis("off")

                # add title
                self.arr[0, 0].set_title("color_front")
                self.arr[0, 1].set_title("depth_front")
                self.artists.append(self.arr[0, 0].imshow(obs["color_sensor_front"]))
                self.artists.append(self.arr[0, 1].imshow(obs["depth_sensor_front"]))

                self.arr[1, 0].set_title("color_left")
                self.arr[1, 1].set_title("depth_left")
                self.artists.append(self.arr[1, 0].imshow(obs["color_sensor_left"]))
                self.artists.append(self.arr[1, 1].imshow(obs["depth_sensor_left"]))

                self.arr[2, 0].set_title("color_right")
                self.arr[2, 1].set_title("depth_right")
                self.artists.append(self.arr[2, 0].imshow(obs["color_sensor_right"]))
                self.artists.append(self.arr[2, 1].imshow(obs["depth_sensor_right"]))

                self.arr[3, 0].set_title("color_back")
                self.arr[3, 1].set_title("depth_back")
                self.artists.append(self.arr[3, 0].imshow(obs["color_sensor_back"]))
                self.artists.append(self.arr[3, 1].imshow(obs["depth_sensor_back"]))

                # add top down observation
                self.arr[1, 2].set_title("top_down")
                self.artists.append(self.arr[1, 2].imshow(obs["top_down"]))
        else:
            if not self.enable_rgb_depth:
                # update the artists
                def update_artist(artists, idx, obs_name):
                    if obs_type == "color_sensor":
                        artists[idx].set_data(obs[obs_name][:, :, 0:3])
                    elif obs_type == "depth_sensor":
                        artists[idx].set_data(obs[obs_name])
                    elif obs_type == "semantic_sensor":
                        _img = Image.new(
                            "P", (obs[obs_name].shape[1], obs[obs_name].shape[0])
                        )
                        _img.putpalette(d3_40_colors_rgb.flatten())
                        _img.putdata((obs[obs_name].flatten() % 40).astype(np.uint8))
                        _img = _img.convert("RGBA")
                        artists[idx].set_data(_img)
                    else:
                        raise Exception(f"Invalid observation name {obs_type}")

                # update the plot with the latest offline_dataset
                update_artist(self.artists, 0, obs_type + "_front")
                update_artist(self.artists, 1, obs_type + "_left")
                update_artist(self.artists, 2, obs_type + "_right")
                update_artist(self.artists, 3, obs_type + "_back")

                # update the top down view
                self.artists[-1].set_data(obs["top_down"])
            else:
                self.artists[0].set_data(obs["color_sensor_front"])
                self.artists[1].set_data(obs["depth_sensor_front"])

                self.artists[2].set_data(obs["color_sensor_left"])
                self.artists[3].set_data(obs["depth_sensor_left"])

                self.artists[4].set_data(obs["color_sensor_right"])
                self.artists[5].set_data(obs["depth_sensor_right"])

                self.artists[6].set_data(obs["color_sensor_back"])
                self.artists[7].set_data(obs["depth_sensor_back"])

                # add top down observation
                self.artists[8].set_data(obs["top_down"])

        self.fig.canvas.draw()
        plt.pause(0.1)

    # display a top-down map
    def display_top_down_map(self):
        plt.figure(figsize=(12, 8))
        ax = plt.subplot(1, 1, 1)
        ax.axis("off")
        if len(self.top_down_map.shape) != 3:
            rows, cols = self.top_down_map.shape
        else:
            rows, cols, _ = self.top_down_map.shape
        plt.title(f"Size {rows} x {cols}")
        plt.imshow(self.top_down_map)
        plt.show()

    # rgb2gray
    @staticmethod
    def rgb2gray(obs):
        return rgb2gray(obs)

    def is_terminal(self, loc):
        # convert to numpy array
        agent_loc = np.array([loc[0], loc[2]])  # x, z
        goal_loc = np.array([self.goal_loc[0], self.goal_loc[2]])  # x, z

        # compute the geometric distance: 2-norm
        self.agent_goal_dist = np.linalg.norm((agent_loc - goal_loc), ord=2)

        # check the termination
        done = True if self.agent_goal_dist <= self.dist_eps else False
        # check the timeout
        timeout = True if self.step_counter == self.max_episode_length else False

        # return termination flag
        return done, timeout

    # gym close function
    def close(self):
        self.simulator.close()
