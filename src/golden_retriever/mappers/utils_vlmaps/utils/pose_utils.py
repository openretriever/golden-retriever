import numpy as np
import quaternion
from habitat.utils.geometry_utils import quaternion_rotate_vector

ANGLE_EPS = 0.001


def normalize(v):
    return v / np.linalg.norm(v)


def get_r_matrix(ax_, angle):
    ax = normalize(ax_)
    if np.abs(angle) > ANGLE_EPS:
        S_hat = np.array(
            [[0.0, -ax[2], ax[1]], [ax[2], 0.0, -ax[0]], [-ax[1], ax[0], 0.0]],
            dtype=np.float32,
        )
        R = (
            np.eye(3)
            + np.sin(angle) * S_hat
            + (1 - np.cos(angle)) * (np.linalg.matrix_power(S_hat, 2))
        )
    else:
        R = np.eye(3)
    return R


def r_between(v_from_, v_to_):
    v_from = normalize(v_from_)
    v_to = normalize(v_to_)
    ax = normalize(np.cross(v_from, v_to))
    angle = np.arccos(np.dot(v_from, v_to))
    return get_r_matrix(ax, angle)


def rotate_camera_to_point_at(up_from, lookat_from, up_to, lookat_to):
    inputs = [up_from, lookat_from, up_to, lookat_to]
    for i in range(4):
        inputs[i] = normalize(np.array(inputs[i]).reshape((-1,)))
    up_from, lookat_from, up_to, lookat_to = inputs
    r1 = r_between(lookat_from, lookat_to)

    new_x = np.dot(r1, np.array([1, 0, 0]).reshape((-1, 1))).reshape((-1))
    to_x = normalize(np.cross(lookat_to, up_to))
    angle = np.arccos(np.dot(new_x, to_x))
    if angle > ANGLE_EPS:
        if angle < np.pi - ANGLE_EPS:
            ax = normalize(np.cross(new_x, to_x))
            flip = np.dot(lookat_to, ax)
            if flip > 0:
                r2 = get_r_matrix(lookat_to, angle)
            elif flip < 0:
                r2 = get_r_matrix(lookat_to, -1.0 * angle)
        else:
            # Angle of rotation is too close to 180 degrees, direction of rotation
            # does not matter.
            r2 = get_r_matrix(lookat_to, angle)
    else:
        r2 = np.eye(3)
    return np.dot(r2, r1)


def transform_sim_pose_2_xyo(pose):
    """
    Habitat coordinate frame:
        X to the right
        Y to the up
        Z out of the screen
        Agent is on XZ plane

    New coordinate frame:
        X in to the screen
        Y to the left
        Z to the up

    Return:
        x, y in meters
        o in radius [-np.pi, np.pi]
    """
    # Position in Habitat simulator
    position = pose[0:3]

    # Orientation in quaternion (w, x, y, z)
    rotation = quaternion.from_float_array(np.array(pose[3:], dtype=np.float))

    # Convert the Habitat frame to
    x = -position[2]  # Agent front
    y = -position[0]  # Agent left

    # Convert the rotation from quaternion to euler angle following the order ZYX (yaw, pitch, and roll)
    axis = quaternion.as_euler_angles(rotation)[
        0
    ]  # return the rotation w.r.t. x, y, z in Habitat

    if (axis % (2 * np.pi)) < 0.1 or (
        axis % (2 * np.pi)
    ) > 2 * np.pi - 0.1:  # the value 0.1 seems to be empirical
        o = quaternion.as_euler_angles(rotation)[1]  # [-2*np.pi, 2*np.pi]
    else:
        o = 2 * np.pi - quaternion.as_euler_angles(rotation)[1]  # align the rotation

    if o > np.pi:
        o -= 2 * np.pi  # normalize the value between [-np.pi, np.pi]

    return np.array([x, y, o], dtype=np.float32)


def get_l2_distance(x1, x2, y1, y2):
    """
    Computes the L2 distance between two points.
    """
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def get_rel_pos_change(pos1, pos2):
    x1, y1, o1 = pos1
    x2, y2, o2 = pos2

    # Change the orientation to the agents's local frame
    theta = np.arctan2(y2 - y1, x2 - x1) - o1

    dist = get_l2_distance(x1, x2, y1, y2)

    dx = dist * np.cos(theta)
    dy = dist * np.sin(theta)

    do = o2 - o1

    return dx, dy, do


def get_global_map_pose(pose, rel_pose_change):
    x, y, o = pose
    dx, dy, do = rel_pose_change

    global_dx = dx * np.sin(np.deg2rad(o)) + dy * np.cos(np.deg2rad(o))
    global_dy = dx * np.cos(np.deg2rad(o)) - dy * np.sin(np.deg2rad(o))

    x += global_dy
    y += global_dx
    o += np.rad2deg(do)

    if o > 180.0:
        o -= 360.0

    return x, y, o


def convert_pose_map2polar(map_source_pose, map_target_pose):
    """DD-PPO requires PointGoal represented in egocentric polar coordinates (rho, phi)
    Egocentric polar:
        - Heading direction: 0
        - To the right of the heading direction: -0 -> -np.pi
        - To the left of the heading direction: +0 -> np.pi
    """
    # Source pose
    x_1, y_1, o_1 = map_source_pose
    # Target pose
    x_2, y_2 = map_target_pose

    # Compute the offsets
    dx = x_2 - x_1
    dy = y_2 - y_1

    # Compute the rho
    rho = (dx**2 + dy**2) ** 0.5
    # Compute the phi
    phi = np.arctan2(dy, dx) - np.deg2rad(o_1)

    # Constrain the phi within [-np.pi, np.pi]
    if phi > np.pi:
        phi -= 2 * np.pi
    if phi < -np.pi:
        phi += 2 * np.pi

    return np.array([rho, phi], dtype=np.float32)


def get_rot_and_trans_matrices(batch_pose1, batch_pose2, batch_last_map_poses):
    """Compute the rotation matrix and the translation matrix
    Pose1 is the last pose;
    Pose2 is the current pose
    """
    rot_mat_list, trans_mat_list, curr_map_pose_list = [], [], []
    for pose1, pose2, last_map_pose in zip(
        batch_pose1, batch_pose2, batch_last_map_poses, strict=False
    ):
        # Convert Sim Pose --> XYO Pose
        last_xyo_pose = transform_sim_pose_2_xyo(pose1)
        curr_xyo_pose = transform_sim_pose_2_xyo(pose2)

        # Compute the relative change of the xyo pose
        dx, dy, do = get_rel_pos_change(last_xyo_pose, curr_xyo_pose)

        # Compute the current map pose
        curr_map_pose = get_global_map_pose(last_map_pose, (dx, dy, do))
        curr_map_pose_cm = (
            curr_map_pose[0] * 100.0,
            curr_map_pose[1] * 100.0,
            np.deg2rad(curr_map_pose[2]),
        )

        # Compute the rotation matrix
        rotation_matrix = get_r_matrix(
            [0.0, 0.0, 1.0], angle=curr_map_pose_cm[2] - np.pi / 2.0
        )
        # Compute the translation matrix
        translation_matrix = np.array(curr_map_pose_cm[0:2])

        # Save the rotation and translation matrix
        rot_mat_list.append(rotation_matrix.T)
        trans_mat_list.append(translation_matrix)
        curr_map_pose_list.append(curr_map_pose)

    return (
        np.array(rot_mat_list),
        np.array(trans_mat_list),
        np.array(curr_map_pose_list),
    )


def convert_pose_sim2map(
    curr_sim_pose: list,
    init_sim_pose: list,
    init_map_pose: list,
    map_resolution_cm: int,
) -> dict:
    """
    Convert the current simulator pose to the pose on the map.

    Habitat coordinates (right-handed):
        - X to the right
        - Y to the up
        - Z out of the screen

    Map coordinates (left-handed):
        - X to the right
        - Y to the down
        - Z out of the screen

    :param curr_sim_pose: Current pose in the simulator
    :param init_sim_pose: Initial pose in the simulator
    :param init_map_pose: By default, the agents is always initialized as follows:
        - Position: In the middle of the map = (12.0, 12.0)
        - Rotation: Facing to the right = 0.0 in radius
    :param map_resolution_cm: resolution of the map
    :return: Current pose on the map
            - grid_loc: row and col indices on the map
            - meter_loc: x and y in meters on the map
    """
    # Transform: Habitat ----> Map (absolute)
    init_pose_xyo = transform_sim_pose_2_xyo(init_sim_pose)
    curr_pose_xyo = transform_sim_pose_2_xyo(curr_sim_pose)

    # Transform: Map (absolute) ----> Map (relative)
    dx, dy, do = get_rel_pos_change(curr_pose_xyo, init_pose_xyo)

    # Update the current map pose: current pose + relative pose (in meters)
    wp_map_pose = get_global_map_pose(init_map_pose, (dx, dy, do))
    x = wp_map_pose[0] * 100.0 / map_resolution_cm
    y = wp_map_pose[1] * 100.0 / map_resolution_cm

    # Compute the row and column
    row_idx, col_idx = int(y), int(x)

    return {"grid_loc": (row_idx, col_idx), "meter_loc": wp_map_pose}


def cartesian_to_polar(x, y):
    """Convert Cartesian to Polar"""
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi


def compute_gps_and_compass_point_goal(
    source_position: np.ndarray, source_rotation: np.ndarray, goal_position: np.ndarray
):
    """
    GPS+Compass.

    For the agents in simulator the forward direction is along negative-z.
    In polar coordinate format the angle returned is azimuth to the goal.

    Args:
        source_position: agents's position [x, y, z] in meters
        source_rotation: agents's rotation [w, x, y, z] in quaternion
        goal_position: goal position [x, y, z] in meters
    Return:
        numpy.ndarray: represent in polar coordinates.
    The relative goal to the agents's current pose represented in the polar coordinate
    """
    direction_vector = goal_position - source_position
    direction_vector_agent = quaternion_rotate_vector(
        source_rotation.inverse(), direction_vector
    )

    rho, phi = cartesian_to_polar(-direction_vector_agent[2], direction_vector_agent[0])

    return np.array([rho, -phi], dtype=np.float32)
