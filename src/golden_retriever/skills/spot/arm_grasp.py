# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""Tutorial to show how to use Spot's arm.
"""
import argparse
import math
import sys
import time

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util
import cv2
import numpy as np
from bosdyn.api import estop_pb2, geometry_pb2, manipulation_api_pb2
from bosdyn.client.estop import EstopClient
from bosdyn.client.frame_helpers import (
    VISION_FRAME_NAME,
    get_vision_tform_body,
    math_helpers,
)
from bosdyn.client.image import ImageClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import (
    RobotCommandBuilder,
    RobotCommandClient,
    blocking_stand,
)
from bosdyn.client.robot_state import RobotStateClient

from retriever.skills.spot.automated_skills.image import *

def verify_estop(robot):
    """Verify the robot is not estopped"""

    client = robot.ensure_client(EstopClient.default_service_name)
    if client.get_status().stop_level != estop_pb2.ESTOP_LEVEL_NONE:
        error_message = (
            "Robot is estopped. Please use an external E-Stop client, such as the"
            " estop SDK example, to configure E-Stop."
        )
        robot.logger.error(error_message)
        raise Exception(error_message)

def arm_object_pick(config, robot, image, pick_vec):
    """A simple example of using the Boston Dynamics API to command Spot's arm."""
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    manipulation_api_client = robot.ensure_client(
        ManipulationApiClient.default_service_name
    )

    # Now, we are ready to power on the robot. This call will block until the power
    # is on. Commands would fail if this did not happen. We can also check that the robot is
    # powered at any point.
    robot.logger.info("Powering on robot... This may take a several seconds.")
    robot.power_on(timeout_sec=20)
    assert robot.is_powered_on(), "Robot power on failed."
    robot.logger.info("Robot powered on.")

    # Tell the robot to stand up. The command service is used to issue commands to a robot.
    # The set of valid commands for a robot depends on hardware configuration. See
    # RobotCommandBuilder for more detailed examples on command building. The robot
    # command service requires timesync between the robot and the client.
    robot.logger.info("Commanding robot to stand...")
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    blocking_stand(command_client, timeout_sec=10)
    robot.logger.info("Robot standing.")

    # Build the proto
    pick_vec = geometry_pb2.Vec2(x=pick_vec[0], y=pick_vec[1])
    grasp = manipulation_api_pb2.PickObjectInImage(
        pixel_xy=pick_vec,
        transforms_snapshot_for_camera=image.shot.transforms_snapshot,
        frame_name_image_sensor=image.shot.frame_name_image_sensor,
        camera_model=image.source.pinhole,
    )

    # Optionally add a grasp constraint
    add_grasp_constraint(config, grasp, robot_state_client)

    # Ask the robot to pick up the object
    grasp_request = manipulation_api_pb2.ManipulationApiRequest(
        pick_object_in_image=grasp
    )

    # Send the request
    # Ask the robot to pick up the object
    grasp_request = manipulation_api_pb2.ManipulationApiRequest(
        pick_object_in_image=grasp
    )

    # Send the request
    cmd_response = manipulation_api_client.manipulation_api_command(
        manipulation_api_request=grasp_request
    )

    # Get feedback from the robot
    while True:
        feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
            manipulation_cmd_id=cmd_response.manipulation_cmd_id
        )

        # Send the request
        response = manipulation_api_client.manipulation_api_feedback_command(
            manipulation_api_feedback_request=feedback_request
        )

        print(
            f"Current state: {manipulation_api_pb2.ManipulationFeedbackState.Name(response.current_state)}"
        )

        if (
            response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED
            or response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_FAILED
        ):
            carry_cmd = RobotCommandBuilder.arm_carry_command()
            command_client.robot_command(carry_cmd)
            time.sleep(2)
            break

        time.sleep(0.25)

        robot.logger.info("Finished grasp.")
        time.sleep(4.0)


def add_grasp_constraint(config, grasp, robot_state_client):
    # There are 3 types of constraints:
    #   1. Vector alignment
    #   2. Full rotation
    #   3. Squeeze grasp
    #
    # You can specify more than one if you want and they will be OR'ed together.

    # For these options, we'll use a vector alignment constraint.
    use_vector_constraint = config.force_top_down_grasp or config.force_horizontal_grasp

    # Specify the frame we're using.
    grasp.grasp_params.grasp_params_frame_name = VISION_FRAME_NAME

    if use_vector_constraint:
        if config.force_top_down_grasp:
            # Add a constraint that requests that the x-axis of the gripper is pointing in the
            # negative-z direction in the vision frame.

            # The axis on the gripper is the x-axis.
            axis_on_gripper_ewrt_gripper = geometry_pb2.Vec3(x=1, y=0, z=0)

            # The axis in the vision frame is the negative z-axis
            axis_to_align_with_ewrt_vo = geometry_pb2.Vec3(x=0, y=0, z=-1)

        if config.force_horizontal_grasp:
            # Add a constraint that requests that the y-axis of the gripper is pointing in the
            # positive-z direction in the vision frame.  That means that the gripper is constrained to be rolled 90 degrees and pointed at the horizon.

            # The axis on the gripper is the y-axis.
            axis_on_gripper_ewrt_gripper = geometry_pb2.Vec3(x=0, y=1, z=0)

            # The axis in the vision frame is the positive z-axis
            axis_to_align_with_ewrt_vo = geometry_pb2.Vec3(x=0, y=0, z=1)

        # Add the vector constraint to our proto.
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.vector_alignment_with_tolerance.axis_on_gripper_ewrt_gripper.CopyFrom(
            axis_on_gripper_ewrt_gripper
        )
        constraint.vector_alignment_with_tolerance.axis_to_align_with_ewrt_frame.CopyFrom(
            axis_to_align_with_ewrt_vo
        )

        # We'll take anything within about 10 degrees for top-down or horizontal grasps.
        constraint.vector_alignment_with_tolerance.threshold_radians = 0.17

    elif config.force_45_angle_grasp:
        # Demonstration of a RotationWithTolerance constraint.  This constraint allows you to
        # specify a full orientation you want the hand to be in, along with a threshold.
        #
        # You might want this feature when grasping an object with known geometry and you want to
        # make sure you grasp a specific part of it.
        #
        # Here, since we don't have anything in particular we want to grasp,  we'll specify an
        # orientation that will have the hand aligned with robot and rotated down 45 degrees as an
        # example.

        # First, get the robot's position in the world.
        robot_state = robot_state_client.get_robot_state()
        vision_T_body = get_vision_tform_body(
            robot_state.kinematic_state.transforms_snapshot
        )

        # Rotation from the body to our desired grasp.
        body_Q_grasp = math_helpers.Quat.from_pitch(0.785398)  # 45 degrees
        vision_Q_grasp = vision_T_body.rotation * body_Q_grasp

        # Turn into a proto
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.rotation_with_tolerance.rotation_ewrt_frame.CopyFrom(
            vision_Q_grasp.to_proto()
        )

        # We'll accept anything within +/- 10 degrees
        constraint.rotation_with_tolerance.threshold_radians = 0.17

    elif config.force_squeeze_grasp:
        # Tell the robot to just squeeze on the ground at the given point.
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.squeeze_grasp.SetInParent()


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument(
        "-t",
        "--force-top-down-grasp",
        help="Force the robot to use a top-down grasp (vector_alignment demo)",
        action="store_true",
    )
    parser.add_argument(
        "-f",
        "--force-horizontal-grasp",
        help="Force the robot to use a horizontal grasp (vector_alignment demo)",
        action="store_true",
    )
    parser.add_argument(
        "-r",
        "--force-45-angle-grasp",
        help="Force the robot to use a 45 degree angled down grasp (rotation_with_tolerance demo)",
        action="store_true",
    )
    parser.add_argument(
        "-s",
        "--force-squeeze-grasp",
        help="Force the robot to use a squeeze grasp",
        action="store_true",
    )
    options = parser.parse_args()

    num = 0
    if options.force_top_down_grasp:
        num += 1
    if options.force_horizontal_grasp:
        num += 1
    if options.force_45_angle_grasp:
        num += 1
    if options.force_squeeze_grasp:
        num += 1

    if num > 1:
        print("Error: cannot force more than one type of grasp.  Choose only one.")
        sys.exit(1)

    try:
        config = options
        bosdyn.client.util.setup_logging(config.verbose)

        sdk = bosdyn.client.create_standard_sdk("ArmObjectPickClient")
        robot = sdk.create_robot(config.hostname)
        bosdyn.client.util.authenticate(robot)
        robot.time_sync.wait_for_sync()

        assert robot.has_arm(), "Robot requires an arm to run this skill."

        # Verify the robot is not estopped and that an external application has registered and holds
        # an estop endpoint.
        verify_estop(robot)

        lease_client = robot.ensure_client(
            bosdyn.client.lease.LeaseClient.default_service_name
        )
        lease_client.take()
        # image_client = robot.ensure_client(ImageClient.default_service_name)

        #sources = ["frontleft_fisheye_image", "frontright_fisheye_image"]
        sources = ["left_fisheye_image", "right_fisheye_image"]
        image_dict = get_images_as_cv2(robot, sources)

        window_name = "Grasp Object"
        request_manager = ImageSideBySideManager(image_dict, window_name)
        request_manager.get_user_input()
        assert request_manager.user_input_set(), "Failed to get input from user."

        pick_vec = request_manager.get_selected_pixel()
        image = request_manager.image_dict[request_manager.clicked_source][0]
        arm_object_pick(options, robot, image, pick_vec)
        return True
    except Exception:  # pylint: disable=broad-except
        logger = bosdyn.client.util.get_logger()
        logger.exception("Threw an exception")
        return False


if __name__ == "__main__":
    if not main():
        sys.exit(1)
