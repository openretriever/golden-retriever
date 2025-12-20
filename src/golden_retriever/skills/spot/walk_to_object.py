# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""Tutorial to show how to walk the robot to an object, usually in preparation for manipulation.
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
from bosdyn.api import geometry_pb2, manipulation_api_pb2
from bosdyn.client.image import ImageClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import RobotCommandClient, blocking_stand
from google.protobuf import wrappers_pb2
from retriever.skills.spot.automated_skills.image import *

def walk_to_object(config, robot, image, move_point):
    """Get an image and command the robot to walk up to a selected object.
    We'll walk "up to" the object, not on top of it.  The idea is that you
    want to interact or manipulate the object."""

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

    walk_vec = move_point
    walk_vec = geometry_pb2.Vec2(x=walk_vec[0], y=walk_vec[1])

    # Optionally populate the offset distance parameter.
    if config.distance is None:
        offset_distance = None
    else:
        offset_distance = wrappers_pb2.FloatValue(value=config.distance)

    # Build the proto
    walk_to = manipulation_api_pb2.WalkToObjectInImage(
        pixel_xy=walk_vec,
        transforms_snapshot_for_camera=image.shot.transforms_snapshot,
        frame_name_image_sensor=image.shot.frame_name_image_sensor,
        camera_model=image.source.pinhole,
        offset_distance=offset_distance,
    )

    # Ask the robot to pick up the object
    walk_to_request = manipulation_api_pb2.ManipulationApiRequest(
        walk_to_object_in_image=walk_to
    )

    # Send the request
    cmd_response = manipulation_api_client.manipulation_api_command(
        manipulation_api_request=walk_to_request
    )

    # Get feedback from the robot
    while True:
        time.sleep(0.25)
        feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
            manipulation_cmd_id=cmd_response.manipulation_cmd_id
        )

        # Send the request
        response = manipulation_api_client.manipulation_api_feedback_command(
            manipulation_api_feedback_request=feedback_request
        )

        print(
            "Current state: ",
            manipulation_api_pb2.ManipulationFeedbackState.Name(response.current_state),
        )

        if response.current_state == manipulation_api_pb2.MANIP_STATE_DONE:
            break


def arg_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{repr(x)} not a number")
    return x


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument(
        "-d",
        "--distance",
        help="Distance from object to walk to (meters).",
        default=None,
        type=arg_float,
    )
    options = parser.parse_args()

    try:
        config = options
        # See hello_spot.py for an explanation of these lines.
        bosdyn.client.util.setup_logging(config.verbose)

        sdk = bosdyn.client.create_standard_sdk("WalkToObjectClient")
        robot = sdk.create_robot(config.hostname)
        bosdyn.client.util.authenticate(robot)
        robot.time_sync.wait_for_sync()

        assert robot.has_arm(), "Robot requires an arm to run this example."

        # Verify the robot is not estopped and that an external application has registered and holds
        # an estop endpoint.
        assert not robot.is_estopped(), (
            "Robot is estopped. Please use an external E-Stop client, "
            "such as the estop SDK example, to configure E-Stop."
        )

        lease_client = robot.ensure_client(
            bosdyn.client.lease.LeaseClient.default_service_name
        )
        lease_client.take()
        # image_client = robot.ensure_client(ImageClient.default_service_name)

        sources = ["frontleft_fisheye_image", "frontright_fisheye_image"]
        image_dict = get_images_as_cv2(robot, sources)

        window_name = "Walk to Object"
        request_manager = ImageSideBySideManager(image_dict, window_name)
        request_manager.get_user_input()
        assert request_manager.user_input_set(), "Failed to get input from user."

        move_point = request_manager.get_selected_pixel()
        image = request_manager.image_dict[request_manager.clicked_source][0]
        walk_to_object(options, robot, image, move_point)
        return True
    except Exception:  # pylint: disable=broad-except
        logger = bosdyn.client.util.get_logger()
        logger.exception("Threw an exception")
        return False


if __name__ == "__main__":
    if not main():
        sys.exit(1)
