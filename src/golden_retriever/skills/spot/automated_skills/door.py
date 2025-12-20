from retriever.skills.spot.push_bar_door import open_door_main as open_door_push_bar, pitch_up, stand
from retriever.skills.spot.grasp_door import open_door_main as open_door_handle
from retriever.skills.spot.push_bar_door import RequestManager
from retriever.skills.spot.push_bar_door import power_on, stand, initialize_robot, check_estop
from bosdyn.client.util import add_base_arguments, authenticate, setup_logging
from bosdyn.client.lease import LeaseClient
from retriever.skills.spot.automated_skills.image import get_images_as_cv2

from bosdyn.api.spot import door_pb2

import argparse


import ray
from PIL import Image
from retriever.models.vlms.molmo_quantized_actor import MolmoQuantizedActor, draw_points
from retriever.models.api_models.utils.openai_utils import prepare_openai_Image_messages, call_openai_api
from glob import glob
from retriever.models.common_utils import timer, Timer
import time

import sys

def get_door_hinge_prompts():
    openai_prompt_prefix = "You are a world champion robot in opening doors. You have a very advance algorithm to calculate the push point of the given door. Given the push point your task is to determine whether the selected door opens to the left or the right."
    openai_prompt_suffix = "In the above image the push point has been shown by a blue dot and a white outline. Now tell me if the door swings open to the 'LEFT' or 'RIGHT'. Make sure to get the correct door. Only output 1 word(\'LEFT\' or \'RIGHT\')"
    return openai_prompt_prefix, openai_prompt_suffix

def open_grasp_door(options, molmo, robot, side_by_side, request_manager, model = "gpt-4o"):
    # Open the door by grasping the handle.
    molmo_prompt = "You are a world champion robot in opening doors. Point at the handle of the door where you want to grab to open the door. Return only ONE GRASP POINT ON THE METAL HANDLE. Make sure to find the ideal point to easily grab the handle to open the door and make sure its in the middle of the door handle."
    output = ray.get(molmo.predict.remote(side_by_side, molmo_prompt, points=True))
    points = output["points"]
    temp_with_handle = draw_points(side_by_side, points)

    request_manager.set_handle_position_side_by_side(points[0])
    openai_prompt_prefix, openai_prompt_suffix = get_door_hinge_prompts()
    input = prepare_openai_Image_messages(prefix=openai_prompt_prefix, suffix=openai_prompt_suffix, images = [temp_with_handle])
    with Timer(enable_print=True) as timer:
        output2 = call_openai_api(input, model=model, max_tokens=100, temperature=0.1)
        output2.lower()
    if output2 == "left":
        print("left was hinge output: overriding to right. All door hinges for pull doors are right on first floor or exp")
        hinge = door_pb2.DoorCommand.HINGE_SIDE_LEFT
    elif output2 == "right":
        hinge = door_pb2.DoorCommand.HINGE_SIDE_RIGHT
    else:
        hinge = door_pb2.DoorCommand.HINGE_SIDE_RIGHT
        print("WARNING DOOR HINGE SIDE UNDECIDED. DEFAULTING TO RIGHT")
    point = request_manager.get_walk_to_object_in_image_request()
    image = request_manager.image_dict[request_manager.clicked_source][0]

    open_door_handle(options, robot, image, point, hinge)


def open_push_door(options, molmo, robot, side_by_side, request_manager, model = "gpt-4o"):
    # Open the door by pushing it.
    molmo_prompt = "You are a world champion robot in opening doors. Point at the push bar of the door where you want to push to open the door. If it is a double door point at only the ONE PUSH POINT ON THE METAL PUSH BAR. DO NOT POINT BETWEEN THE DOUBLE DOORS POINT AT THE PUSH BAR. Make sure to find the ideal point to push to easily open the door and make sure its on the metal touch bar."
    output = ray.get(molmo.predict.remote(side_by_side, molmo_prompt, points=True))
    points = output["points"]
    temp_with_handle = draw_points(side_by_side, points)
    #temp_with_handle.show()
    print("points: ", points)
    request_manager.set_handle_position_side_by_side(points[0])
    openai_prompt_prefix, openai_prompt_suffix = get_door_hinge_prompts()
    input = prepare_openai_Image_messages(prefix=openai_prompt_prefix, suffix=openai_prompt_suffix, images = [temp_with_handle])
    with Timer(enable_print=True) as timer:
        output2 = call_openai_api(input, model=model, max_tokens=100, temperature=0.1)
        output2 = output2.lower()
    if output2 == "left":
        hinge = door_pb2.DoorCommand.HINGE_SIDE_LEFT
    elif output2 == "right":
        hinge = door_pb2.DoorCommand.HINGE_SIDE_RIGHT
    else:
        hinge = door_pb2.DoorCommand.HINGE_SIDE_RIGHT
        print("WARNING DOOR HINGE SIDE UNDECIDED. DEFAULTING TO RIGHT")
    point = request_manager.get_walk_to_object_in_image_request()
    image = request_manager.image_dict[request_manager.clicked_source][0]

    open_door_push_bar(options, robot, image, point, hinge)


def open_door(options, robot, molmo, sources = ['frontleft_fisheye_image', 'frontright_fisheye_image'], debug=False, model = "gpt-4o"):
    pitch_up(robot)
    image_dict = get_images_as_cv2(robot, sources)
    window_name = 'Open Door'
    request_manager = RequestManager(image_dict, window_name)
    # request_manager.get_user_input_handle_and_hinge()
    side_by_side = request_manager.side_by_side
    temp_side_by_side = side_by_side.copy()
    temp_side_by_side = Image.fromarray(temp_side_by_side)
    openai_prompt_prefix = ("You are a world champion robot in opening doors. You must figure out if the following door is a 'PUSH_BAR_DOOR' or a 'PULL_GRASP_HANDLE_DOOR'.\n"
                  "A 'PUSH_BAR_DOOR' has a metal push bar in the middle and is swings in the push direction.\n"
                  "A 'PULL_GRASP_HANDLE_DOOR' has a metal door handle handle that you will grasp/grab on and the door swings in the pull direction. Only choose this option if there isnt a push/touch bar on the door.\n")

    openai_prompt_suffix = "Only output 1 word ('PUSH_BAR_DOOR', 'PULL_GRASP_HANDLE_DOOR', or 'INVALID')."

    input = prepare_openai_Image_messages(prefix=openai_prompt_prefix, suffix=openai_prompt_suffix, images = [temp_side_by_side])
    with Timer(enable_print=True) as timer:
        output = call_openai_api(input, model=model, max_tokens=100, temperature=0.1)

    if output == 'PUSH_BAR_DOOR':
        print('PUSH_BAR_DOOR called')
        open_push_door(options, molmo, robot, temp_side_by_side, request_manager)
        return True
    elif output == 'PULL_GRASP_HANDLE_DOOR':
        print('PULL_GRASP_HANDLE_DOOR called')
        open_grasp_door(options, molmo, robot, temp_side_by_side, request_manager)
        return True
    else:
        print('Invalid door type.')
        return False

def main():
    ignore = glob("*/")
    ignore = [item for item in ignore if "src" not in item]
    ignore = ignore + glob("src/*/")
    ignore = [item for item in ignore if "models" not in item]
    ignore = ["\\" + item for item in ignore]
    ignore.append('\\.git\\')
    ignore = ignore + glob("src/robots/*/")
    
    parser = argparse.ArgumentParser(description=__doc__)
    add_base_arguments(parser)
    parser.add_argument('--debug', action='store_true', help='Show intermediate debug data.')
    parser.add_argument('--ray-server', type=str, default="ray://grail-mars.neu.edu:10001", help="Ray server address")

    options = parser.parse_args()

    ray.init(address=options.ray_server, runtime_env={"working_dir": ".", "excludes": ignore})

    use_gpu = True
    actor_options = {"num_gpus": 2} if use_gpu else {}
    molmo = MolmoQuantizedActor.options(**actor_options).remote(use_gpu=use_gpu)
    time.sleep(15)

    
    setup_logging(options.verbose)

    robot = initialize_robot(options)
    assert robot.has_arm(), 'Robot requires an arm to open door.'

    # Verify the robot is not estopped.
    check_estop(robot)

    # A lease is required to drive the robot.
    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    # Note that the take lease API is used, rather than acquire. Using acquire is typically a
    # better practice, but in this example, a user might want to switch back and forth between
    # using the tablet and using this script. Using take make this a bit less painful.
    lease_client.take()

    # Power on the robot.
    power_on(robot)

    # Stand the robot.
    stand(robot)

    # Pitch the robot up. This helps ensure that the door is in the field of view of the front
    # cameras.
    pitch_up(robot)

    open_door(options, robot, molmo)


if __name__ == '__main__':
    if not main():
        sys.exit(1)