from retriever.skills.spot.arm_grasp import arm_object_pick
from retriever.skills.spot.automated_skills.image import ImageSideBySideManager, get_images_as_cv2, get_aligned_image, get_unaligned_pixel

from bosdyn.client import create_standard_sdk
from bosdyn.client.util import authenticate, setup_logging, add_base_arguments, get_logger
from bosdyn.client.lease import LeaseClient

from bosdyn.api import geometry_pb2
import cv2

import argparse
import ray
from PIL import Image
from retriever.models.vlms.molmo_quantized_actor import MolmoQuantizedActor, draw_points
from glob import glob
import time
import sys

def grasp(options, robot, molmo, prompt_text, sources = ['frontleft_fisheye_image', 'frontright_fisheye_image'], debug=False):
    molmo_prompt = "You are a world champion robot in grasping objects. Your job is to point at the " + prompt_text + ". Only return one point. Return the ideal grasping point so that you can easily hold the object with your gripper. Try to prioritize a top down grasp if possible followed by a horizontal grasp. ONLY PROVIDE ONE POINT. Return the center top of the object."

    image = None
    pick_vec = None


    if len(sources) == 2:
        print('2 input sources', end = ' ')
        print(sources)


        image_dict = get_images_as_cv2(robot, sources)
        window_name = 'Grasp Object'
        request_manager = ImageSideBySideManager(image_dict, window_name)
        # request_manager.get_user_input_handle_and_hinge()
        side_by_side = request_manager.side_by_side
        temp_side_by_side = side_by_side.copy()
        #temp_side_by_side = cv2.cvtColor(temp_side_by_side, cv2.COLOR_BGR2RGB)
        temp_side_by_side = Image.fromarray(temp_side_by_side)
        output = ray.get(molmo.predict.remote(temp_side_by_side, molmo_prompt, points=True))
        points = output["points"]

        request_manager.set_selected_position_side_by_side(points[0])
        grasp_pixel = request_manager.get_selected_pixel()
        pick_vec = geometry_pb2.Vec2(x=grasp_pixel[0], y=grasp_pixel[1])
        image = request_manager.image_dict[request_manager.clicked_source][0]
        if debug:
            print(points)
            temp_image = draw_points(temp_side_by_side, points)
            temp_image.show()

    elif len(sources) == 1:
        print('1 input source', end = ' ')
        print(sources)

        image_dict = get_images_as_cv2(robot, sources)
        for key in image_dict:
            image = image_dict[key][0]
            image_copy = image_dict[key][1].copy()
            image_copy = get_aligned_image(image_copy)
            break

        #temp_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
        temp_image = image_copy
        temp_image = Image.fromarray(temp_image)

        output = ray.get(molmo.predict.remote(temp_image, molmo_prompt, points=True))
        points = output["points"]

        pixel = get_unaligned_pixel(image_copy, points[0])
        pick_vec = geometry_pb2.Vec2(x=pixel[0], y=pixel[1])

        if debug:
            print(points)
            temp_image = draw_points(temp_image, points)
            temp_image.show()

    elif len(sources) > 2:
        print('More than 2 input sources', end=' ')
        print(sources)
        print('Not implemented yet')
        sys.exit(1)
    else:
        print('Error: Cannot take in less than 1 source')
        sys.exit(1)

    arm_object_pick(options, robot, image, pick_vec)
    return True


def main():
    parser = argparse.ArgumentParser()
    add_base_arguments(parser)
    parser.add_argument('-t', '--force-top-down-grasp',
                        help='Force the robot to use a top-down grasp (vector_alignment demo)',
                        action='store_true')
    parser.add_argument('-f', '--force-horizontal-grasp',
                        help='Force the robot to use a horizontal grasp (vector_alignment demo)',
                        action='store_true')
    parser.add_argument(
        '-r', '--force-45-angle-grasp',
        help='Force the robot to use a 45 degree angled down grasp (rotation_with_tolerance demo)',
        action='store_true')
    parser.add_argument('-s', '--force-squeeze-grasp',
                        help='Force the robot to use a squeeze grasp', action='store_true')
    parser.add_argument('--object-name', help='Name of the object to grasp', default='red brick')
    parser.add_argument('--ray-server', type=str, default="ray://grail-mars.neu.edu:10001", help="Ray server address")

    options = parser.parse_args()
    config = options
    setup_logging(config.verbose)

    sdk = create_standard_sdk('ArmObjectPickClient')

    robot = sdk.create_robot(config.hostname)
    authenticate(robot)
    robot.time_sync.wait_for_sync()

    ignore = glob("*/")
    ignore = [item for item in ignore if "src" not in item]
    ignore = ignore + glob("src/*/")
    ignore = [item for item in ignore if "models" not in item]
    ignore = ["\\" + item for item in ignore]
    ignore.append('\\.git\\')

    ray.init(address=options.ray_server, runtime_env={"working_dir": ".", "excludes": ignore})

    use_gpu = True
    actor_options = {"num_gpus": 2} if use_gpu else {}
    molmo = MolmoQuantizedActor.options(**actor_options).remote(use_gpu=use_gpu)
    time.sleep(0.5)

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
        print('Error: cannot force more than one type of grasp.  Choose only one.')
        sys.exit(1)

    try:
        assert robot.has_arm(), 'Robot requires an arm to run this skill.'

        # Verify the robot is not estopped and that an external application has registered and holds
        # an estop endpoint.
        assert not robot.is_estopped(), 'Robot is estopped. Please use an external E-Stop client to ' \
                                        'un-estop the robot.'


        lease_client = robot.ensure_client(LeaseClient.default_service_name)
        lease_client.take()
        # image_client = robot.ensure_client(ImageClient.default_service_name)

        sources = ['frontleft_fisheye_image', 'frontright_fisheye_image']
        #sources = ['left_fisheye_image', 'right_fisheye_image']
        #sources = ['back_fisheye_image']
        #sources = ['hand_color_image']
        if grasp(config, robot, molmo, options.object_name, sources, debug=True):
            return True
        else:
            return False
    except Exception as exc:  # pylint: disable=broad-except
        logger = get_logger()
        logger.exception('Threw an exception')
        print(exc)
        return False


if __name__ == '__main__':
    if not main():
        sys.exit(1)