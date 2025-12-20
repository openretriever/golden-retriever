from retriever.skills.spot.walk_to_object import walk_to_object, arg_float
from retriever.skills.spot.automated_skills.image import ImageSideBySideManager, get_images_as_cv2, get_aligned_image, get_unaligned_pixel


from bosdyn.client import create_standard_sdk
from bosdyn.client.util import authenticate, setup_logging, get_logger, add_base_arguments
from bosdyn.client.lease import LeaseClient

from bosdyn.api import geometry_pb2
#import cv2

import argparse
import ray
from PIL import Image
from retriever.models.vlms.molmo_quantized_actor import MolmoQuantizedActor, draw_points
from glob import glob
import time
import sys


def walk(options, robot, molmo, prompt_text, sources = ['frontleft_fisheye_image', 'frontright_fisheye_image'], debug = False):
    molmo_prompt = "You are a world champion robot in walking to objects. Your job is to point at the " + prompt_text + ". Only return one point. Return the center of the object so that you can walk to it easily. ONLY PROVIDE ONE POINT."

    image = None
    pick_vec = None

    if len(sources) == 2:
        print('2 input sources', end = ' ')
        print(sources)

        image_dict = get_images_as_cv2(robot, sources)
        window_name = 'Walk to Object'
        request_manager = ImageSideBySideManager(image_dict, window_name)
        side_by_side = request_manager.side_by_side
        temp_side_by_side = side_by_side.copy()
        #temp_side_by_side = cv2.cvtColor(temp_side_by_side, cv2.COLOR_BGR2RGB)
        temp_side_by_side = Image.fromarray(temp_side_by_side)
        output = ray.get(molmo.predict.remote(temp_side_by_side, molmo_prompt, points=True))
        points = output["points"]

        request_manager.set_selected_position_side_by_side(points[0])
        object_pixel = request_manager.get_selected_pixel()
        pick_vec = geometry_pb2.Vec2(x=object_pixel[0], y=object_pixel[1])
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
            #image = image_dict[key][0]
            image_copy = image_dict[key][1].copy()
            #image_copy = get_aligned_image(image_copy)
            break
        temp_image = image_copy
        #temp_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
        temp_image = Image.fromarray(temp_image)
        output = ray.get(molmo.predict.remote(temp_image, molmo_prompt, points=True))
        points = output["points"]

        pick_vec = geometry_pb2.Vec2(x=points[0][0], y=points[0][1])
        image = image_dict[key][0]
        if debug:
            print(points)
            temp_image = draw_points(temp_image, points)
            temp_image.show()

    elif len(sources) > 2:
        print('More than 2 input sources', end = ' ')
        print(sources)
        print('Not implemented yet')
        sys.exit(1)
    else:
        print('Error: Cannot take in less than 1 source')
        sys.exit(1)

    walk_to_object(options, robot, image, pick_vec)
    return True







def main():
    parser = argparse.ArgumentParser()
    add_base_arguments(parser)
    parser.add_argument('-d', '--distance', help='Distance from object to walk to (meters).',
                        default=None, type=arg_float)
    parser.add_argument('--object-name', help='Name of the object to grasp', default='blue brick')
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
        if walk(options, robot, molmo, options.object_name, sources, debug=True):
            return True
        else:
            return False
    except Exception as e:
        logger = get_logger()
        logger.exception('Threw an exception')
        print(e)
        return False

if __name__ == '__main__':
    if not main():
        sys.exit(1)
