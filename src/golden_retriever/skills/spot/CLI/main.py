from retriever.skills.spot.automated_skills.image import ImageSideBySideManager, get_images_as_cv2, draw_text_on_image
from retriever.skills.spot.automated_skills.grasp import grasp
from retriever.skills.spot.automated_skills.walk import walk
from retriever.skills.spot.walk_to_landmark import GraphNavInterface, load_annotations, init_graphnav
from retriever.skills.spot.automated_skills.door import open_door
from retriever.skills.spot.spot_place import place_at_relative_position as drop

from bosdyn.client import create_standard_sdk
from bosdyn.client.util import authenticate, setup_logging, get_logger, add_base_arguments
from bosdyn.client.lease import LeaseClient
from bosdyn.client import math_helpers
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand


import argparse

from PIL import Image
from glob import glob
import time
import sys


import ray
from retriever.models.vlms.molmo_quantized_actor import MolmoQuantizedActor, draw_points
from retriever.models.api_models.utils.openai_utils import prepare_openai_Image_captioned_messages, prepare_openai_Image_messages, prepare_system_messages, call_openai_api, prepare_openai_messages
from retriever.models.common_utils import timer, Timer



def init_robot(options):
    """Initialize robot.

    Args:
        options: (dict) Options dictionary.

    Returns:
        Robot: Robot object.
    """
    sdk = create_standard_sdk('AutomatedControlInterface')
    robot = sdk.create_robot(options.hostname)
    authenticate(robot)
    robot.time_sync.wait_for_sync()
    assert robot.has_arm(), 'Robot requires an arm.'
    assert not robot.is_estopped(), 'Robot is estopped. Please use an external E-Stop client to ' \
                                    'un-estop the robot.'
    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    lease_client.take()
    robot.power_on(timeout_sec=20)
    assert robot.is_powered_on(), 'Robot power on failed.'

    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    blocking_stand(command_client, timeout_sec=10)

    return robot

def init_options():
    parser = argparse.ArgumentParser()
    add_base_arguments(parser)
    parser.add_argument('-u', '--upload-filepath',
                        help='Full filepath to graph and snapshots to be uploaded.', required=True)
    parser.add_argument('--upload-annotations', help='Full filepath to yaml annotations.', required=True)
    parser.add_argument('-r','--ray-server', type=str, default="ray://grail-mars.neu.edu:10001", help="Ray server address")
    parser.add_argument('--debug', action='store_true', help='Show intermediate debug data.')
    options = parser.parse_args()
    return options, parser


def get_camera_images(robot):
    source1 = get_camera_source('FRONT')
    source2 = get_camera_source('SIDE')
    source3 = get_camera_source('BACK')
    source4 = get_camera_source('HAND')

    sources = source1 + source2 + source3
    image_dict1 = get_images_as_cv2(robot, source1)
    image_dict2 = get_images_as_cv2(robot, source2)
    image_dict3 = get_images_as_cv2(robot, source3)
    image_dict4 = get_images_as_cv2(robot, source4)

    request_manager1 = ImageSideBySideManager(image_dict1, 'Front Camera')
    request_manager2 = ImageSideBySideManager(image_dict2, 'Side Camera')

    front_side_by_side = request_manager1.side_by_side.copy()
    side_side_by_side = request_manager2.side_by_side.copy()

    front_side_by_side = Image.fromarray(front_side_by_side)
    front_side_by_side = draw_text_on_image(front_side_by_side, "FRONT")
    side_side_by_side = Image.fromarray(side_side_by_side)
    side_side_by_side = draw_text_on_image(side_side_by_side, "SIDE")

    back = image_dict3['back_fisheye_image'][1].copy()
    back = Image.fromarray(back)
    back = draw_text_on_image(back, "BACK")

    hand = image_dict4['hand_color_image'][1].copy()
    hand = Image.fromarray(hand)
    hand = draw_text_on_image(hand, "HAND")

    return [front_side_by_side, side_side_by_side, back, hand]
    #return [front_side_by_side]

def get_grasp_options(options,force_top_down_grasp=True, force_horizontal_grasp=False, force_45_angle_grasp=False, force_squeeze_grasp=False):
    predefined_values = {
        "hostname" : options.hostname,
        "verbose" : options.verbose,
        "force_top_down_grasp" : force_top_down_grasp,
        "force_horizontal_grasp" : force_horizontal_grasp,
        "force_45_angle_grasp" : force_45_angle_grasp,
        "force_squeeze_grasp" : force_squeeze_grasp,
    }
    new_options = argparse.Namespace(**predefined_values)
    return new_options

def get_walk_options(options, distance = 0.5):
    predefined_values = {
        "hostname" : options.hostname,
        "verbose" : options.verbose,
        "distance" : distance,
    }
    new_options = argparse.Namespace(**predefined_values)
    return new_options

def get_place_options(options):
    predefined_values = {
        "hostname": options.hostname,
        "verbose": options.verbose,
        "target_pos": math_helpers.Vec3(0.8, 0.0, 0.2)
    }
    new_options = argparse.Namespace(**predefined_values)
    return new_options

def ray_init(server_address="ray://grail-mars.neu.edu:10001"):
    ignore = glob("*/")
    ignore = [item for item in ignore if "src" not in item]
    ignore = ignore + glob("src/*/")
    ignore = [item for item in ignore if "models" not in item]
    ignore = ["\\" + item for item in ignore]
    ignore.append('\\.git\\')

    ray.init(address=server_address, runtime_env={"working_dir": ".", "excludes": ignore})



def init_molmo(use_gpu=True):
    actor_options = {"num_gpus": 2} if use_gpu else {}
    molmo = MolmoQuantizedActor.options(**actor_options).remote(use_gpu=use_gpu)
    time.sleep(1)
    return molmo

def get_gpt_prompt(robot, task):
    prefix = ("Task:" + task + "\n" +
              "Robot has a front camera, side camera, and a back camera. Robot cameras:"
)

    suffix = "Possible outputs: 'WALK_TO_LANDMARK', 'OPEN_DOOR', 'PICK_OBJECT', 'DROP_OBJECT', 'WALK_TO_OBJECT', 'IMPOSSIBLE', 'SUCCESS'."

    gpt_prompt = prepare_openai_Image_captioned_messages(prefix=prefix, suffix=suffix, images=get_camera_images(robot),captions=["Image 1: Front Cameras", "Image 2: Side Cameras", "Image 3: Back Camera", "Image 4: Hand Camera"])
    #gpt_prompt = prepare_openai_Image_messages(prefix=prefix, suffix=suffix, images=get_camera_images(robot))
    return gpt_prompt


def get_system_prompt(task):
    content = ("You are a world champion robot in completing simple manipulation tasks. Your job is to choose which "
              "skill to use to complete a given task. The skills you possess are: 'WALK_TO_LANDMARK', 'OPEN_DOOR', 'PICK_OBJECT', "
               "'DROP_OBJECT', and "
              "'WALK_TO_OBJECT'. When given a task, you must choose the correct skill to use. Choose the correct skill to "
               "use and ONLY output the skill name. After your response is verified you will later be asked for other "
               "parameters for each respective skill (example-object name and camera name in which object exists)\n" +

              "Description of skills:\n " +

               "'WALK_TO_LANDMARK': The robot will walk to an annotated landmark in the building map. Examples of "
               "landmarks can be '101_home', '101_door_2_outside', 'main_elevator', etc. Only "
               "pick this skill if the robot needs to navigate somewhere and a landmark is also given like "
               "'Inside side of the door in Room 101': '101_door_1_inside', etc.\n" +

                "'OPEN_DOOR': The robot will open the door in front of it. Only pick this skill if the robot is in front"
                " of a door and it needs to be opened.\n" +

              "'PICK_OBJECT': The robot will pick up an object given a name and it is visible in camera. It will "
              "navigate to the object if it isn't too far and then pick it up. If the object is visible but far away"
              " in camera it might be helpful to use other objects between the robot and the target object and use "
              "the 'WALK_TO_OBJECT' skill multiple times to help the robot navigate to the object. But only do this in "
              "very rare cases. Normally you can just call 'PICK_OBJECT' if the object is visible to you.\n" +

              "'DROP_OBJECT': The robot will drop the object it is currently holding at the current location.\n" +

              "'WALK_TO_OBJECT': The robot will walk to the object given a distance in meters. The object must be "
              "visible in camera. If the object is visible but far away in camera it might be helpful to use other "
              "objects between the robot and the object and use this skill multiple times to help the robot navigate"
              " to the object. But only do this in very rare cases. Normally you can just call 'WALK_TO_OBJECT' if the "
              "object is visible to you.\n\n\n" +

              "If the task is not possible to complete, output 'IMPOSSIBLE'. \n" +

              "If nothing needs to be done, output 'SUCCESS'.\n" +

              "Robot is the boston dynamics spot dog robot and has a front camera, side camera, back camera, and hand camera.\n" +

              "The camera feed images are ordered 'FRONT', 'SIDE', 'BACK', and 'HAND'. When asked make sure to only output the camera where the object is present. \n" +

               "TASK: " + task + "\n"
            )

    gpt_prompt = prepare_system_messages(content)
    return gpt_prompt

def get_gpt_prompt_check_end(robot):
    prefix = ("Is the task successfully completed or do you need to call another skill to complete the task? "
                   "If you need to call another skill, only provide the name of the skill to call. If the task is "
                   "successfully completed, please output 'SUCCESS'.")
    suffix = "Possible outputs: 'WALK_TO_LANDMARK', 'OPEN_DOOR', 'PICK_OBJECT', 'DROP_OBJECT', 'WALK_TO_OBJECT', 'IMPOSSIBLE', 'SUCCESS'."
    gpt_prompt = prepare_openai_Image_captioned_messages(prefix=prefix, suffix=suffix, images=get_camera_images(robot), captions=["Image 1: Front Cameras", "Image 2: Side Cameras", "Image 3: Back Camera", "Image 4: Hand Camera"])
    #gpt_prompt = prepare_openai_Image_messages(prefix=prefix, suffix=suffix, images=get_camera_images(robot))
    return gpt_prompt

def get_object_name_prompt():
    prefix = ("What is the name of the object you want to pass to the chosen skill? Feel free to use adjectives to describe the object if needed (example-blue brick). "
              "The skill will using a highly intelligent pointing algorithm to determine the object in the camera and then perform needed skill.")
    gpt_prompt = prepare_openai_messages(prefix)
    return gpt_prompt

def get_landmark_options(annotations):
    ret = []
    for key in annotations.keys():
        ret.append(key)
    return ret
            
            
def get_landmark_name_prompt(annotations):
    landmark_options = get_landmark_options(annotations)
    keys_string = ', '.join(landmark_options)
    prefix = ("What is the name of the landmark you want to pass to the chosen skill? The robot will use the landmark to navigate to the desired location. The options for landmarks are: \n" +
              keys_string + "\n"
              + "Choose the correct landmark name from the list above. If the landmark given by the user is cannot be matched to the list above, the give an output of will output 'IMPOSSIBLE'.")
    gpt_prompt = prepare_openai_messages(prefix)
    return gpt_prompt

def get_camera_prompt(robot):
    prefix = ("Which camera do you want to use to complete the current skill? \n" +
              "The robot has a front camera, side camera, " +
              "and back camera. YOU CAN ONLY CHOOSE ONE CAMERA. IT WILL BE PASSED INTO THE SKILL AND ONLY THE PERSPECTIVE OF THAT CAMERA WOULD BE USED. So only choose the most relevant camera to the current task.\n")

    suffix = "Possible outputs: 'FRONT', 'SIDE', 'BACK', 'HAND'."

    gpt_prompt = prepare_openai_Image_captioned_messages(prefix=prefix, suffix=suffix, images=get_camera_images(robot),captions=["image 1: 'FRONT'", "image 2: 'SIDE'", "image 3: 'BACK'", "image 4: 'HAND'"])
    #gpt_prompt = prepare_openai_Image_messages(prefix=prefix, suffix=suffix, images=get_camera_images(robot))

    return gpt_prompt


def get_camera_source(camera):
    if camera == "FRONT":
        sources = ['frontleft_fisheye_image', 'frontright_fisheye_image']
    elif camera == "SIDE":
        sources = ['left_fisheye_image', 'right_fisheye_image']
    elif camera == "BACK":
        sources = ['back_fisheye_image']
    elif camera == "HAND":
        sources = ['hand_color_image']
    else:
        print("Invalid camera source. Defaulting to front.")
        sources = ['frontleft_fisheye_image', 'frontright_fisheye_image']

    return sources


def run_gpt_response(annotations, options, history, response, robot, molmo, graph_nav_command_line, model = "gpt-4o", debug=False):
    if response == "WALK_TO_LANDMARK":
        print('WALK_TO_LANDMARK')
        landmark_prompt = get_landmark_name_prompt(annotations)
        history = history + (landmark_prompt)
        with Timer(enable_print=True) as timer:
            landmark = call_openai_api(history, model=model, max_tokens=100, temperature=0.2)
            history.append({"role": "assistant", "content": landmark})
            print("LANDMARK: ", landmark)
        if landmark not in get_landmark_options(annotations):
            print("Invalid landmark.")
            response = "IMPOSSIBLE"
            run_gpt_response(annotations, options, history, response, robot, molmo, graph_nav_command_line)
        else:
            entry = annotations[landmark]
            if entry["type"] == "waypoint":
                graph_nav_command_line.navigate_to_waypoint(entry["id"])
            elif entry["type"] == "seed_coordinate":
                coords = entry["coordinates"]
                graph_nav_command_line.navigate_to_seed_x_y_yaw(
                    coords["x"], coords["y"], coords["yaw"]
                )
            print("Walking to landmark completed.")
            print("Running task completion check")
            check_end = get_gpt_prompt_check_end(robot)
            history = history + (check_end)
            with Timer(enable_print=True) as timer:
                response = call_openai_api(history, model=model, max_tokens=100, temperature=0.2)
                history.append({"role": "assistant", "content": response})
                print("RESPONSE: ", response)
            run_gpt_response(annotations, options, history, response, robot, molmo, graph_nav_command_line)
    elif response == "OPEN_DOOR":
        print('OPEN_DOOR')
        open_door(options, robot, molmo)
        print("Door open ran")
        check_end = get_gpt_prompt_check_end(robot)
        history = history + (check_end)
        with Timer(enable_print=True) as timer:
            response = call_openai_api(history, model=model, max_tokens=100, temperature=0.2)
            history.append({"role": "assistant", "content": response})
            print("RESPONSE: ", response)
        run_gpt_response(annotations, options, history, response, robot, molmo, graph_nav_command_line)
    elif response == "PICK_OBJECT":
        print("PICK_OBJECT")
        object_name_prompt = get_object_name_prompt()
        history = history + (object_name_prompt)
        with Timer(enable_print=True) as timer:
            object_name = call_openai_api(history, model=model, max_tokens=100, temperature=0.2)
            history.append({"role": "assistant", "content": object_name})
            print("OBJECT NAME: ", object_name)
        grasp_options = get_grasp_options(options)
        camera_prompt = get_camera_prompt(robot)
        history = history + (camera_prompt)
        with Timer(enable_print=True) as timer:
            camera = call_openai_api(history, model=model, max_tokens=100, temperature=0.2)
            history.append({"role": "assistant", "content": camera})
            print("CAMERA: ", camera)
        sources = get_camera_source(camera)
        grasp(grasp_options, robot, molmo, object_name, sources, debug=debug)
        print("Grasp completed.")
        print("Running task completion check")
        check_end = get_gpt_prompt_check_end(robot)
        history = history + (check_end)
        with Timer(enable_print=True) as timer:
            response = call_openai_api(history, model=model, max_tokens=100, temperature=0.2)
            history.append({"role": "assistant", "content": response})
            print("RESPONSE: ", response)
        run_gpt_response(annotations, options, history, response, robot, molmo, graph_nav_command_line)
    elif response == "DROP_OBJECT":
        print("DROP_OBJECT")
        drop_options = get_place_options(options)
        target_pos = drop_options.target_pos
        drop(robot, target_pos)
        pass
    elif response == "WALK_TO_OBJECT":
        print("WALK_TO_OBJECT")
        object_name_prompt = get_object_name_prompt()
        history = history + (object_name_prompt)
        with Timer(enable_print=True) as timer:
            object_name = call_openai_api(history, model=model, max_tokens=100, temperature=0.2)
            history.append({"role": "assistant", "content": object_name})
            print("OBJECT NAME: ", object_name)
        walk_options = get_walk_options(options)
        camera_prompt = get_camera_prompt(robot)
        history = history + (camera_prompt)
        with Timer(enable_print=True) as timer:
            camera = call_openai_api(history, model=model, max_tokens=100, temperature=0.2)
            history.append({"role": "assistant", "content": camera})
            print("CAMERA: ", camera)
        sources = get_camera_source(camera)
        walk(walk_options, robot, molmo, object_name, sources, debug=debug)
        print("WALK completed.")
        print("Running task completion check")
        check_end = get_gpt_prompt_check_end(robot)
        history = history + (check_end)
        with Timer(enable_print=True) as timer:
            response = call_openai_api(history, model=model, max_tokens=100, temperature=0.2)
            history.append({"role": "assistant", "content": response})
            print("RESPONSE: ", response)
        run_gpt_response(annotations, options, history, response, robot, molmo, graph_nav_command_line)
    elif response == "IMPOSSIBLE":
        print("Task is impossible to complete.")
    elif response == "SUCCESS":
        print("Task is successfully completed.")
    else:
        print("Invalid response.")
        #print("Response: ", response)
    return True

def main():
    options, _ = init_options()
    ray_init(options.ray_server)
    molmo = init_molmo()

    robot = init_robot(options)
    graph_nav_command_line = GraphNavInterface(robot, options.upload_filepath, options=options,
                                               use_gps=False)  # Upload data
    
    init_graphnav(graph_nav_command_line)
    
    annotations = load_annotations(options.upload_annotations)

    model = "gpt-4o"
    time.sleep(15)
    print("Welcome to the CLI interface! (Type 'q' to quit)")
    #queue = ["Walk over to the red box", "Grab the red box", "Walk over to the blue cart to your side", "Drop the box you are holding", "Walk over to the backpack" , "Grab the backpack", "Walk over to the waterbottle on top of the chair", "Drop the backpack", "Walk to the table" , "Grab the coffee cup on top of the table", "Walk over to the trashcan", "Drop the coffee cup", "Go to Door 2 inside Room 101", "Open the door", "q"]
    #queue = ["Walk over to the landmark outside door 2 of room 101", "Open the door", "Walk over to the landmark in room 101 demo space 1", "Walk over to the red box", "Pick up the red box", "Walk over to the blue cart", "drop the box", "Go to the landmark in room 101 door 1 inside the room", "Open the door", "Go to landmark in the exp lobby"]
    queue = []
    if len(queue)==0:
        while True:
            user_input = input("Enter something: ")
            if user_input.lower() == 'q' or user_input.lower() == 'quit' or user_input.lower() == 'exit' or user_input.lower() == 'exit()':
                print("Goodbye!")
                break
            else:
                print("User Input: ", user_input)

            chat_history = get_system_prompt(user_input)
            chat_history = chat_history + (get_gpt_prompt(robot, user_input))
            # print("Chat History: ", chat_history)
            with Timer(enable_print=True) as timer:
                output = call_openai_api(chat_history, model=model, max_tokens=1000, temperature=0.2)
                # print("GPT Response: ", output)

            chat_history.append({"role": "assistant", "content": output})
            run_gpt_response(annotations, options, chat_history, output, robot, molmo, graph_nav_command_line)
    else:
        for user_input in queue:
            if user_input.lower() == 'q' or user_input.lower() == 'quit' or user_input.lower() == 'exit' or user_input.lower() == 'exit()':
                print("Goodbye!")
                break
            else:
                print("User Input: ", user_input)

            chat_history = get_system_prompt(user_input)
            chat_history = chat_history + (get_gpt_prompt(robot, user_input))
            #print("Chat History: ", chat_history)
            with Timer(enable_print=True) as timer:
                output = call_openai_api(chat_history, model=model, max_tokens=1000, temperature=0.2)
                #print("GPT Response: ", output)

            chat_history.append({"role": "assistant", "content": output})
            run_gpt_response(annotations, options, chat_history, output, robot, molmo, graph_nav_command_line)

    return True

if __name__ == '__main__':
    if not main():
        sys.exit(1)