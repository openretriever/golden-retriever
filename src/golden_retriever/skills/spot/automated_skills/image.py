import cv2
import numpy as np
import math
import sys
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.api import image_pb2
from bosdyn.client import frame_helpers
from PIL import Image, ImageDraw, ImageFont
import argparse
from bosdyn.client.util import authenticate, setup_logging, get_logger, add_base_arguments
from bosdyn.client import create_standard_sdk


def draw_text_on_image(image, text):
    """Draw text on an image.

    Args:
        image: PIL Image object.
        text: Text to draw.

    Returns:
        PIL Image object with text drawn on it.
    """
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_offset_y = image.size[1] - 25  # Use size attribute instead of shape
    text_offset_x = (image.size[0] - text_width) // 2
    draw.text((text_offset_x, text_offset_y), text, font=font, fill="white")
    return image


def get_images_as_cv2(robot, sources):
    """Request image sources from robot. Decode and store as OpenCV image as well as proto.

    Args:
        robot: (Robot) Interface to Spot robot.
        sources: (list) String names of image sources.

    Returns:
        dict: Dictionary from image source name to (image proto, CV2 image) pairs.
    """
    image_client = robot.ensure_client(ImageClient.default_service_name)
    #image_responses = image_client.get_image_from_sources(sources)
    image_request = [
        build_image_request(source, pixel_format=image_pb2.Image.PIXEL_FORMAT_RGB_U8)
        for source in sources
    ]
    image_responses = image_client.get_image(image_request)

    image_dict = dict()
    num_bytes = 3

    for response in image_responses:

        # Convert image proto to CV2 image, for display later.
        image = np.frombuffer(response.shot.image.data, dtype=np.uint8)
        #image = cv2.imdecode(image, -1)
        #image = (
        #    image
        #    if len(image.shape) == 3
        #    else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        #)
        if response.shot.image.format == image_pb2.Image.FORMAT_RAW:
            try:
                # Attempt to reshape array into an RGB rows X cols shape.
                image = image.reshape((response.shot.image.rows, response.shot.image.cols, num_bytes))
                #image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            except ValueError:
                # Unable to reshape the image data, trying a regular decode.
                image = cv2.imdecode(image, -1)
        else:
            image = cv2.imdecode(image, -1)

        if len(image.shape) == 2:  # Grayscale image
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Convert to RGB
        elif len(image.shape) == 3 and image.shape[2] == 3:  # Assume BGR format for color image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        image_dict[response.source.name] = (response, image)
    return image_dict




def get_aligned_image(image):
    """Rotate image to be aligned with robot Z axis.

    Args:
        image: (cv2.Image) Image to rotate.

    Returns:
        cv2.Image: Rotated image.
    """
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

def get_unaligned_pixel(image, pixel):

    height, width = image.shape
    pixel_xy = [0, 0]
    th = -math.pi / 2
    xm = width / 2
    ym = height / 2
    x = pixel[0] - xm
    y = pixel[1] - ym
    pixel_xy[0] = math.cos(th) * x - math.sin(th) * y + ym
    pixel_xy[1] = math.sin(th) * x + math.cos(th) * y + xm
    return pixel_xy

class ImageSideBySideManager:
    """Helper object for displaying side by side images to the user and requesting user selected
    touchpoints. This class handles the bookkeeping for converting between a touchpoints of side by
    side display image of the frontleft and frontright fisheye images and the individual images.

    Args:
        image_dict: (dict) Dictionary from image source name to (image proto, CV2 image) pairs.
        window_name: (str) Name of display window.
    """

    def __init__(self, image_dict, window_name):
        self.image_dict = image_dict
        self.window_name = window_name
        self.selected_position_side_by_side = None
        self._side_by_side = None
        self.clicked_source = None
        self.front = False
    @property
    def side_by_side(self):
        """cv2.Image: Side by side rotated frontleft and frontright fisheye images"""
        if self._side_by_side is not None:
            return self._side_by_side

        # Convert CV2 images to numpy for processing.
        fr_fisheye_image = None
        fl_fisheye_image = None


        for key in self.image_dict:
            if "left" in key.lower():
                if "front" in key.lower():
                    self.front = True
                    fl_fisheye_image = self.image_dict[key][1]
                    fl_fisheye_image = cv2.rotate(fl_fisheye_image, cv2.ROTATE_90_CLOCKWISE)
                else:
                    fl_fisheye_image = self.image_dict[key][1]
                    
            elif "right" in key.lower():
                if "front" in key.lower():
                    self.front = True
                    fr_fisheye_image = self.image_dict[key][1]
                    fr_fisheye_image = cv2.rotate(fr_fisheye_image, cv2.ROTATE_90_CLOCKWISE)
                else:
                    fr_fisheye_image = self.image_dict[key][1]
                    fr_fisheye_image = cv2.rotate(fr_fisheye_image, cv2.ROTATE_90_CLOCKWISE)
                    fr_fisheye_image = cv2.rotate(fr_fisheye_image, cv2.ROTATE_90_CLOCKWISE)
            else:
                print("Error: Image source not recognized")
                sys.exit(1)
        if self.front:
            self._side_by_side = np.hstack([fr_fisheye_image, fl_fisheye_image])
        else:
            self._side_by_side = np.hstack([fl_fisheye_image, fr_fisheye_image])

        return self._side_by_side

    def user_input_set(self):
        """bool: True if handle and hinge position set."""
        return (self.selected_position_side_by_side)

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.selected_position_side_by_side:
                self.selected_position_side_by_side = (x, y)

    def set_selected_position_side_by_side(self, selected_position_side_by_side):
        """Set the selected position in the side by side image.

        Args:
            selected_position_side_by_side: (tuple) Pixel position in side by side image.
        """
        self.selected_position_side_by_side = selected_position_side_by_side

    def get_selected_pixel(self):
        """Convert from touchpoints in side by side image to a WalkToObjectInImage request.
        Optionally show debug image of touch point.


        Returns:
            ManipulationApiRequest: Request with WalkToObjectInImage info populated.
        """

        # Figure out which source the user actually clicked.
        height, width, _ = self.side_by_side.shape
        if self.front:
            if self.selected_position_side_by_side[0] > width / 2:
                for key in self.image_dict:
                    if "left" in key.lower():
                        self.clicked_source = key
                rotated_pixel = self.selected_position_side_by_side
                rotated_pixel = (rotated_pixel[0] - width / 2, rotated_pixel[1])
            else:
                for key in self.image_dict:
                    if "right" in key.lower():
                        self.clicked_source = key
                rotated_pixel = self.selected_position_side_by_side

            # Undo pixel rotation by rotation 90 deg CCW.
            pixel_xy = [0,0]
            th = -math.pi / 2
            xm = width / 4
            ym = height / 2
            x = rotated_pixel[0] - xm
            y = rotated_pixel[1] - ym
            pixel_xy[0] = math.cos(th) * x - math.sin(th) * y + ym
            pixel_xy[1] = math.sin(th) * x + math.cos(th) * y + xm

            #pixel_xy = geometry_pb2.Vec2(x=pixel_xy[0], y=pixel_xy[1])
            
        else:
            if self.selected_position_side_by_side[0] > width / 2:
                for key in self.image_dict:
                    if "right" in key.lower():
                        self.clicked_source = key
                rotated_pixel = self.selected_position_side_by_side
                rotated_pixel = (rotated_pixel[0] - width / 2, rotated_pixel[1])
                pixel_xy = [0, 0]
                th = -math.pi  # -180 degrees
                xm = width / 4
                ym = height / 2

                x = rotated_pixel[0] - xm
                y = rotated_pixel[1] - ym

                # Applying 180-degree rotation formula
                pixel_xy[0] = -x + xm
                pixel_xy[1] = -y + ym
            else:
                for key in self.image_dict:
                    if "left" in key.lower():
                        self.clicked_source = key
                rotated_pixel = self.selected_position_side_by_side
                pixel_xy = [0, 0]
                th = 0 
                xm = width / 4
                ym = height / 2

                x = rotated_pixel[0] - xm
                y = rotated_pixel[1] - ym
                # Applying 180-degree rotation formula
                pixel_xy[0] = rotated_pixel[0]
                pixel_xy[1] = rotated_pixel[1]
        return pixel_xy
            
    @property
    def vision_tform_sensor(self):
        """Look up vision_tform_sensor for sensor which user clicked.

        Returns:
            math_helpers.SE3Pose
        """
        clicked_image_proto = self.image_dict[self.clicked_source][0]
        frame_name_image_sensor = clicked_image_proto.shot.frame_name_image_sensor
        snapshot = clicked_image_proto.shot.transforms_snapshot
        return frame_helpers.get_a_tform_b(snapshot, frame_helpers.VISION_FRAME_NAME,
                                           frame_name_image_sensor)

    def get_user_input(self):
        """Open window showing the side by side fisheye images with on-screen prompts for user."""
        cv2.imshow(self.window_name, self.side_by_side)
        cv2.setMouseCallback(self.window_name, self._on_mouse)
        while not self.user_input_set():
            cv2.waitKey(1)
        cv2.destroyAllWindows()

def main():
    sources = ['frontleft_fisheye_image', 'frontright_fisheye_image']
    sources_2 = ['left_fisheye_image', 'right_fisheye_image']
    sources_3 = ['back_fisheye_image']
    sources_4 = ['hand_color_image']
    
    
    parser = argparse.ArgumentParser()
    add_base_arguments(parser)
    options = parser.parse_args()
    config = options
    setup_logging(config.verbose)

    sdk = create_standard_sdk('ImageTester')

    robot = sdk.create_robot(config.hostname)
    authenticate(robot)
    robot.time_sync.wait_for_sync()
    
    image_dict = get_images_as_cv2(robot, sources)
    image_dict_2 = get_images_as_cv2(robot, sources_2)
    image_dict_3 = get_images_as_cv2(robot, sources_3)
    image_dict_4 = get_images_as_cv2(robot, sources_4)
    
    window_name = 'Image Tester'
    request_manager = ImageSideBySideManager(image_dict, window_name)
    request_manager_2 = ImageSideBySideManager(image_dict_2, window_name)
    #request_manager_3 = ImageSideBySideManager(image_dict_3, window_name)
    #request_manager_4 = ImageSideBySideManager(image_dict_4, window_name)
    
    
    side_by_side = request_manager.side_by_side
    temp_side_by_side = side_by_side.copy()
    temp_side_by_side = Image.fromarray(temp_side_by_side)
    temp_side_by_side.show()
    
    
    side_by_side = request_manager_2.side_by_side
    temp_side_by_side = side_by_side.copy()
    temp_side_by_side = Image.fromarray(temp_side_by_side)
    temp_side_by_side.show()
    
    for key in image_dict_3:
        image_copy_3 = image_dict_3[key][1].copy()
        break
    
    for key in image_dict_4:
        image_copy_4 = image_dict_4[key][1].copy()
        break
    
    temp_image_3 = Image.fromarray(image_copy_3)
    temp_image_4 = Image.fromarray(image_copy_4)
    
    temp_image_3.show()
    temp_image_4.show()
    
if __name__ == '__main__':
    if not main():
        sys.exit(1)