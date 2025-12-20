"""Utility functions for parsing Gemini model responses and handling data classes."""

import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union, cast, Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class PointData:
    """Data class for storing point information.
    
    Attributes:
        label: Label for the point
        normalized_point: Point coordinates in normalized space [0-1000]
        denormalized_point: Point coordinates in image space (computed from normalized_point)
    """
    label: str
    normalized_point: List[float]
    denormalized_point: Optional[List[float]] = None

    @classmethod
    def from_gemini_response(cls, response: Dict[str, Union[List[float], str]]) -> "PointData":
        """Create PointData from Gemini response.
        
        Args:
            response: Dictionary containing point and label
            
        Returns:
            PointData object
        """
        point = cast(List[float], response["point"])
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError(f"Invalid point format: {point}")
            
        # Points from Gemini are already normalized to 0-1000
        # We only need to validate the range
        y, x = point
        if not (0 <= y <= 1000 and 0 <= x <= 1000):
            raise ValueError(f"Point coordinates out of range: {point}")
            
        return cls(
            label=str(response["label"]),
            normalized_point=point,
        )

    def denormalize(self, image_size: Tuple[int, int]) -> None:
        """Convert normalized point [0-1000] back to image coordinates.
        
        Args:
            image_size: Size of the image (width, height)
        """
        y, x = self.normalized_point
        self.denormalized_point = [
            y * image_size[1] / 1000,  # y coordinate
            x * image_size[0] / 1000,  # x coordinate
        ]


@dataclass
class BoxData:
    """Data class for storing box information.
    
    Attributes:
        label: Label for the box
        box_2d: Box coordinates in normalized space [0-1000] in [y1, x1, y2, x2] format
        normalized_box: Box coordinates in normalized space [0-1000]
        denormalized_box: Box coordinates in image space (computed from normalized_box)
    """
    label: str
    box_2d: List[float]
    normalized_box: Optional[List[float]] = None
    denormalized_box: Optional[List[float]] = None

    @classmethod
    def from_gemini_response(cls, response: Dict[str, Union[List[float], str]]) -> "BoxData":
        """Create BoxData from Gemini response.
        
        Args:
            response: Dictionary containing box and label
            
        Returns:
            BoxData object
        """
        # Get box_2d data
        box_2d = cast(List[float], response.get("box_2d"))
        if box_2d is None:
            raise ValueError("Response must contain 'box_2d' key")
            
        if not isinstance(box_2d, list) or len(box_2d) != 4:
            raise ValueError(f"Invalid box_2d format: {box_2d}")
            
        # Boxes from Gemini are already normalized to 0-1000
        # We only need to validate the range
        y1, x1, y2, x2 = box_2d
        if not all(0 <= coord <= 1000 for coord in [y1, x1, y2, x2]):
            raise ValueError(f"Box coordinates out of range: {box_2d}")
            
        return cls(
            label=str(response["label"]),
            box_2d=box_2d,
            normalized_box=box_2d,
        )

    def denormalize(self, image_size: Tuple[int, int]) -> None:
        """Convert normalized box [0-1000] back to image coordinates.
        
        Args:
            image_size: Size of the image (width, height)
        """
        if self.normalized_box is None:
            return
            
        y1, x1, y2, x2 = self.normalized_box
        self.denormalized_box = [
            y1 * image_size[1] / 1000,  # y1
            x1 * image_size[0] / 1000,  # x1
            y2 * image_size[1] / 1000,  # y2
            x2 * image_size[0] / 1000,  # x2
        ]


def parse_gemini_point_response(
    response: str,
    image_size: Optional[Tuple[int, int]] = None,
) -> Dict[str, List[PointData]]:
    """Parse Gemini point response to extract point coordinates and labels.
    
    Args:
        response: Raw response from Gemini
        image_size: Optional size of the input image (width, height) for denormalization
        
    Returns:
        Dictionary with points
    """
    # Check for failure message
    if isinstance(response, str) and "failed" in response.lower():
        logging.warning("Gemini returned failure message")
        return {"points": []}
        
    try:
        # Extract JSON from response
        start = response.find("[")
        end = response.rfind("]") + 1
        assert start != -1 and end != 0, "No JSON array found in response"
            
        json_str = response[start:end]
        points_data = json.loads(json_str)
        assert isinstance(points_data, list), "Response must be a list"
        
        # Convert to PointData objects
        points = []
        for item in points_data:
            assert isinstance(item, dict), "Each point must be a dictionary"
            assert "point" in item, "Point must have 'point' key"
            assert "label" in item, "Point must have 'label' key"
            
            point_data = PointData.from_gemini_response(item)
            if image_size:
                point_data.denormalize(image_size)
            points.append(point_data)
        
        return {"points": points}
    except (json.JSONDecodeError, KeyError, ValueError, AssertionError) as e:
        logging.error(f"Failed to parse Gemini point response: {e}")
        logging.error(f"Response: {response}")
        return {"points": []}


def parse_gemini_detection_response(
    response: str,
    image_size: Optional[Tuple[int, int]] = None,
) -> Dict[str, List[BoxData]]:
    """Parse Gemini detection response to extract bounding boxes and labels.
    
    Args:
        response: Raw response from Gemini
        image_size: Optional size of the input image (width, height) for denormalization
        
    Returns:
        Dictionary with bounding boxes
    """
    # Check for failure message
    if isinstance(response, str) and "failed" in response.lower():
        logging.warning("Gemini returned failure message")
        return {"detections": []}
        
    try:
        # Extract JSON from response
        start = response.find("[")
        end = response.rfind("]") + 1
        assert start != -1 and end != 0, "No JSON array found in response"
            
        json_str = response[start:end]
        detection_data = json.loads(json_str)
        assert isinstance(detection_data, list), "Response must be a list"
        
        # Convert to BoxData objects
        boxes = []
        for item in detection_data:
            assert isinstance(item, dict), "Each box must be a dictionary"
            assert "box_2d" in item, "Box must have 'box_2d' key"
            assert "label" in item, "Box must have 'label' key"
            
            box_data = BoxData.from_gemini_response(item)
            if image_size:
                box_data.denormalize(image_size)
            boxes.append(box_data)
        
        return {"detections": boxes}
    except (json.JSONDecodeError, KeyError, ValueError, AssertionError) as e:
        logging.error(f"Failed to parse Gemini detection response: {e}")
        logging.error(f"Response: {response}")
        return {"detections": []}


def draw_points(
    image: Image.Image,
    points_data: Sequence[Union[PointData, Dict[str, Union[List[float], str]]]],
    point_radius: int = 4,
    point_color: Tuple[int, int, int] = (255, 0, 0),  # Red
    outline_color: Tuple[int, int, int] = (255, 255, 255),  # White
    outline_width: int = 2,
) -> Image.Image:
    """Draw points on an image with enhanced visualization.
    
    Args:
        image: PIL Image to draw on
        points_data: Sequence of PointData objects or dictionaries with point and label
        point_radius: Radius of the points in pixels
        point_color: RGB color for the points
        outline_color: RGB color for the point outlines
        outline_width: Width of the point outlines in pixels
        
    Returns:
        Image with points drawn
    """
    draw = ImageDraw.Draw(image)
    
    # Create a color palette for different labels
    colors = {
        'a': (255, 0, 0),    # Red
        'b': (0, 255, 0),    # Green
        'c': (0, 0, 255),    # Blue
        'd': (255, 255, 0),  # Yellow
        'e': (255, 0, 255),  # Magenta
        'f': (0, 255, 255),  # Cyan
        'g': (255, 128, 0),  # Orange
        'h': (128, 0, 255),  # Purple
        'i': (0, 255, 128),  # Teal
        'j': (255, 0, 128),  # Pink
    }
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("LiberationSans-Regular.ttf", 12)
        except (OSError, IOError):
            font = ImageFont.load_default()
    
    for point_data in points_data:
        # Get point coordinates and label
        if isinstance(point_data, dict):
            point = cast(List[float], point_data["point"])
            label = str(point_data.get("label", ""))
        else:
            if point_data.denormalized_point is None:
                logging.warning("Point not denormalized, skipping visualization")
                continue
            point = point_data.denormalized_point
            label = point_data.label
        
        # Get coordinates
        y, x = point
        
        # Use label-specific color if available, otherwise use default
        point_color = colors.get(label.lower(), point_color)
        
        # Draw white outline first
        draw.ellipse(
            (
                x - point_radius - outline_width,
                y - point_radius - outline_width,
                x + point_radius + outline_width,
                y + point_radius + outline_width,
            ),
            fill=outline_color,
        )
        
        # Draw colored point
        draw.ellipse(
            (
                x - point_radius,
                y - point_radius,
                x + point_radius,
                y + point_radius,
            ),
            fill=point_color,
        )
        
        # Draw label with background for better visibility
        if label:
            # Get text size
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Draw text background with rounded corners
            padding = 4
            draw.rounded_rectangle(
                (
                    x + point_radius + 5 - padding,
                    y - text_height/2 - padding,
                    x + point_radius + 5 + text_width + padding,
                    y + text_height/2 + padding
                ),
                radius=4,
                fill=outline_color
            )
            
            # Draw text
            draw.text(
                (x + point_radius + 5, y - text_height/2),
                label,
                fill=point_color,
                font=font
            )
    
    return image


def draw_boxes(
    image: Image.Image,
    boxes_data: Sequence[BoxData],
    box_width: int = 2,
    box_color: Tuple[int, int, int] = (0, 150, 255),  # Light blue
    label_color: Tuple[int, int, int] = (0, 150, 255),  # Light blue
    outline_color: Tuple[int, int, int] = (255, 255, 255),  # White
) -> Image.Image:
    """Draw bounding boxes on an image with enhanced visualization.
    
    Args:
        image: PIL Image to draw on
        boxes_data: Sequence of BoxData objects
        box_width: Width of the box lines in pixels
        box_color: RGB color for the boxes
        label_color: RGB color for the box labels
        outline_color: RGB color for the label outlines
        
    Returns:
        Image with boxes drawn
    """
    draw = ImageDraw.Draw(image)
    
    # Create a color palette for different labels
    colors = {
        'red': (255, 0, 0),      # Red
        'green': (0, 255, 0),    # Green
        'blue': (0, 0, 255),     # Blue
        'yellow': (255, 255, 0), # Yellow
        'purple': (128, 0, 255), # Purple
        'orange': (255, 165, 0), # Orange
        'pink': (255, 192, 203), # Pink
        'cyan': (0, 255, 255),   # Cyan
        'brown': (165, 42, 42),  # Brown
        'gray': (128, 128, 128), # Gray
    }
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)  # Slightly larger font
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("LiberationSans-Regular.ttf", 14)
        except (OSError, IOError):
            font = ImageFont.load_default()
    
    for box_data in boxes_data:
        # Get denormalized box coordinates
        assert box_data.denormalized_box is not None, "Box must have denormalized coordinates"
        y1, x1, y2, x2 = box_data.denormalized_box
        
        # Extract color from label (e.g., "red block" -> "red")
        label_color_name = box_data.label.split()[0].lower()
        box_color = colors.get(label_color_name, box_color)
        
        # Draw box with rounded corners
        draw.rounded_rectangle(
            (x1, y1, x2, y2),
            radius=4,
            outline=box_color,
            width=box_width,
        )
        
        # Draw label with background for better visibility
        if box_data.label:
            # Get text size
            text_bbox = draw.textbbox((0, 0), box_data.label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Draw text background with rounded corners
            padding = 4
            draw.rounded_rectangle(
                (
                    x1 - padding,
                    y1 - text_height - padding,
                    x1 + text_width + padding,
                    y1 + padding
                ),
                radius=4,
                fill=outline_color
            )
            
            # Draw text
            draw.text(
                (x1, y1 - text_height),
                box_data.label,
                fill=box_color,
                font=font
            )
    
    return image


def draw_masks(
    image: Image.Image,
    masks_data: Sequence[Dict[str, Union[np.ndarray, str]]],
    mask_alpha: float = 0.3,
    mask_color: Tuple[int, int, int] = (0, 0, 255),  # Blue
    label_color: Tuple[int, int, int] = (0, 0, 255),  # Blue
    outline_color: Tuple[int, int, int] = (255, 255, 255),  # White
) -> Image.Image:
    """Draw segmentation masks on an image with enhanced visualization.
    
    Args:
        image: PIL Image to draw on
        masks_data: Sequence of dictionaries with mask and label
        mask_alpha: Opacity of the masks (0-1)
        mask_color: RGB color for the masks
        label_color: RGB color for the mask labels
        outline_color: RGB color for the label outlines
        
    Returns:
        Image with masks drawn
    """
    # Convert PIL Image to numpy array for mask overlay
    image_np = np.array(image)
    
    # Create a color palette for different labels
    colors = {
        'a': (255, 0, 0),    # Red
        'b': (0, 255, 0),    # Green
        'c': (0, 0, 255),    # Blue
        'd': (255, 255, 0),  # Yellow
        'e': (255, 0, 255),  # Magenta
        'f': (0, 255, 255),  # Cyan
        'g': (255, 128, 0),  # Orange
        'h': (128, 0, 255),  # Purple
        'i': (0, 255, 128),  # Teal
        'j': (255, 0, 128),  # Pink
    }
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("LiberationSans-Regular.ttf", 12)
        except (OSError, IOError):
            font = ImageFont.load_default()
    
    for mask_data in masks_data:
        mask = mask_data["mask"]
        label = str(mask_data["label"])
        
        # Use label-specific color if available, otherwise use default
        mask_color = colors.get(label.lower(), mask_color)
        
        # Create colored mask with gradient effect
        colored_mask = np.zeros_like(image_np)
        y_indices, x_indices = np.where(mask)
        if len(y_indices) > 0 and len(x_indices) > 0:
            centroid_y = int(np.mean(y_indices))
            centroid_x = int(np.mean(x_indices))
            
            # Create distance map for gradient effect
            y_dist = np.abs(y_indices - centroid_y)
            x_dist = np.abs(x_indices - centroid_x)
            dist = np.sqrt(y_dist**2 + x_dist**2)
            max_dist = np.max(dist)
            alpha_map = 1 - (dist / max_dist) * 0.5  # Gradient from center
            
            # Apply gradient to mask
            for i, (y, x) in enumerate(zip(y_indices, x_indices)):
                alpha = mask_alpha * alpha_map[i]
                colored_mask[y, x] = (*mask_color, int(alpha * 255))
        
        # Overlay mask with alpha blending
        image_np = cv2.addWeighted(
            image_np,
            1.0,
            colored_mask,
            mask_alpha,
            0,
        )
        
        # Draw label at mask centroid
        if label:
            y_indices, x_indices = np.where(mask)
            if len(y_indices) > 0 and len(x_indices) > 0:
                centroid_y = int(np.mean(y_indices))
                centroid_x = int(np.mean(x_indices))
                
                # Convert back to PIL for text drawing
                image_pil = Image.fromarray(image_np)
                draw = ImageDraw.Draw(image_pil)
                
                # Get text size
                text_bbox = draw.textbbox((0, 0), label, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                # Draw text background with rounded corners and shadow
                padding = 4
                shadow_offset = 2
                draw.rounded_rectangle(
                    (
                        centroid_x - text_width/2 - padding + shadow_offset,
                        centroid_y - text_height/2 - padding + shadow_offset,
                        centroid_x + text_width/2 + padding + shadow_offset,
                        centroid_y + text_height/2 + padding + shadow_offset
                    ),
                    radius=4,
                    fill=(0, 0, 0, 128)  # Semi-transparent black shadow
                )
                
                draw.rounded_rectangle(
                    (
                        centroid_x - text_width/2 - padding,
                        centroid_y - text_height/2 - padding,
                        centroid_x + text_width/2 + padding,
                        centroid_y + text_height/2 + padding
                    ),
                    radius=4,
                    fill=outline_color
                )
                
                # Draw text
                draw.text(
                    (centroid_x - text_width/2, centroid_y - text_height/2),
                    label,
                    fill=mask_color,
                    font=font
                )
                
                image_np = np.array(image_pil)
    
    return Image.fromarray(image_np)


def visualize_results(
    image: Image.Image,
    results: Dict[str, Any],
    point_radius: int = 4,
    box_width: int = 2,
    mask_alpha: float = 0.5,
) -> Image.Image:
    """Visualize results from Gemini response on an image."""
    # Draw points
    if "points" in results and results["points"]:
        points = results["points"]
        image = draw_points(
            image,
            points,
            point_radius=point_radius,
        )

    # Draw boxes with labels
    if "boxes" in results and results["boxes"]:
        boxes = results["boxes"]
        labels = results.get("labels", [])
        
        # Create BoxData objects
        box_data_list = []
        for i, box in enumerate(boxes):
            assert len(box) == 4, f"Invalid box format: {box}"
            # Use the label from the results, or generate a default one
            label = labels[i] if i < len(labels) else f"object_{i+1}"
            
            # Create BoxData with denormalized coordinates
            # The boxes in results are already denormalized, so we can use them directly
            box_data = BoxData(
                label=label,
                box_2d=box,  # These are already denormalized
                normalized_box=None,  # We don't need normalized coordinates
                denormalized_box=box  # Use the denormalized coordinates directly
            )
            
            box_data_list.append(box_data)
        
        # Only draw boxes if we have valid box data
        if box_data_list:
            image = draw_boxes(
                image,
                box_data_list,
                box_width=box_width,
            )

    # Draw masks
    if "masks" in results and results["masks"]:
        masks = results["masks"]
        image = draw_masks(
            image,
            masks,
            mask_alpha=mask_alpha,
        )

    return image 