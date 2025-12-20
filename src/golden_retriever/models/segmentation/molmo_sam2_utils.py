import base64
import io
import os
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mask_util


def decode_base64_image(image_data: str) -> Image.Image:
    """Decode base64 string to PIL Image"""
    try:
        # Decode base64 string to bytes
        image_bytes = base64.b64decode(image_data)
        # Convert bytes to PIL Image
        return Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        raise ValueError(f"Failed to process image: {str(e)}")


def draw_mask(
    image: Image.Image, mask: np.ndarray, alpha: float = 0.5, color: tuple = (255, 0, 0)
) -> Image.Image:
    """Draw a binary mask on an image with transparency"""
    image = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Convert binary mask to image coordinates
    mask_image = Image.fromarray((mask * 255).astype("uint8"), "L")

    # Create colored overlay
    colored_overlay = Image.new("RGBA", image.size, (*color, int(255 * alpha)))
    overlay.paste(colored_overlay, mask=mask_image)

    return Image.alpha_composite(image, overlay).convert("RGB")


def draw_box(image: Image.Image, box: list) -> Image.Image:
    """Draw bounding box on image with confidence score

    Args:
        image: PIL Image to draw on
        box: List of [x1, y1, x2, y2, confidence, class_id]
    """
    try:
        draw = ImageDraw.Draw(image)
        if len(box) >= 6:  # [x1, y1, x2, y2, conf, class_id]
            x1, y1, x2, y2, conf, _ = box

            # Make line width proportional to image size
            width, height = image.size
            line_width = max(
                2, min(width, height) // 200
            )  # Minimum 2px, scales with image size

            # Draw rectangle
            draw.rectangle(
                [float(x1), float(y1), float(x2), float(y2)],
                outline="red",
                width=line_width,
            )

            # Scale font size based on image size
            font_size = max(
                12, min(width, height) // 50
            )  # Minimum 12px, scales with image size
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()

            # Draw confidence score
            conf_text = f"{conf:.2f}"
            draw.text(
                (
                    float(x1),
                    float(y1) - font_size - 4,
                ),  # Adjust position based on font size
                conf_text,
                fill="red",
                font=font,
                stroke_width=max(1, line_width // 2),
                stroke_fill="white",
            )
        return image
    except Exception as e:
        print(f"Error drawing box {box}: {str(e)}")
        return image
    
    
def draw_points(
    image: Image.Image, points: List[List[float]], color: str = "red"
) -> Image.Image:
    """Draw points on image with size proportional to image dimensions"""
    draw = ImageDraw.Draw(image)
    width, height = image.size

    # Scale point radius and outline width with image size
    point_radius = max(5, min(width, height) // 100)  # Minimum 5px radius
    outline_width = max(2, point_radius // 2)  # Minimum 2px outline

    for x, y in points:
        # Draw point with outline
        bbox = [x - point_radius, y - point_radius, x + point_radius, y + point_radius]
        draw.ellipse(bbox, fill=color, outline="white", width=outline_width)

    return image


def process_masks(
    masks_tensor: Any, verbose: bool = False
) -> Tuple[List[Dict], Dict[str, float]]:
    """Process mask tensor into RLE format with optional size analysis

    Args:
        masks_tensor: PyTorch tensor containing masks
        verbose: Whether to compute and return size analysis

    Returns:
        Tuple of (processed masks list, size analysis dict)
    """
    masks_np = masks_tensor.cpu().numpy()
    size_info = {}

    if verbose:
        size_info.update(
            {
                "shape": masks_np.shape,
                "dtype": str(masks_np.dtype),
                "num_masks": len(masks_np),
                "nonzero_first": np.count_nonzero(masks_np[0]),
            }
        )

    # Convert to RLE format
    rle_masks = [
        mask_util.encode(np.asfortranarray(mask.astype(np.uint8))) for mask in masks_np
    ]

    # Convert RLE format to be JSON serializable
    processed_masks = [
        {
            "size": [int(x) for x in rle["size"]],
            "counts": rle["counts"].decode()
            if isinstance(rle["counts"], bytes)
            else rle["counts"],
        }
        for rle in rle_masks
    ]

    if verbose:
        original_size = len(str(masks_tensor.tolist())) / 1024 / 1024
        compressed_size = len(str(processed_masks)) / 1024 / 1024
        rle_counts_size = sum(len(m["counts"]) for m in processed_masks) / 1024 / 1024

        size_info.update(
            {
                "original_mb": original_size,
                "compressed_mb": compressed_size,
                "rle_counts_mb": rle_counts_size,
                "compression_ratio": original_size / compressed_size,
            }
        )

    return processed_masks, size_info


def save_visualization(
    image: Image.Image,
    points: list,
    sam_result,
    img_idx: int,
    prompt_idx: int,
    save_dir: str = None,
) -> tuple:
    """Save visualization of points, boxes, and segmentation masks

    Args:
        image: Original image
        points: List of points
        sam_result: SAM model output
        img_idx: Image index
        prompt_idx: Prompt index
        save_dir: Directory to save visualizations (if None, uses temporary directory)

    Returns:
        Tuple of (points_path, boxes_path, masks_path)
    """
    # Use temporary directory if save_dir is not specified
    if save_dir is None:
        save_dir = tempfile.mkdtemp()

    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Generate filenames with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    points_filename = f"prompt{prompt_idx}_img{img_idx}_points_{timestamp}.png"
    boxes_filename = f"prompt{prompt_idx}_img{img_idx}_boxes_{timestamp}.png"
    masks_filename = f"prompt{prompt_idx}_img{img_idx}_masks_{timestamp}.png"

    points_path = None
    boxes_path = None
    masks_path = None

    # 1. Draw and save points visualization if points exist
    if points:
        points_path = os.path.join(save_dir, points_filename)
        points_img = image.copy()
        points_img = draw_points(points_img, points)
        points_img.save(points_path)

    # Only save detection/segmentation if we have valid results
    if points and hasattr(sam_result, "masks") and len(sam_result.masks) > 0:
        # 2. Draw and save boxes visualization
        if hasattr(sam_result, "boxes") and sam_result.boxes is not None:
            boxes_path = os.path.join(save_dir, boxes_filename)
            boxes_img = image.copy()
            for i, box in enumerate(sam_result.boxes.data):
                # Add confidence score and class_id (if available) to box data
                box_data = box.tolist()
                if hasattr(sam_result, "scores"):
                    box_data.append(float(sam_result.scores[i]))
                if hasattr(sam_result, "labels"):
                    box_data.append(int(sam_result.labels[i]))
                # Ensure box_data has at least 6 elements for draw_box function
                while len(box_data) < 6:
                    box_data.append(0.0)  # Add default values if needed
                boxes_img = draw_box(boxes_img, box_data)
            boxes_img.save(boxes_path)

        # 3. Draw and save masks visualization
        masks_path = os.path.join(save_dir, masks_filename)
        masks_img = image.copy()
        for i, mask in enumerate(sam_result.masks.data):
            color = (255, 0, 0) if i == 0 else (0, 255, 0)  # Different colors for multiple masks
            masks_img = draw_mask(masks_img, mask.cpu().numpy(), alpha=0.5, color=color)
        masks_img.save(masks_path)

    return points_path, boxes_path, masks_path


def encode_image(image_path: str) -> str:
    """Read image file and encode to base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def format_timing_summary(timings: dict, num_images: int, prompts: list) -> str:
    """Format timing summary for all operations"""
    summary = "\nTiming Summary:\n"
    
    # Calculate averages for base64 decoding
    base64_times = [
        t for k, t in timings.items() if k.startswith("base64_decode")
    ]
    if base64_times:
        avg_base64 = sum(base64_times) / len(base64_times)
        summary += f"Base64 decode (avg): {avg_base64:.3f}s\n"

    # Process each image-prompt combination
    for img_idx in range(num_images):
        for prompt_idx, prompt in enumerate(prompts):
            summary += f"\nImage {img_idx}, Prompt '{prompt}':\n"
            
            # Molmo inference timing
            molmo_key = f"molmo_inference_{img_idx}_{prompt_idx}"
            if molmo_key in timings:
                summary += f"  Molmo inference: {timings[molmo_key]:.3f}s\n"
            
            # SAM inference timing (only if it was performed)
            sam_key = f"sam_inference_{img_idx}_{prompt_idx}"
            if sam_key in timings:
                summary += f"  SAM inference: {timings[sam_key]:.3f}s\n"
            else:
                summary += "  SAM inference: skipped (no points detected)\n"

    return summary
