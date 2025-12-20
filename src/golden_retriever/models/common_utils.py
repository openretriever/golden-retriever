import time
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from PIL import Image
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks


def timer(enable_print=True):
    """
    Timer decorator to measure the execution time of a function.
    Args:
        enable_print:
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            if enable_print:
                print(
                    f"Function {func.__name__!r} executed in {(end_time - start_time):.4f} seconds"
                )
            return result

        return wrapper

    return decorator


class Timer:
    """
    Timer context manager to measure the execution time of a block of code.
    """

    def __init__(self, enable_print=True, name="Block"):
        self.enable_print = enable_print
        self.name = name
        # self.elapsed_time = 0  # Initialize elapsed_time

    def get_elapsed_time(self):
        """Get the elapsed time in seconds even in the context."""
        return time.time() - self.start_time

    def __enter__(self):
        self.start_time = time.time()
        return self  # Returning self to access its attributes

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Get the elapsed time when exiting the context."""
        self.elapsed_time = self.get_elapsed_time()
        if self.enable_print:
            print(f"[{self.name}] executed in {self.elapsed_time:.4f} seconds")

    @staticmethod
    def get_current_time():
        from datetime import datetime

        return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def download_image(url):
    response = requests.get(url)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def load_imagefile(image):
    image_pil = Image.open(image).convert("RGB")

    return image_pil


def load_image(image):
    if image.startswith("http"):
        image_pil = download_image(image)
    else:
        image_pil = Image.open(image).convert("RGB")

    return image_pil


def save_mask(mask_np, filename):
    mask_image = Image.fromarray((mask_np * 255).astype(np.uint8))
    mask_image.save(filename)


def save_image(image_np, filename):
    image_np.save(filename)


def display_image(image, title=None):
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title("Image: " + ("" if title is None else title))
    ax.axis("off")
    plt.show()


def display_image_with_masks(image, masks):
    num_masks = len(masks)

    fig, axes = plt.subplots(1, num_masks + 1, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    for i, mask_np in enumerate(masks):
        axes[i + 1].imshow(mask_np, cmap="gray")
        axes[i + 1].set_title(f"Mask {i+1}")
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.show()


def display_image_with_boxes(image, boxes, logits, phrases):
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title("Image with Bounding Boxes")
    ax.axis("off")

    for box, logit, phrases in zip(boxes, logits, phrases, strict=False):
        x_min, y_min, x_max, y_max = box
        confidence_score = round(
            logit.item(), 2
        )  # Convert logit to a scalar before rounding
        box_width = x_max - x_min
        box_height = y_max - y_min

        # Draw bounding box
        rect = plt.Rectangle(
            (x_min, y_min),
            box_width,
            box_height,
            fill=False,
            edgecolor="red",
            linewidth=1,
        )
        ax.add_patch(rect)

        # Add confidence score as text
        ax.text(
            x_min,
            y_min,
            f"{phrases}: {confidence_score}",
            fontsize=8,
            color="white",
            verticalalignment="top",
        )

    plt.show()


def draw_image(image, masks, boxes, labels, alpha=0.4):
    image = torch.from_numpy(image).permute(2, 0, 1)
    if len(boxes) > 0:
        image = draw_bounding_boxes(
            image, boxes, colors=["red"] * len(boxes), labels=labels, width=2
        )
    if len(masks) > 0:
        image = draw_segmentation_masks(
            image, masks=masks, colors=["cyan"] * len(masks), alpha=alpha
        )
    return image.numpy().transpose(1, 2, 0)


def print_bounding_boxes(boxes):
    print("Bounding Boxes:")
    for i, box in enumerate(boxes):
        print(f"Box {i+1}: {box}")


def print_detected_phrases(phrases):
    print("\nDetected Phrases:")
    for i, phrase in enumerate(phrases):
        print(f"Phrase {i+1}: {phrase}")


def print_logits(logits):
    print("\nConfidence:")
    for i, logit in enumerate(logits):
        print(f"Logit {i+1}: {logit}")
