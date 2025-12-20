import base64
import io

import numpy as np
from PIL import Image

VLM_MODEL_LIST = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-vision-preview",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-2.0-flash-exp",
    "claude-3-5-sonnet-latest",
    "claude-3-opus-20240229",
]


def tensor_to_base64_image(tensor):
    """
    Converts a tensor representing an image into a base64-encoded PNG image string.

    Parameters:
        tensor: PyTorch tensors

    Returns:
        A base64-encoded string of the image in PNG format.
    """
    array = tensor.detach().cpu().numpy()

    # Ensure the array is in H x W x C format
    if array.ndim == 3:
        # If channels are first, transpose to channels last
        if array.shape[0] in [1, 3, 4]:
            array = np.transpose(array, (1, 2, 0))
    elif array.ndim == 2:
        # If grayscale image, expand dimensions
        array = np.expand_dims(array, axis=2)
    else:
        raise ValueError("Array must be a 2D or 3D tensor representing an image.")

    # Handle data type and scaling
    if array.dtype != np.uint8:
        # Assume the data is in [0, 1] and scale to [0, 255]
        array = (array * 255).clip(0, 255).astype(np.uint8)

    # Convert to image
    image = Image.fromarray(array)

    # Encode image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    base64_image = base64.b64encode(img_bytes).decode("utf-8")

    return base64_image
