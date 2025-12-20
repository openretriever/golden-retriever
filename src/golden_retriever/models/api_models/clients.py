from __future__ import annotations

import base64
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, Union, cast

import cv2
import google.generativeai as genai
import numpy as np
import openai
from google.generativeai.generative_models import GenerativeModel
from google.generativeai.types import GenerationConfig
from PIL import Image, ImageDraw, ImageFont
from tenacity import retry, stop_after_attempt, wait_random_exponential

# ##################################################################
# #################### LLM CLIENT ABSTRACTIONS #####################
# ##################################################################


class LLMClient(Protocol):
    """
    A protocol for a client that can interact with a large language model.
    """

    def predict(
        self, prompt: str, image: Optional[Union[str, Image.Image]] = None
    ) -> str | None:
        """
        Given a prompt and an optional image, returns the model's prediction.
        """
        ...


# ##################################################################
# #################### OPENAI IMPLEMENTATION #######################
# ##################################################################


def _set_openai_key(key: Optional[str] = None):
    if key is None:
        key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY environment variable must be set.")
    openai.api_key = key


def _prepare_openai_messages(content: str):
    return [{"role": "user", "content": content}]


def _prepare_openai_Image_messages(
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    images: Optional[List[Image.Image]] = None,
    image_size: Optional[int] = 512,
):
    if images is None:
        images = []
    elif not isinstance(images, list):
        images = [images]

    content = []
    if prefix:
        content.append({"text": prefix, "type": "text"})

    for image in images:
        image = image.convert("RGB")
        frame = np.array(image)[:, :, ::-1].copy()
        if image_size:
            factor = image_size / max(frame.shape[:2])
            frame = cv2.resize(frame, dsize=None, fx=factor, fy=factor)
        _, buffer = cv2.imencode(".png", frame)
        frame_b64 = base64.b64encode(buffer).decode("utf-8")
        content.append(
            {
                "image_url": {"url": f"data:image/png;base64,{frame_b64}"},
                "type": "image_url",
            }
        )

    if suffix:
        content.append({"text": suffix, "type": "text"})

    return [{"role": "user", "content": content}]


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def _call_openai_api(messages: list, model: str, **kwargs):
    client = openai.OpenAI()
    try:
        completion = client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )
        assert len(completion.choices) == 1
        return completion.choices[0].message.content
    except openai.InvalidRequestError as e:
        logging.error(f"Invalid request error: {e}")
        raise


class OpenAIClient(LLMClient):
    """An LLM client implementation for OpenAI models."""

    def __init__(self, model: str = "gpt-4o", **kwargs):
        self._model = model
        self._kwargs = kwargs
        _set_openai_key()

    def predict(
        self, prompt: str, image: Optional[Union[str, Image.Image]] = None
    ) -> str | None:
        if image is not None:
            pil_image = Image.open(image) if isinstance(image, str) else image
            messages = _prepare_openai_Image_messages(prefix=prompt, images=[pil_image])
        else:
            messages = _prepare_openai_messages(prompt)
        response = _call_openai_api(
            messages, model=self._model, temperature=0.0, **self._kwargs
        )
        return response


# ##################################################################
# ##################### GEMINI IMPLEMENTATION ######################
# ##################################################################

GEMINI_MODEL_NAME = "gemini-1.5-flash"


class GeminiClient(LLMClient):
    """Client for interacting with Google Gemini models."""

    def __init__(self, api_key: Optional[str] = None, model: str = GEMINI_MODEL_NAME, **kwargs):
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY must be set or passed to the constructor."
            )
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model)
        self._config = GenerationConfig(**kwargs)


    def predict(
        self, prompt: str, image: Optional[Union[str, Image.Image]] = None
    ) -> str | None:
        """Generate a response from Gemini."""
        try:
            content: List[Union[str, Image.Image]] = [prompt]
            if image is not None:
                pil_image = Image.open(image) if isinstance(image, str) else image
                content.append(pil_image)
            
            response = self._model.generate_content(content, generation_config=self._config)
            response.resolve()
            return response.text
        except Exception as e:
            logging.error(f"Gemini API call failed: {type(e).__name__}: {e}")
            raise


# ##################################################################
# #################### LLM CLIENT FACTORY ##########################
# ##################################################################


def create_llm_client(client_type: str = "openai", **kwargs) -> LLMClient:
    """
    Factory function to create an LLM client based on the specified type.

    Args:
        client_type: The type of client to create ("openai" or "gemini").
        **kwargs: Additional arguments to pass to the client constructor.

    Returns:
        An instance of a class that implements the LLMClient protocol.
    """
    if client_type.lower() == "openai":
        return OpenAIClient(**kwargs)
    elif client_type.lower() == "gemini":
        return GeminiClient(**kwargs)
    else:
        raise ValueError(f"Unknown LLM client type: '{client_type}'")


# ##################################################################
# #################### GEMINI PARSING UTILS ########################
# ##################################################################


@dataclass
class PointData:
    """Data class for storing point information."""

    label: str
    normalized_point: List[float]
    denormalized_point: Optional[List[float]] = None

    @classmethod
    def from_gemini_response(
        cls, response: Dict[str, Union[List[float], str]]
    ) -> "PointData":
        point = cast(List[float], response["point"])
        y, x = point
        if not (0 <= y <= 1000 and 0 <= x <= 1000):
            raise ValueError(f"Point coordinates out of range: {point}")
        return cls(label=str(response["label"]), normalized_point=point)

    def denormalize(self, image_size: Tuple[int, int]) -> None:
        y, x = self.normalized_point
        self.denormalized_point = [y * image_size[1] / 1000, x * image_size[0] / 1000]


@dataclass
class BoxData:
    """Data class for storing box information."""

    label: str
    box_2d: List[float]
    denormalized_box: Optional[List[float]] = None

    @classmethod
    def from_gemini_response(
        cls, response: Dict[str, Union[List[float], str]]
    ) -> "BoxData":
        box_2d = cast(List[float], response.get("box_2d"))
        if not all(0 <= coord <= 1000 for coord in box_2d):
            raise ValueError(f"Box coordinates out of range: {box_2d}")
        return cls(label=str(response["label"]), box_2d=box_2d, denormalized_box=box_2d)

    def denormalize(self, image_size: Tuple[int, int]) -> None:
        y1, x1, y2, x2 = self.box_2d
        self.denormalized_box = [
            y1 * image_size[1] / 1000,
            x1 * image_size[0] / 1000,
            y2 * image_size[1] / 1000,
            x2 * image_size[0] / 1000,
        ]

def parse_gemini_response_to_json(response: str) -> List[Dict]:
    """Extracts a JSON array from a string response."""
    try:
        start = response.find("[")
        end = response.rfind("]") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON array found in response")
        json_str = response[start:end]
        return json.loads(json_str)
    except (json.JSONDecodeError, ValueError) as e:
        logging.error(f"Failed to parse JSON from response: {e}\nResponse: {response}")
        return []

def parse_gemini_point_response(
    response: str, image_size: Optional[Tuple[int, int]] = None
) -> Dict[str, List[PointData]]:
    """Parse Gemini point response to extract point coordinates and labels."""
    points_data = parse_gemini_response_to_json(response)
    points = [PointData.from_gemini_response(p) for p in points_data]
    if image_size:
        for p in points:
            p.denormalize(image_size)
    return {"points": points}


def parse_gemini_detection_response(
    response: str, image_size: Optional[Tuple[int, int]] = None
) -> Dict[str, List[BoxData]]:
    """Parse Gemini detection response to extract bounding boxes and labels."""
    detection_data = parse_gemini_response_to_json(response)
    boxes = [BoxData.from_gemini_response(d) for d in detection_data]
    if image_size:
        for b in boxes:
            b.denormalize(image_size)
    return {"detections": boxes}

# ... (Visualization functions can also be moved here if desired) ... 