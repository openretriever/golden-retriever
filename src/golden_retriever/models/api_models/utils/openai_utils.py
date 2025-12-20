import base64
import os
from typing import List, Optional

import cv2
import numpy as np
import openai
import ray
from PIL import Image
from rich import print
from tenacity import retry, stop_after_attempt, wait_random_exponential

from retriever.models.common_utils import Timer, timer


def set_openai_key(key: Optional[str] = None):
    if key is None:
        assert "OPENAI_API_KEY" in os.environ
        key = os.environ["OPENAI_API_KEY"]
    openai.api_key = key


def prepare_openai_messages(content: str):
    return [{"role": "user", "content": content}]


def prepare_openai_vision_messages(
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    image_paths: Optional[List[str]] = None,
    image_size: Optional[int] = 512,
):
    if image_paths is None:
        image_paths = []
    elif not isinstance(image_paths, list):
        image_paths = [image_paths]

    content = []

    if prefix:
        content.append({"text": prefix, "type": "text"})

    for path in image_paths:
        if not isinstance(path, str):
            print(f"Invalid image path: {path}")
            continue
        if not os.path.exists(path):
            print(f"Image file not found: {path}")
            continue

        frame = cv2.imread(path)
        if image_size:
            factor = image_size / max(frame.shape[:2])
            frame = cv2.resize(frame, dsize=None, fx=factor, fy=factor)
        _, buffer = cv2.imencode(".png", frame)
        frame = base64.b64encode(buffer).decode("utf-8")
        content.append(
            {
                "image_url": {"url": f"data:image/png;base64,{frame}"},
                "type": "image_url",
            }
        )

    if suffix:
        content.append({"text": suffix, "type": "text"})

    return [{"role": "user", "content": content}]


def prepare_openai_Image_messages(
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
        if not isinstance(image, Image.Image):
            print("Invalid image:")
            continue

        image = image.convert("RGB")
        open_cv_image = np.array(image)

        frame = open_cv_image[:, :, ::-1].copy()
        if image_size:
            factor = image_size / max(frame.shape[:2])
            frame = cv2.resize(frame, dsize=None, fx=factor, fy=factor)
        _, buffer = cv2.imencode(".png", frame)
        frame = base64.b64encode(buffer).decode("utf-8")
        content.append(
            {
                "image_url": {"url": f"data:image/png;base64,{frame}"},
                "type": "image_url",
            }
        )

    if suffix:
        content.append({"text": suffix, "type": "text"})

    return [{"role": "user", "content": content}]


def prepare_openai_Image_captioned_messages(
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    captions: Optional[List[str]] = None,
    images: Optional[List[Image.Image]] = None,
    image_size: Optional[int] = None,
):
    content = []

    if captions is None:
        captions = []

    if images is None:
        images = []
    elif not isinstance(images, list):
        images = [images]

    if len(captions) != len(images):
        raise ValueError("Number of captions must match number of images")


    if prefix:
        content.append({"text": prefix, "type": "text"})

    for image, caption in zip(images, captions):

        if not isinstance(image, Image.Image):
            print(f"Invalid image:")
            continue

        image = image.convert("RGB")
        open_cv_image = np.array(image)

        frame = open_cv_image[:, :, ::-1].copy()
        if image_size:
            factor = image_size / max(frame.shape[:2])
            frame = cv2.resize(frame, dsize=None, fx=factor, fy=factor)
        _, buffer = cv2.imencode(".png", frame)
        frame = base64.b64encode(buffer).decode("utf-8")
        #buffered = cv2.imencode(".png", np.array(image)[:, :, ::-1])[1]
        #frame = base64.b64encode(buffered).decode("utf-8")

        content.append(
            {
                "image_url": {"url": f"data:image/png;base64,{frame}"},
                "type": "image_url",
            }
        )
        #content.append({"text": caption, "type": "text"})

    if suffix:
        content.append({"text": suffix, "type": "text"})

    return [{"role": "user", "content": content}]

def prepare_system_messages(content: str):
    return [{"role": "system", "content": content}]

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_openai_api(
    messages: list,
    model: str = "gpt-4",
    seed: Optional[int] = None,
    max_tokens: int = 32,
    temperature: float = 0.2,
    verbose: bool = False,
):
    client = openai.OpenAI()
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            seed=seed,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if verbose:
            print("openai api response: {}".format(completion))
        assert len(completion.choices) == 1
        return completion.choices[0].message.content
    except openai.error.InvalidRequestError as e:
        print(f"Invalid request error: {e}")
        #print(f"Error details: {e.json_body}")
        raise


@ray.remote
def call_openai_api_ray(
    messages: list,
    model: str = "gpt-4",
    seed: Optional[int] = None,
    max_tokens: int = 32,
    temperature: float = 0.2,
    verbose: bool = False,
    api_key: Optional[str] = None,
):
    # with Timer(enable_print=verbose):
    client = openai.OpenAI(api_key=api_key)
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        seed=seed,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if verbose:
        print("openai api response: {}".format(completion))
    assert len(completion.choices) == 1
    return completion.choices[0].message.content


if __name__ == "__main__":
    # initialize local Ray cluster
    mode = "local"
    if mode == "local":
        ray.init()
    elif mode == "client":
        runtime_env = {
            "pip": [
                "tenacity",
                "openai",
            ],
        }
        # ray.init(address="ray://localhost:10001", runtime_env=runtime_env)
        ray.init(address="ray://128.30.227.158:10001", runtime_env=runtime_env)
    else:
        raise ValueError

    if mode == "local":
        set_openai_key(key=None)
    elif mode == "client":
        # NOTE: need to set up OpenAI key on the remote machine
        # set_openai_key_remote = ray.remote(set_openai_key)
        # ray.get(set_openai_key_remote.remote(key=os.environ["OPENAI_API_KEY"]))
        pass

    messages = prepare_openai_messages("What color are apples?")
    print("input:", messages)

    model = "gpt-4o"
    with Timer(enable_print=True) as timer:
        output = call_openai_api(messages, model=model, max_tokens=512, temperature=0.5)

    print(f"Elapsed time: {timer.get_elapsed_time():.4f} seconds")
    print("output: {}".format(output))

    with Timer(enable_print=True) as timer:
        output_future = [
            call_openai_api_ray.remote(
                _messages,
                model=model,
                max_tokens=512,
                temperature=x,
                verbose=False,
                # NOTE: pass local API key to remote Ray function
                api_key=os.environ["OPENAI_API_KEY"],
            )
            for x in [0.5]
            for _messages in [
                prepare_openai_messages("What color are apples?"),
                prepare_openai_messages("What color are oranges?"),
                prepare_openai_messages("What color are watermelons?"),
                prepare_openai_messages("What color are bananas?"),
                prepare_openai_messages("What color are pineapples?"),
            ]
        ]
        output = ray.get(output_future)

    print(f"Elapsed time: {timer.get_elapsed_time():.4f} seconds")
    print("output: {}".format(output))