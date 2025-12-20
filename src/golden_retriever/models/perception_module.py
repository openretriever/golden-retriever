from __future__ import annotations
from PIL import Image

from retriever.core.types import (
    Module,
    NLCommand,
    ObjectDescriptionDict,
    Observation,
    UnorderedObjectSet,
)
from retriever.models.segmentation.pointing_gemini_sam2_client import (
    PointingGeminiSAM2Client,
)


class PerceptionModule(
    Module[tuple[Observation, NLCommand], tuple[UnorderedObjectSet, ObjectDescriptionDict]]
):
    """
    A module that uses a perception client to detect and describe objects.
    """

    def __init__(self, client: PointingGeminiSAM2Client):
        self._client = client

    def __call__(
        self, inp: tuple[Observation, NLCommand]
    ) -> tuple[UnorderedObjectSet, ObjectDescriptionDict]:
        """
        Takes an observation and a natural language command, and returns
        the set of detected objects and their descriptions.
        """
        obs, command = inp

        if not obs.images:
            raise ValueError("Observation contains no images.")

        # Take the first image from the observation dict for processing.
        # A more robust implementation might look for a specific camera name.
        first_image = next(iter(obs.images.values()))
        
        # 1. Convert observation to an image format the client can use.
        pil_image = Image.fromarray(first_image.rgb)

        # 2. Call the client's predict method with detection enabled.
        # We assume a single image and single prompt for now.
        client_output = self._client.predict(
            images=pil_image,
            prompts=[command.text],
            points=False,
            segmentation=False,
            detection=True,
            save_visualization=False,
        )

        # 3. Parse the client's JSON output.
        detected_object_ids = set()
        object_descriptions = {}

        # The result is a list of results per image/prompt pair.
        # We only sent one pair, so we just look at the first result.
        if not client_output.get("results"):
            return UnorderedObjectSet(detected_object_ids), ObjectDescriptionDict(
                object_descriptions
            )

        result = client_output["results"][0]
        labels = result.get("labels", [])

        # 4. Populate and return UnorderedObjectSet and ObjectDescriptionDict.
        for i, label in enumerate(labels):
            # Create a unique ID for each detected object instance.
            object_id = f"{label.replace(' ', '_')}_{i}"
            detected_object_ids.add(object_id)
            object_descriptions[object_id] = label

        return UnorderedObjectSet(detected_object_ids), ObjectDescriptionDict(
            object_descriptions
        ) 