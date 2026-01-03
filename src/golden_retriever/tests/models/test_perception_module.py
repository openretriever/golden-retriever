import numpy as np

from retriever.core.types import (
    NLCommand,
    ObjectDescriptionDict,
    Observation,
    RGBDImage,
    UnorderedObjectSet,
)
from retriever.models.perception_module import PerceptionModule
from retriever.models.segmentation.pointing_gemini_sam2_client import (
    PointingGeminiSAM2Client,
)


class MockPointingGeminiSAM2Client(PointingGeminiSAM2Client):
    """A mock client that returns a canned response."""

    def predict(self, *args, **kwargs):
        # This response simulates the detection of two distinct "red cup" objects
        # and one "blue mug" object.
        return {
            "results": [
                {
                    "labels": ["red cup", "blue mug", "red cup"],
                    # The other keys like "boxes" are not used by the module yet.
                }
            ],
            "timings": {},
        }


def test_perception_module_parsing():
    """
    Tests that the PerceptionModule correctly parses a mocked client response.
    """
    # 1. Set up the mock client and the module.
    # Pass dummy values for host/port since we're mocking the predict method.
    mock_client = MockPointingGeminiSAM2Client(host="dummy", port=0)
    perception_module = PerceptionModule(client=mock_client)

    # 2. Create dummy inputs with the new generic Observation structure.
    dummy_rgbd = RGBDImage(
        rgb=np.zeros((10, 10, 3), dtype=np.uint8), depth=np.zeros((10, 10))
    )
    dummy_obs = Observation(images={"front_camera": dummy_rgbd})
    dummy_command = NLCommand(text="find the cups")

    # 3. Call the module.
    object_set, object_descriptions = perception_module((dummy_obs, dummy_command))

    # 4. Assert that the output is parsed correctly.
    # The module should create unique IDs for each detected object instance.
    expected_ids = {"red_cup_0", "blue_mug_1", "red_cup_2"}
    assert object_set.objects == expected_ids

    expected_descriptions = {
        "red_cup_0": "red cup",
        "blue_mug_1": "blue mug",
        "red_cup_2": "red cup",
    }
    assert object_descriptions.descriptions == expected_descriptions 