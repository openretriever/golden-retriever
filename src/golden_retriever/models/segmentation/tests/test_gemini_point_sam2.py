import os
from pathlib import Path

import pytest
import ray
from PIL import Image

from retriever.models.segmentation.gemini_point_sam2 import GeminiPointActor


@pytest.fixture(scope="module")
def ray_init():
    """Initialize Ray for testing."""
    ray.init()
    yield
    ray.shutdown()


@pytest.fixture
def test_image():
    """Create a test image."""
    return Image.new("RGB", (512, 512), "white")


@pytest.fixture
def gemini_actor(ray_init):
    """Create a Gemini point actor."""
    return GeminiPointActor.options(num_gpus=0).remote(use_gpu=False)


def test_gemini_point_only(gemini_actor, test_image):
    """Test Gemini point capability without segmentation."""
    prompt = "Point at the center of the image"
    result = ray.get(
        gemini_actor.predict.remote(
            test_image, prompt, points=True, segmentation=False
        )
    )
    assert "points" in result
    assert isinstance(result["points"], list)
    for point_data in result["points"]:
        assert "point" in point_data
        assert "label" in point_data
        assert isinstance(point_data["point"], list)
        assert len(point_data["point"]) == 2
        assert all(0 <= x <= 1000 for x in point_data["point"])


def test_gemini_point_with_segmentation(gemini_actor, test_image):
    """Test Gemini point capability with segmentation."""
    prompt = "Point at the center of the image"
    result = ray.get(
        gemini_actor.predict.remote(
            test_image, prompt, points=True, segmentation=True
        )
    )
    assert "points" in result
    assert "segmentation" in result
    assert isinstance(result["points"], list)
    assert isinstance(result["segmentation"], list)
    assert len(result["points"]) == len(result["segmentation"])
    for point_data in result["points"]:
        assert "point" in point_data
        assert "label" in point_data
        assert isinstance(point_data["point"], list)
        assert len(point_data["point"]) == 2
        assert all(0 <= x <= 1000 for x in point_data["point"])
    for mask in result["segmentation"]:
        assert isinstance(mask, Image.Image)


def test_gemini_segmentation_only(gemini_actor, test_image):
    """Test Gemini point capability with segmentation only."""
    prompt = "Point at the center of the image"
    result = ray.get(
        gemini_actor.predict.remote(
            test_image, prompt, points=False, segmentation=True
        )
    )
    assert isinstance(result, list)
    for mask in result:
        assert isinstance(mask, Image.Image)


def test_gemini_invalid_image(gemini_actor):
    """Test Gemini point capability with invalid image."""
    prompt = "Point at the center of the image"
    with pytest.raises(Exception):
        ray.get(
            gemini_actor.predict.remote(
                "invalid_path.jpg", prompt, points=True, segmentation=False
            )
        )


def test_gemini_invalid_prompt(gemini_actor, test_image):
    """Test Gemini point capability with invalid prompt."""
    with pytest.raises(Exception):
        ray.get(
            gemini_actor.predict.remote(
                test_image, "", points=True, segmentation=False
            )
        )


def test_draw_points(gemini_actor, test_image):
    """Test point visualization."""
    prompt = "Point at the center of the image"
    result = ray.get(
        gemini_actor.predict.remote(
            test_image, prompt, points=True, segmentation=False
        )
    )
    visualized = ray.get(gemini_actor.draw_points.remote(test_image, result))
    assert isinstance(visualized, Image.Image)
    assert visualized.size == test_image.size 