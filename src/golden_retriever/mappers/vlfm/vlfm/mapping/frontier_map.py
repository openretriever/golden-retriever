# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
from glob import glob
from typing import List, Tuple

import numpy as np

# from .....models.embeddings.blipv2imagetextmatch_actor import Blipv2ImageTextMatchingActor
import ray
import torch
from retriever.models.embeddings.blipv2imagetextmatch_actor import (
    Blipv2ImageTextMatchingActor,
)

# from vlfm.vlm.blip2itm import BLIP2ITMClient
from ..vlm.blip2itm import BLIP2ITMClientRay


class Frontier:
    def __init__(self, xyz: np.ndarray, cosine: float):
        self.xyz = xyz
        self.cosine = cosine


class FrontierMap:
    frontiers: List[Frontier] = []

    def __init__(self, encoding_type: str = "cosine"):
        ignore = glob("*/")
        ignore = [item for item in ignore if "src" not in item]
        ignore = ignore + glob("src/*/")
        ignore = [item for item in ignore if "models" not in item]
        ignore = ["\\" + item for item in ignore]
        ignore.append("\\.git\\")

        if not ray.is_initialized():
            if os.environ.get("RAY_CONNECT", "auto") == "auto":
                ray.init(runtime_env={"working_dir": ".", "excludes": ignore})
            else:
                ray.init(
                    address=os.environ.get("RAY_CONNECT", "auto"),
                    runtime_env={"working_dir": ".", "excludes": ignore},
                )

        use_gpu = torch.cuda.is_available()
        # pg = placement_group([{"GPU": 1}])
        actor_options = {"num_gpus": 0.20} if use_gpu else {}
        # actor_options = {"placement_group": pg, "placement_group_bundle_index": 0} if use_gpu else {}
        # strat = PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_bundle_index=0, )

        model = Blipv2ImageTextMatchingActor.options(**actor_options).remote(
            use_gpu=use_gpu
        )

        # model = Blipv2ImageTextMatchingActor.options(scheduling_strategy=strat).remote(use_gpu=use_gpu)
        self.encoder: BLIP2ITMClientRay = BLIP2ITMClientRay(model)
        # self.encoder: BLIP2ITMClient = BLIP2ITMClient()

    def reset(self) -> None:
        self.frontiers = []

    def update(
        self, frontier_locations: List[np.ndarray], curr_image: np.ndarray, text: str
    ) -> None:
        """
        Takes in a list of frontier coordinates and the current image observation from
        the robot. Any stored frontiers that are not present in the given list are
        removed. Any frontiers in the given list that are not already stored are added.
        When these frontiers are added, their cosine field is set to the encoding
        of the given image. The image will only be encoded if a new frontier is added.

        Args:
            frontier_locations (List[np.ndarray]): A list of frontier coordinates.
            curr_image (np.ndarray): The current image observation from the robot.
            text (str): The text to compare the image to.
        """
        # Remove any frontiers that are not in the given list. Use np.array_equal.
        self.frontiers = [
            frontier
            for frontier in self.frontiers
            if any(
                np.array_equal(frontier.xyz, location)
                for location in frontier_locations
            )
        ]

        # Add any frontiers that are not already stored. Set their image field to the
        # given image.
        cosine = None
        for location in frontier_locations:
            if not any(
                np.array_equal(frontier.xyz, location) for frontier in self.frontiers
            ):
                if cosine is None:
                    cosine = self._encode(curr_image, text)
                self.frontiers.append(Frontier(location, cosine))

    def _encode(self, image: np.ndarray, text: str) -> float:
        """
        Encodes the given image using the encoding type specified in the constructor.

        Args:
            image (np.ndarray): The image to encode.

        Returns:

        """
        return self.encoder.cosine(image, text)

    def sort_waypoints(self) -> Tuple[np.ndarray, List[float]]:
        """
        Returns the frontier with the highest cosine and the value of that cosine.
        """
        # Use np.argsort to get the indices of the sorted cosines
        cosines = [f.cosine for f in self.frontiers]
        waypoints = [f.xyz for f in self.frontiers]
        sorted_inds = np.argsort([-c for c in cosines])  # sort in descending order
        sorted_values = [cosines[i] for i in sorted_inds]
        sorted_frontiers = np.array([waypoints[i] for i in sorted_inds])

        return sorted_frontiers, sorted_values
