import unittest

import ray

from retriever.models.embeddings.clip_actor import CLIPActor


class RayClipTest(unittest.TestCase):
    def test_ray_clip_actor_server(self):
        # Initialize Ray
        # TODO Test: Replace with your Ray cluster's address; see `docs/run-system.md` for instructions
        ray.init("ray://localhost:10002")
        print(ray.available_resources())

        # Variable to control GPU usage
        use_gpu = True  # Set to True to use GPU, False to not use GPU
        # use_gpu = False  # Set to True to use GPU, False to not use GPU

        # Options dictionary to dynamically set num_gpus
        actor_options = {"num_gpus": 1} if use_gpu else {}

        # Create an actor instance with dynamic GPU allocation
        clip_actor = CLIPActor.options(**actor_options).remote(use_gpu=use_gpu)

        print(clip_actor)

        ray.shutdown()

    # a version with testing locally
    def test_ray_clip_actor_local(self):
        # Initialize Ray
        ray.init()
        print(ray.available_resources())

        # Variable to control GPU usage
        use_gpu = False

        # Options dictionary to dynamically set num_gpus
        actor_options = {"num_gpus": 1} if use_gpu else {}

        # Create an actor instance with dynamic GPU allocation
        clip_actor = CLIPActor.options(**actor_options).remote(use_gpu=use_gpu)

    def test_ray_clip_actor_local_ray_mode(self):
        # Initialize Ray
        ray.init("ray://localhost:10002", local_mode=True)
        print(ray.available_resources())

        # Variable to control GPU usage
        use_gpu = False

        # Options dictionary to dynamically set num_gpus
        actor_options = {"num_gpus": 1} if use_gpu else {}

        # Create an actor instance with dynamic GPU allocation
        clip_actor = CLIPActor.options(**actor_options).remote(use_gpu=use_gpu)


if __name__ == "__main__":
    unittest.main()
