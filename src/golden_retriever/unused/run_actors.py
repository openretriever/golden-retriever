import logging
import time

import ray
import ray.runtime_env
from omegaconf import DictConfig

from retriever.actors.memory_actor_base import EnvMemoryActorBase, RobotMemoryActorBase
from retriever.robots.robot_actor_base import ArmRobotActorBase, MobileArmRobotActorBase


def _start_actors(cfg: DictConfig, console):
    """
    Start the actors for the system
    """
    actors = {}  # Dictionary to hold actor handles
    actor_classes = {}  # Dictionary to hold actor classes

    # Instantiate and store actor handles
    # for actor_name, actor_cfg in cfg.actors.items():
    #     module_name, class_name = actor_cfg.classname.rsplit(".", 1)
    #     actor_class = get_class_by_name(module_name, class_name)
    #
    #     # actor_handle = actor_class.options(**actor_cfg.options).remote(**actor_cfg.params)
    #     actor_class = actor_class.options(**actor_cfg.options)
    #     actor_classes[actor_name] = actor_class

    # NOTE: Currently using manual initialization
    # Need to generate in order, as some actors may depend on others
    # 1. memory actors

    # Robot memory actor for storing robot state/perception data
    # NOTE: GPU allocation currently disabled, can be enabled by uncommenting the gpu option
    memory_actor1 = (
        ray.remote(RobotMemoryActorBase)
        # .options(num_cpus=1.0, num_gpus=0.1, max_concurrency=8)
        .options(num_cpus=1.0, num_gpus=0.0, max_concurrency=8).remote()
    )

    # Environment memory actor for storing environment state/map data
    # NOTE: GPU allocation currently disabled, can be enabled by uncommenting the gpu option
    map_actor1 = (
        ray.remote(EnvMemoryActorBase)
        # .options(num_cpus=1.0, num_gpus=0.3, max_concurrency=8)
        .options(num_cpus=1.0, num_gpus=0.0, max_concurrency=8).remote()
    )

    # 2. robot actors

    # Static arm robot actor initialized with memory actors
    arm_robot_actor = (
        ray.remote(ArmRobotActorBase)
        .options(num_cpus=1.0, max_concurrency=8)
        .remote(actors={"memory_actor": memory_actor1, "map_actor": map_actor1})
    )
    actors["arm_robot_actor1"] = arm_robot_actor

    # Mobile arm robot actor (robot with mobile base + arm) initialized with memory actors
    mobile_robot_actor = (
        ray.remote(MobileArmRobotActorBase)
        .options(num_cpus=1.0, max_concurrency=8)
        .remote(actors={"memory_actor": memory_actor1, "map_actor": map_actor1})
    )
    actors["mobile_robot_actor1"] = mobile_robot_actor

    # 3. environment actors

    # Start memory actors
    # for actor_name, actor_class in actor_classes.items():
    #     if "memory" in actor_name:
    #         actors[actor_name] = actor_class.remote(**actor_cfg.params)

    # NOTE: Initialize actor references
    pending_ids = list(actors.values())

    # Start continuous actors that have a run method
    for actor in actors.values():
        if hasattr(actor, "run"):
            actor.run.remote()

    try:
        while True:
            # console.clear()

            # ready_ids, pending_ids = ray.wait(pending_ids, timeout=0.2)
            # for ready_id in ready_ids:
            #     # result = ray.get(ready_id)
            #     pending_ids.remove(ready_id)
            #
            # for i, pending_id in enumerate(pending_ids):
            #     # console.print(f"[bold blue]Actor {i+1}[/]: Processing taskns: {pending_id}")
            #     logging.info(f"Actor {i+1}: Processing tasks: {pending_id}")

            # Log available Ray resources for monitoring
            logging.info(
                f"Running. Info: available resources {ray.available_resources()}"
            )
            time.sleep(5)

    except SystemExit:
        # console.print("[red]Shutting down...[/]")
        logging.warning("Shutting down Ray run...")
    finally:
        # Shutdown Ray when done
        ray.shutdown()

    return actors
