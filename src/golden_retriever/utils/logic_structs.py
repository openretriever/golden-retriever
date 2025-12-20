import logging
import time


class ContinuousActorBase:
    """
    A base class for continuous actors. There is a method `run()` that runs continuously and processes tasks from a
    task queue.

    Note: See https://docs.ray.io/en/latest/ray-core/actors/async_api.html
    "Keep in mind that the Python’s Global Interpreter Lock (GIL) will only allow one thread of Python code running
    at once."
    "This means if you are just parallelizing Python code, you won’t get true parallelism. If you call Numpy, Cython,
    Tensorflow, or PyTorch code, these libraries will release the GIL when calling into C/C++ functions."
    "Neither the Threaded Actors nor AsyncIO for Actors model will allow you to bypass the GIL."
    """

    # task_queue: Queue[Any]

    def __init__(self):
        # self.task_queue = asyncio.Queue()
        # self.task_queue = collections.queue()
        self.task_queue = []
        pass

    async def run(self):
        while True:
            # task = await self.task_queue.get()
            task = self.task_queue.pop()
            logging.info(f"Start a task: {task}")

            try:
                # Process the task
                logging.info("Processing a task!")
                await self.process_task(task)
            except Exception as e:
                # Handle exception, log error, etc.
                logging.error(e)
                time.sleep(1)

    async def process_task(self, task):
        # Define in subclass: actual processing of a task
        raise NotImplementedError

    async def get_status(self):
        # Define in subclass: return status of the actor
        return {
            "status": "running" if len(self.task_queue) > 0 else "idle",
            "task_queue_size": self.task_queue.qsize(),
        }

    def add_task(self, task):
        # This method is synchronous and immediately returns to the caller
        self.task_queue.put_nowait(task)


class OnDemandActorBase:
    def __init__(self):
        # Initialization for on-demand actor
        pass

    def perform_action(self, action):
        # Perform the action and return the result
        pass
