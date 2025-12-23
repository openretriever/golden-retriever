"""
Threaded Code Executor for Code as Policies.

Ref: "Code as Policies", Liang et al. 2022
"""

import threading
import queue
import time
import logging
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

@dataclass
class ExecutionRequest:
    """A request from the policy thread to the environment."""
    command: str  # "pick", "place", "move_to", "say", "get_object_position"
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    event: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: Exception | None = None

class PolicyExecutor:
    """
    Executes Python code in a separate thread to allow blocking calls.
    Bridged to the main loop via a request queue.
    """

    def __init__(self):
        self.request_queue = queue.Queue()
        self.execution_thread = None
        self.running = False
        self._stop_event = threading.Event()

    def start_execution(self, code: str, context: dict = None):
        """Start executing the code in a new thread."""
        if self.execution_thread and self.execution_thread.is_alive():
            logger.warning("Code already running, stopping previous...")
            self.stop()
        
        self._stop_event.clear()
        self.execution_thread = threading.Thread(
            target=self._run_code,
            args=(code, context),
            daemon=True
        )
        self.execution_thread.start()
        self.running = True

    def stop(self):
        """Stop execution."""
        self._stop_event.set()
        # We can't force kill logic threads easily in Python easily, 
        # but the API calls check _stop_event.
        if self.execution_thread:
            self.execution_thread.join(timeout=1.0)
        self.running = False

    def _run_code(self, code: str, context: dict = None):
        """Internal runner."""
        logger.info("[Executor] Thread started.")
        
        # 1. Define API bridges
        def check_stop():
            if self._stop_event.is_set():
                raise InterruptedError("Execution stopped by user.")

        def _bridge_call(command: str, *args, **kwargs):
            check_stop()
            req = ExecutionRequest(command=command, args=args, kwargs=kwargs)
            self.request_queue.put(req)
            
            # Wait for main thread to process
            while not req.event.is_set():
                if self._stop_event.is_set():
                    raise InterruptedError("Execution stopped while waiting.")
                time.sleep(0.01)
            
            if req.error:
                raise req.error
            return req.result

        # 2. Build Safe Global Scope
        nav_api = {
            "pick": lambda name: _bridge_call("pick", name),
            "place": lambda loc: _bridge_call("place", loc),
            "move_to": lambda loc: _bridge_call("move_to", loc),
            "say": lambda msg: _bridge_call("say", msg),
            "get_object_position": lambda name: _bridge_call("get_object_position", name),
            "print": lambda *args: logger.info(f"[Script] {' '.join(map(str, args))}"),
        }

        # 3. Execute
        try:
            exec(code, {"__builtins__": {}}, nav_api)
            logger.info("[Executor] Script finished successfully.")
        except Exception as e:
            logger.error(f"[Executor] Script error: {e}")
            logger.error(traceback.format_exc())
            _bridge_call("say", f"Error: {e}")
        finally:
            self.running = False
