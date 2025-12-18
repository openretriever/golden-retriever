"""
Retriever Advanced Example: Web Command Interface for VLA

This demo allows sending natural language instructions to a mock VLA model
via a web-based chat interface.

How to run:
    1. Hybrid Mode (Immediate triggers + 1Hz heartbeat):
       pixi run demo-web --mode hybrid

    2. Queue Mode (Buffered commands @ 1Hz):
       pixi run demo-web --mode queue

    3. Optional Backend Selection:
       pixi run demo-web --mode hybrid --backend multiprocessing
"""
import os
import sys
import queue
import time
import threading
import argparse
from dataclasses import dataclass

# Ensure the project root and src/ are in sys.path so we can import 'retriever'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
src_root = os.path.join(project_root, "src")
for path in [src_root, project_root]:
    if path not in sys.path:
        sys.path.insert(0, path)

# -- Imports --
from retriever.flow import (
    Pipeline, Rate, Hybrid,
    Flow, flow_io, 
    Events
)

# =============================================================================
# 1. Schema Definition
# =============================================================================

@flow_io
@dataclass
class VLACommand:
    """A natural language instruction for a VLA model."""
    instruction: str
    timestamp: float

# =============================================================================
# 2. Web Interface Source
# =============================================================================

# These are expected to be in the environment.
# If not, the user should: pixi run -e dev python -m pip install fastapi uvicorn
try:
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel
except ImportError:
    # Initial fallback definition
    class FastAPI: pass
    class BaseModel: pass

app = FastAPI()

class CommandRequest(BaseModel):
    command: str

# Shared state between FastAPI and Flow (thread-safe queue)
_command_queue = queue.Queue()

@app.post("/command")
async def receive_command(req: CommandRequest):
    # Enqueue the command with the current timestamp
    cmd_data = (req.command, time.time())
    _command_queue.put(cmd_data)
    return {"status": "ok", "command": req.command, "queue_size": _command_queue.qsize()}

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    # Determine the path relative to the root or current file
    static_file = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(static_file, "r") as f:
        return f.read()

class WebInstructionFlow(Flow[None, VLACommand]):
    """
    Source node that starts a background FastAPI server.
    Emits a VLACommand whenever a new command arrives via the web UI.
    Blocks until a command is available (Event Source behavior).
    """
    def init(self):
        # Start server in a background thread
        # We disable colors and use a simple log config to avoid 'isatty' errors
        # when stdout is redirected to the Retriever logger.
        self.thread = threading.Thread(
            target=uvicorn.run,
            args=(app,),
            kwargs={
                "host": "0.0.0.0", 
                "port": 8000, 
                "log_level": "warning",
                "use_colors": False,
                "log_config": None 
            },
            daemon=True
        )
        self.thread.start()
        print(f"\n[WebInstructionFlow] Web UI started at http://localhost:8000")
        print(f"[WebInstructionFlow] Waiting for commands...\n")

    def run(self, _):
        # Block until a command is available.
        # This ensures we only emit when there is actual data, 
        # allowing downstream Trigger/Hybrid clocks to work correctly.
        while True:
            try:
                # Use a timeout to allow the loop to check for shutdown signals/interrupts if needed,
                # though in this simple demo rely on daemon thread/process kill.
                instruction, timestamp = _command_queue.get(timeout=0.5)
                return VLACommand(instruction=instruction, timestamp=timestamp)
            except queue.Empty:
                continue

# =============================================================================
# 3. VLA Mock Sink
# =============================================================================

class VLAMock(Flow[VLACommand, None]):
    """
    A mock VLA model that maintains an internal 'current task' state.
    It can receive either a single VLACommand (Immediate mode) 
    or a list of VLACommand objects (Queue mode with Events adapter).
    """
    def init(self):
        self.current_task = "Searching for objects..."
        self.heartbeat_count = 0
        self.last_processed_timestamp = 0.0

    def run(self, commands):
        """
        Receives input which can be a single command or a list of commands.
        """
        valid_commands = []
        
        # Polymorphic handling based on input type
        if isinstance(commands, list):
            # Received a batch (Events adapter)
            valid_commands = [c for c in commands if c is not None and isinstance(c, VLACommand) and c.timestamp is not None]
        
        elif isinstance(commands, VLACommand):
            # Received a single item (Latest/Trigger)
            # Check if it's a valid command (timestamp is mandatory)
            if commands.timestamp is not None:
                # Check for SoA (list of values) just in case
                if isinstance(commands.timestamp, list):
                     for i, t in zip(commands.instruction, commands.timestamp):
                         valid_commands.append(VLACommand(instruction=i, timestamp=t))
                else:
                    valid_commands.append(commands)
        
        # Sort by timestamp
        # Filter out None values just in case
        valid_commands = [c for c in valid_commands if c is not None]
        valid_commands.sort(key=lambda x: x.timestamp)
        
        new_instruction_received = False
        
        for cmd in valid_commands:
            # Timestamp check to ensure we don't re-process old commands
            if cmd.timestamp > self.last_processed_timestamp:
                print(f"\n[VLA] <<< Received New instruction: '{cmd.instruction}'")
                self.current_task = cmd.instruction
                self.last_processed_timestamp = cmd.timestamp

        # Print the "Action Generation" on every tick (1Hz or Trigger)
        print(f"[VLA] Processing... Current Task: '{self.current_task}'")
        
        return None

# =============================================================================
# 4. Pipeline Construction
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Web Command Interface Demo")
    parser.add_argument("--mode", choices=["queue", "hybrid"], default="queue",
                        help="Mode of operation: 'queue' (buffered @ 1Hz) or 'hybrid' (1Hz + Event-driven)")
    parser.add_argument("--backend", choices=["dora", "multiprocessing"], default="dora",
                        help="Runtime backend to use")
    args = parser.parse_args()

    # Create the pipeline
    pipe = Pipeline("web_vla_advanced_demo")
    
    with pipe:
        # 1. Source: Web UI
        # We give it a high max rate, but it blocks on data, so effective rate = data rate.
        web = WebInstructionFlow() @ Rate(hz=100)
        
        # 2. Sink: VLA Mock
        if args.mode == "queue":
            print("[Setup] Configuring QUEUE mode (buffering commands, VLA running at 1Hz)...")
            # Run VLA slowly (1Hz)
            vla = VLAMock() @ Rate(hz=1)
            # Connect with buffer
            pipe.connect(web, vla, sync=Events(buffer_size=50, include_timestamps=False))
            
        else: # hybrid
            print("[Setup] Configuring HYBRID mode (VLA triggered by command OR 1Hz heartbeat)...")
            # Run VLA on 1Hz heartbeat AND whenever 'web' emits (trigger)
            # Trigger on the 'instruction' field of the input VLACommand
            vla = VLAMock() @ Hybrid(hz=1, trigger=["instruction"])
            # Connect normally (Latest value transfer)
            pipe.connect(web, vla)
        
    print("=" * 60)
    print(f"Retriever Advanced Example: Web Command Interface ({args.mode.upper()} mode)")
    print("=" * 60)
    print(f"\nStarting execution ({args.backend} backend)...")
    print("Open your browser at: http://localhost:8000")
    print("Press Ctrl+C to stop.\n")
    
    try:
        pipe.run(backend=args.backend)
    except KeyboardInterrupt:
        print("\nStopping demo...")

if __name__ == "__main__":
    main()
