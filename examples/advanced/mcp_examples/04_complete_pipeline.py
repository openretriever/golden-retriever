"""
MCP Integration Pattern D: Complete Pipeline

This example demonstrates how to integrate MCP flows into a full Retriever Pipeline.
It uses the canonical `Pipeline` API to build and run the graph.

Architecture:
    Source (Rate 0.5Hz) -> Planner (Logic) -> MCP Memory (Side Effect) -> Sink (Logging)

Run with:
    pixi run python examples/advanced/mcp_examples/04_complete_pipeline.py
"""

from dataclasses import dataclass
from pathlib import Path

from retriever import Flow, Pipeline
from retriever.flow import flow_io, Rate, Trigger, Latest
from retriever.context import MCPToolFlow
from retriever.context.flow import MCPRequest, MCPResponse

import json

# --- Data Types ---

@flow_io
@dataclass
class TriggerSignal:
    value: str

class SourceFlow(Flow[None, TriggerSignal]):
    """Generates a periodic signal to start the process."""
    def __init__(self):
        super().__init__()
        self.counter = 0

    def run(self, _):
        self.counter += 1
        msg = "ping" if self.counter % 2 != 0 else "pong"
        print(f"\n[Source] Emitting: {msg}")
        return TriggerSignal(value=msg)

# --- Flow Definitions ---

class PlannerFlow(Flow[TriggerSignal, MCPRequest]):
    """Decides which tool to call based on input."""
    def run(self, input: TriggerSignal):
        # deterministically decide based on input
        if input.value == "ping":
            print("[Planner] Ping received -> Reading memory")
            payload = {"tool_name": "read_memory", "args": {"key": "status"}}
            req = MCPRequest(content=json.dumps(payload))
            print(f"[Planner] Returning: {req}")
            return req
        else:
            print("[Planner] Pong received -> Writing memory")
            payload = {"tool_name": "write_memory", "args": {"key": "status", "value": "active"}}
            req = MCPRequest(content=json.dumps(payload))
        
        print(f"[Planner] Returning: {req}")
        return req

class SinkFlow(Flow[MCPResponse, None]):
    """Logs the result from MCP."""
    def run(self, input: MCPResponse):
        if input.error:
            print(f"[Sink] Error: {input.error}")
        else:
            print(f"[Sink] Result: {input.content}")

# --- Pipeline Construction ---

def main():
    # 1. Setup config (points to local server.py)
    config_path = str(Path(__file__).parent / "mcp.json")

    # 2. Build Pipeline
    pipe = Pipeline("mcp_pipeline")
    
    with pipe:
        # Define Flows with Clocks
        # Source runs every 2 seconds
        source = SourceFlow() @ Rate(hz=0.5)
        
        # Others run when triggered by data arrival
        planner = PlannerFlow() @ Trigger("value")
        
        # The MCP Flow connects to the server defined in mcp.json
        # Trigger on content (atomic arrival)
        memory = MCPToolFlow(server_name="memory", config_path=config_path) @ Trigger("content")
        
        sink = SinkFlow() @ Trigger("content", "error")

        # Connect the graph
        # Source -> Planner
        source.then(planner, map={"value": "value"}, sync=Latest())
        
        # Planner -> Memory
        planner.then(memory, map={"content": "content"}, sync=Latest())
        
        # Memory -> Sink
        memory.then(sink, map={"content": "content", "error": "error"}, sync=Latest())

    # 3. Execution
    print("Starting MCP Pipeline...")
    print("Press Ctrl+C to stop (or wait for duration).")
    
    # Run for 7 seconds
    pipe.run(backend="multiprocessing", duration=7.0)
    print("Pipeline finished.")

if __name__ == "__main__":
    main()
