"""
MCP Integration Pattern B: Reactive Flow

This example demonstrates a functional/reactive integration pattern.
The MCP server connection is wrapped in a dedicated `MCPToolFlow`.
Interactions happen via signal passing (`MCPRequest` -> `MCPResponse`), allowing
for pure composition of the memory server with other processing flows.

Run with:
    pixi run python examples/advanced/mcp_examples/02_reactive_flow.py
"""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

from retriever import Flow
from retriever.context import MCPToolFlow
from retriever.context.flow import MCPRequest, MCPResponse
from retriever.flow import io

# Reuse types or define new ones if needed
@io
@dataclass
class String:
    value: str

class SimplePlannerFlow(Flow[String, MCPRequest]):
    """
    Receives an observation, asks Memory for context.
    Output: Request to read memory.
    """
    async def setup(self):
        pass

    async def step(self, observation: String) -> MCPRequest:
        print(f"\n[Planner] Received observation: {observation.value}")
        # Ask memory for the plan
        payload = {"tool_name": "read_memory", "args": {"key": "last_plan"}}
        return MCPRequest(content=json.dumps(payload))
    
    async def teardown(self):
        pass

class PlanGeneratorFlow(Flow[MCPResponse, MCPRequest]):
    """
    Receives memory context, generates plan, requests to save it.
    """
    async def setup(self):
        pass

    async def step(self, memory_response: MCPResponse) -> MCPRequest:
        content = memory_response.content if memory_response.content else "None"
        print(f"[Generator] Recalled context: {content}")

        new_plan = f"Plan based on previous: {content}"
        print(f"[Generator] Generated: {new_plan}")

        payload = {"tool_name": "write_memory", "args": {"key": "last_plan", "value": new_plan}}
        return MCPRequest(content=json.dumps(payload))

    async def teardown(self):
        pass

async def main():
    # 1. Setup config path
    config_path = str(Path(__file__).parent / "mcp.json")

    # 2. Create the Memory Flow (Pattern B)
    memory_flow = MCPToolFlow(server_name="memory", config_path=config_path)

    # 3. Create Logic Flows
    planner = SimplePlannerFlow()
    generator = PlanGeneratorFlow()

    # 4. Setup all
    memory_flow.init()
    await planner.setup()
    await generator.setup()

    # 5. Simulate the pipeline:
    # Observation -> Planner -> (req) -> Memory -> (resp) -> Generator -> (req) -> Memory

    obs = String("start_task")

    # Step 1: Planner decides what to read
    read_req = await planner.step(obs)

    # Step 2: Memory executes read (SYNC now)
    read_resp = memory_flow.step(read_req)

    # Step 3: Generator makes new plan and write request
    write_req = await generator.step(read_resp)

    # Step 4: Memory executes write (SYNC now)
    memory_flow.step(write_req)

    print("Cycle 1 complete. Verifying persistence...")

    # Step 5: Verify Read-After-Write
    # Planner asks to read again
    read_req_2 = await planner.step(String("verify_persistence"))
    read_resp_2 = memory_flow.step(read_req_2)

    content_2 = read_resp_2.content
    print(f"Verification Read Result: {content_2}")

    if "Plan based on previous" in content_2:
        print("[SUCCESS] Memory persistence verfied.")
    else:
        print("[FAILURE] Could not retrieve written plan.")

    await generator.teardown()
    await planner.teardown()
    memory_flow.finalize()

if __name__ == "__main__":
    asyncio.run(main())
