"""
MCP Integration Pattern A: Utility Client

This example demonstrates the simplest integration pattern: imperative usage.
The Flow holds an instance of `MCPClient` and calls it directly within its step method.

Run with:
    pixi run python examples/advanced/mcp_examples/01_utility_client.py
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path

from retriever import Flow
from retriever.context import MCPClient
from retriever.flow import flow_io

@flow_io
@dataclass
class String:
    value: str

class PlannerFlow(Flow[String, String]):
    def __init__(self):
        super().__init__()
        self.mcp: MCPClient = None

    async def setup(self):
        # Load from the local mcp.json in this directory
        config_path = str(Path(__file__).parent / "mcp.json")
        self.mcp = await MCPClient.from_config(config_path)
        print("PlannerFlow: Connected to MCP servers.")

    async def step(self, observation: String) -> String:
        obs_text = observation.value
        print(f"\n[Planner] Received observation: {obs_text}")
        
        # 1. Read from memory
        try:
            memory_val = await self.mcp.call_tool("memory", "read_memory", {"key": "last_plan"})
            # Result is CallToolResult
            content = memory_val.content[0].text
            print(f"[Planner] Recalled 'last_plan': {content}")
        except Exception as e:
            print(f"[Planner] Could not read memory: {e}")
            content = "None"

        # 2. Plan (simulated)
        new_plan = f"Plan for {obs_text} based on {content}"
        print(f"[Planner] Generated new plan: {new_plan}")
        
        # 3. Write to memory
        await self.mcp.call_tool("memory", "write_memory", {"key": "last_plan", "value": new_plan})
        print("[Planner] Saved 'last_plan' to memory.")
        
        return String(new_plan)

    async def teardown(self):
        if self.mcp:
            await self.mcp.close()

async def main():
    flow = PlannerFlow()
    await flow.setup()
    
    # Simulate a few steps
    await flow.step(String("box_on_table"))
    await asyncio.sleep(1)
    await flow.step(String("box_on_floor"))
    
    await flow.teardown()

if __name__ == "__main__":
    asyncio.run(main())
