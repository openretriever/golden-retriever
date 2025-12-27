"""
MCP Integration Pattern C: Agentic Flow

This example demonstrates the most advanced "Agent" pattern.
The `MCPAgentFlow` automatically discovers available tools from the connected
MCP server and uses a simulated agent (or LLM) to decide which tools to call
to achieve a high-level goal.

Run with:
    pixi run python examples/advanced/mcp_examples/03_agent_flow.py
"""

import asyncio
from pathlib import Path

# Updated import for local module
try:
    from .agent_flow_lib import MCPAgentFlow, AgentState
except ImportError:
    from agent_flow_lib import MCPAgentFlow, AgentState

async def main():
    # Configuration
    base_dir = Path(__file__).parent
    config_path = str(base_dir / "mcp.json")
    
    # Initialize Agent
    agent = MCPAgentFlow(config_path=config_path)
    await agent.setup()
    
    # Simulation Loop
    current_state = AgentState(
        goal="Build a moon base",
        memory_context="Uninitialized",
        last_action="None"
    )
    max_steps = 3
    for i in range(max_steps):
        print(f"--- Step {i+1} ---")
        action = await agent.step(current_state)
        print(f"[Main] Agent Action: {action.action_type} -> {action.details}")

        if action.action_type == "UPDATE_CONTEXT":
            current_state.memory_context = action.details
        elif action.action_type == "COMPLETE":
            print("[Main] Mission Accomplished! Verifying memory...")
            # Verification step: Direct raw read using the agent's connection
            # In a real app we might ask the agent to 'verify', but here we peek.
            res = await agent.mcp.call_tool("memory", "read_memory", {"key": "long_term_goal"})
            saved_val = res.content[0].text
            print(f"[Main] Verification: Memory contains '{saved_val}'")
            if saved_val == current_state.goal:
                 print("[SUCCESS] Goal correctly persisted.")
            else:
                 print(f"[FAILURE] Goal mismatch. Expected {current_state.goal}")
            break

        await asyncio.sleep(0.5)

    await agent.teardown()

if __name__ == "__main__":
    asyncio.run(main())
