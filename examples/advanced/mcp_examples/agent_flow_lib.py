
import asyncio
from typing import Dict, Any, List
from pathlib import Path
from dataclasses import dataclass

from retriever import Flow
from retriever.flow import flow_io
from retriever.context import MCPClient

@flow_io
@dataclass
class AgentState:
    """Represents the internal state of the agent."""
    goal: str
    memory_context: str
    last_action: str

@flow_io
@dataclass
class AgentAction:
    """Represents an action taken by the agent."""
    action_type: str
    details: str

class MCPAgentFlow(Flow[AgentState, AgentAction]):
    """
    An agent that uses MCP tools to accomplish a goal.
    It reads context from memory, decides on an action, and executes tools.
    """
    def __init__(self, config_path: str):
        super().__init__()
        self.config_path = config_path
        self.mcp: MCPClient = None
        self._tools: List[Dict[str, Any]] = []

    async def setup(self):
        self.mcp = await MCPClient.from_config(self.config_path)
        # Pre-fetch tools to understand capabilities
        self._tools = await self.mcp.get_tool_schemas()
        print(f"[MCPAgent] Connected. Available tools: {[t['function']['name'] for t in self._tools]}")

    async def step(self, state: AgentState) -> AgentAction:
        print(f"\n[MCPAgent] Current Goal: {state.goal}")
        
        # 1. Decision Logic (Simulated Planner)
        # In a real system, this would be an LLM call: llm.chat(messages, tools=self._tools)
        
        if "read_memory" in [t['function']['name'] for t in self._tools] and state.memory_context == "Uninitialized":
            print("[MCPAgent] Decided to: Check Memory")
            # Execute tool
            try:
                result = await self.mcp.call_tool("memory", "read_memory", {"key": "long_term_goal"})
                content = result.content[0].text
                return AgentAction(action_type="UPDATE_CONTEXT", details=content)
            except Exception as e:
                return AgentAction(action_type="ERROR", details=str(e))
                
        elif "write_memory" in [t['function']['name'] for t in self._tools]:
            print("[MCPAgent] Decided to: Save Goal")
            try:
                await self.mcp.call_tool("memory", "write_memory", {"key": "long_term_goal", "value": state.goal})
                return AgentAction(action_type="COMPLETE", details="Goal saved to memory")
            except Exception as e:
                 return AgentAction(action_type="ERROR", details=str(e))
        
        return AgentAction(action_type="IDLE", details="No suitable tool found")

    async def teardown(self):
        if self.mcp:
            await self.mcp.close()
