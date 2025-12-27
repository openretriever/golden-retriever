# MCP Integration Examples

This directory demonstrates three patterns for integrating Retriever with the **Model Context Protocol (MCP)**.

All examples use a local, Python-based memory server (`server.py`) and a shared configuration (`mcp.json`).

## Setup

First, ensure you have the `mcp` dependency installed (if managed via pixi):

```bash
pixi add mcp
```

## Examples

### 1. Utility Client (`01_utility_client.py`)
This represents the simplest integration. The Flow holds an instance of `MCPClient` and calls it imperatively within `step()`.
*   **Best for**: Quick prototypes, simple wrappers.
*   **Run**: `pixi run python 01_utility_client.py`

### 2. Reactive Flow (`02_reactive_flow.py`)
This wraps the MCP server connection into a dedicated `MCPToolFlow`. Interactions happen via signal passing (`MCPRequest` -> `MCPResponse`).
*   **Best for**: Complex, purely reactive pipelines where tool execution is a distinct phase.
*   **Run**: `pixi run python 02_reactive_flow.py`

### 3. Agentic Flow (`03_agent_flow.py`)
This is the most advanced pattern. The `MCPAgentFlow` discovers available tools from the server and uses a simulated agent (or LLM) to decide which tools to call to achieve a goal.
*   **Best for**: Autonomous agents.
*   **Run**: `pixi run python 03_agent_flow.py`

## Shared Components
*   `server.py`: A local MCP server implementation (using `fastmcp`) that provides a simple key-value memory store.
*   `mcp.json`: Configuration file defining how to connect to `server.py`.
