# MCP Integration Examples

Three small examples showing different ways to integrate Retriever with a local MCP server.

## Setup

Ensure the `mcp` dependency is installed in the active environment.

## Examples

```bash
pixi run python examples/advanced/mcp_examples/01_utility_client.py
pixi run python examples/advanced/mcp_examples/02_reactive_flow.py
pixi run python examples/advanced/mcp_examples/03_agent_flow.py
```

- `01_utility_client.py`: imperative client calls inside a flow.
- `02_reactive_flow.py`: tool execution as a separate reactive stage.
- `03_agent_flow.py`: an agent-style loop that selects MCP tools dynamically.
