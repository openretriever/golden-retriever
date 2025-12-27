
from mcp.server.fastmcp import FastMCP
import json

# Create an MCP server
mcp = FastMCP("AdvancedMemory")

# Simple in-memory storage
_MEMORY = {}

@mcp.tool()
def read_memory(key: str) -> str:
    """Read a value from memory by key."""
    return _MEMORY.get(key, "Not found")

@mcp.tool()
def write_memory(key: str, value: str) -> str:
    """Write a value to memory."""
    _MEMORY[key] = value
    return f"Stored {key}={value}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
