"""
Dora robot-io node stub for RobotConnection-like command/status ports.

This node is responsible for owning the robot SDK connection in production (e.g., Spot),
receiving typed Command messages, executing via the SDK, and publishing Status/telemetry.

Ports
- Input: `cmd` (Retriever Command as Arrow-compatible dict)
- Outputs: `status` (Retriever Status), `result` (execution result), optional telemetry streams

Implementation Notes
- Convert `cmd` to a high-level dict {"type": str, "parameters": dict} and call a RobotConnection.
- Publish periodic `status` based on RobotConnection.status().
- Handle E-Stop: accept a control message or expose a dedicated port, map to `estop` command.
- Resilience: reconnect on errors, backpressure via internal queue.

This is a placeholder to guide the production node; actual Dora API usage will be wired later.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

try:
    import dora  # type: ignore
    DORA_AVAILABLE = True
except Exception:  # pragma: no cover
    DORA_AVAILABLE = False

from retriever.robots.connection_base import RobotConnection
from retriever.robots.spot.connection import SpotConnectionManager, SpotConnectionConfig


class RobotIONode:
    def __init__(self, manager: RobotConnection):
        self.manager = manager

    def handle_cmd(self, cmd_msg: Dict[str, Any]) -> Dict[str, Any]:
        cmd = {
            "type": cmd_msg.get("action", {}).get("type") or cmd_msg.get("type"),
            "parameters": cmd_msg.get("action", {}).get("parameters") or cmd_msg.get("parameters", {}),
        }
        return self.manager.execute(cmd)

    def get_status(self) -> Dict[str, Any]:
        return self.manager.status()


def main() -> None:
    # Example construction; in production, parse from environment
    mgr = SpotConnectionManager(SpotConnectionConfig(host="192.168.80.3"))
    node = RobotIONode(mgr)
    if not DORA_AVAILABLE:
        print("Dora not available; this is a stub node for reference.")
        print("Status:", node.get_status())
        return
    # Real Dora wiring would go here: subscribe to `cmd`, publish `result` and `status`.


if __name__ == "__main__":
    main()

