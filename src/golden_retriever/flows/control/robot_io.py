from __future__ import annotations

from typing import Dict, Any

from retriever.core.flow import Flow
from golden_retriever.types import Command, Status
from retriever.robots.connection_base import RobotConnection


class RobotIOFlow(Flow[Command, Status]):
    """Flow that bridges typed Command to a RobotConnection.

    Manager must own the SDK handle in the current process. This Flow only marshals
    the typed Command to a dict and back to a Status.
    """

    def __init__(self, manager: RobotConnection):
        super().__init__()
        self.manager = manager

    def run(self, cmd: Command) -> Status:  # type: ignore[override]
        payload: Dict[str, Any] = {"type": cmd.action.type, "parameters": cmd.action.parameters}
        result = self.manager.execute(payload)
        if result.get("ok"):
            return Status(state="completed", message="ok")
        return Status(state="failed", message=str(result.get("error", "unknown error")))
