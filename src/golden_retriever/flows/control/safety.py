from __future__ import annotations

from typing import Any, Dict

from retriever.core.flow import Flow
from retriever_typing import Status
from retriever.robots.connection_base import RobotConnection


class EstopStatusMonitorFlow(Flow[None, bool]):
    """Monitor flow that checks E-Stop status via a RobotConnection.

    Returns True if E-Stop is engaged (unsafe), False otherwise.
    """

    def __init__(self, manager: RobotConnection):
        super().__init__()
        self.manager = manager

    def run(self, _: None) -> bool:  # type: ignore[override]
        status = self.manager.status()
        estop_state = str(status.get("estop_state", "unknown")).lower()
        return estop_state in ("engaged", "pressed", "true", "emergency")


class EmergencyStopFlow(Flow[Any, Status]):
    """Flow that issues an immediate stop/estop command through a RobotConnection.

    If `use_estop` is True, tries 'estop', else falls back to 'stop'.
    """

    def __init__(self, manager: RobotConnection, use_estop: bool = True):
        super().__init__()
        self.manager = manager
        self.use_estop = use_estop

    def run(self, _: Any) -> Status:  # type: ignore[override]
        if self.use_estop:
            result = self.manager.execute({"type": "estop", "parameters": {}})
        else:
            result = self.manager.execute({"type": "stop", "parameters": {}})
        if result.get("ok"):
            return Status(state="completed", message="ok")
        return Status(state="failed", message=str(result.get("error", "unknown error")))
