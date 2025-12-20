from __future__ import annotations

from typing import Any, Dict

from retriever.robots.connection_base import RobotConnection


class MockRobotConnectionManager(RobotConnection):
    """Generic mock RobotConnection for tutorials and tests.

    - Owns no SDK; safe to use on any machine.
    - Implements minimal command set: move_to, stop, estop, estop_release.
    - Exposes simple status with connected/busy/estop fields.
    """

    def __init__(self):
        self._connected = False
        self._busy = False
        self._estop = False

    def connect(self) -> None:
        self._connected = True

    def ensure_connected(self) -> None:
        if not self._connected:
            self.connect()

    def execute(self, command: Dict[str, Any]) -> Dict[str, Any]:
        self.ensure_connected()
        if self._busy:
            return {"ok": False, "error": "Busy", "retry_after_sec": 0.05}
        self._busy = True
        try:
            typ = command.get("type")
            if typ == "move_to":
                return {"ok": True, "telemetry": {"moved": True}}
            if typ == "stop":
                return {"ok": True, "telemetry": {"stopped": True}}
            if typ in ("estop", "e_stop"):
                self._estop = True
                return {"ok": True, "telemetry": {"estop": True}}
            if typ in ("estop_release", "estop_disengage"):
                self._estop = False
                return {"ok": True, "telemetry": {"estop_released": True}}
            return {"ok": False, "error": f"Unknown command type: {typ}"}
        finally:
            self._busy = False

    def status(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "connected": self._connected,
            "busy": self._busy,
            "estop_state": "engaged" if self._estop else "disengaged",
        }

    def close(self) -> None:
        self._connected = False
        self._busy = False
        self._estop = False

