from __future__ import annotations

from typing import Protocol, Dict, Any


class RobotConnection(Protocol):
    """Protocol for owning a robot SDK connection in a single process.

    Implementations should keep non-serializable SDK clients internal and expose
    simple command/status methods. Commands are high-level action dictionaries
    (or can be adapted to a typed Command upstream).
    """

    def connect(self) -> None:
        """Establish connection to the robot."""

    def ensure_connected(self) -> None:
        """Connect or refresh heartbeat if needed; attempt reconnection on failure."""

    def execute(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a high-level command and return a result dict."""

    def status(self) -> Dict[str, Any]:
        """Return current connection/robot status."""

    def close(self) -> None:
        """Release resources and close the connection."""

