"""
Stub Dora planner/producer node that would publish Command messages to robot-io.
This is illustrative; real implementation should use dora Python API to send on a port.
"""
from __future__ import annotations

import time

try:
    import dora  # type: ignore
    DORA_AVAILABLE = True
except Exception:  # pragma: no cover
    DORA_AVAILABLE = False


def main() -> None:
    if not DORA_AVAILABLE:
        print("Dora not available; stub planner prints intended messages.")
        print({"action": {"type": "move_to", "parameters": {"x": 0.5, "y": 0.0, "yaw": 0.0}}})
        return
    # Real Dora send loop would go here
    # d = dora.Node()
    # d.send_output("cmd", payload)
    time.sleep(0.1)


if __name__ == "__main__":
    main()

