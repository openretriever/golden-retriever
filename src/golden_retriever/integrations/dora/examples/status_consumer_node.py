"""
Stub Dora consumer node that would subscribe to robot-io `status` and print.
"""
from __future__ import annotations

try:
    import dora  # type: ignore
    DORA_AVAILABLE = True
except Exception:  # pragma: no cover
    DORA_AVAILABLE = False


def main() -> None:
    if not DORA_AVAILABLE:
        print("Dora not available; status consumer stub.")
        return
    # d = dora.Node()
    # for event in d.inputs():
    #     if event.id == "status":
    #         print(event.payload)


if __name__ == "__main__":
    main()

