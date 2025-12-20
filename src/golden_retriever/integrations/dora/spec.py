from __future__ import annotations

"""
Internal Dora graph specification (compiler target) for coordination.

Frontend users never author Dora graphs; the backend executor compiles Flow graphs
to this spec, then materializes a Dora dataflow. This module defines the minimal
node/connection schema used for coordination around a single robot manager.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class NodeSpec:
    id: str
    path: str  # e.g., "python"
    args: Dict[str, str] = field(default_factory=dict)
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)


@dataclass
class GraphSpec:
    nodes: List[NodeSpec]
    connections: List[Tuple[str, str]]  # (from, to), e.g., ("planner/cmd", "arbiter/cmd_a")


def minimal_robot_io_graph(app_id: str) -> GraphSpec:
    """Build a minimal internal graph spec for coordinator → robot-io.

    Ports (convention):
    - Command streams: `cmd`, `cmd_a`, `cmd_b`
    - Status stream: `status`
    - Events stream (optional): `events`
    """
    planner = NodeSpec(
        id=f"{app_id}-planner",
        path="python",
        args={"script": "<compiled_planner_node>.py"},
        outputs=["cmd"],
    )
    arbiter = NodeSpec(
        id=f"{app_id}-arbiter",
        path="python",
        args={"script": "<compiled_arbiter_node>.py"},
        inputs=["cmd"],
        outputs=["cmd"],
    )
    robot_io = NodeSpec(
        id=f"{app_id}-robot-io",
        path="python",
        args={"script": "retriever/integrations/dora/robot_io_node.py"},
        inputs=["cmd"],
        outputs=["status"],
    )
    return GraphSpec(
        nodes=[planner, arbiter, robot_io],
        connections=[
            (f"{planner.id}/cmd", f"{arbiter.id}/cmd"),
            (f"{arbiter.id}/cmd", f"{robot_io.id}/cmd"),
        ],
    )

