"""Export Retriever IR as an offline, animated configured-clock diagram.

Motion illustrates clock configuration; it is not execution telemetry or replay.
The adjacent template uses the Cytoscape/dagre assets bundled with retriever.ir.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Union

if TYPE_CHECKING:
    from retriever.ir import IR
    from retriever.ir.core import IRNode

TEMPLATE = Path(__file__).parent / "animated_graph_template.html"
_VENDORED = {
    "https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.26.0/cytoscape.min.js":
        "cytoscape.min.js",
    "https://cdnjs.cloudflare.com/ajax/libs/dagre/0.8.5/dagre.min.js":
        "dagre.min.js",
    "https://cdn.jsdelivr.net/npm/cytoscape-dagre@2.5.0/cytoscape-dagre.min.js":
        "cytoscape-dagre.min.js",
}
_ANCHOR_OPEN = '<script id="retriever-graph" type="application/json">'
_ANCHOR_CLOSE = "</script>"
Rate = Union[float, str, None]


def _inline_vendored(html: str) -> str:
    """Inline every dependency or fail before writing an incomplete export."""
    try:
        from retriever.ir import viz
        assets = Path(viz.__file__).parent / "assets"
    except (ImportError, TypeError) as exc:
        raise RuntimeError(
            "Offline export requires retriever.ir and its bundled visualization assets."
        ) from exc
    for url, name in _VENDORED.items():
        path = assets / name
        if not path.is_file():
            raise RuntimeError(f"Offline export requires the bundled asset {name}.")
        code = path.read_text(encoding="utf-8")
        html = html.replace(
            f'<script src="{url}"></script>', "<script>" + code + "</script>"
        )
    return html


def _positive_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _clock_of(node: "IRNode") -> Dict[str, Any]:
    config = node.config if isinstance(node.config, Mapping) else {}
    clock = config.get("clock")
    if not isinstance(clock, Mapping):
        return {"kind": "unknown"}
    for tag in ("Hybrid", "Trigger", "Rate"):
        if tag not in clock:
            continue
        params = clock[tag]
        if not isinstance(params, Mapping):
            return {"kind": "unknown"}
        result: Dict[str, Any] = {"kind": tag.lower()}
        hz = _positive_number(params.get("hz"))
        if tag != "Trigger" and hz is not None:
            result["hz"] = hz
        fields = params.get("trigger_fields") if tag == "Hybrid" else None
        fields = fields or params.get("fields") or params.get("ports") or []
        if isinstance(fields, str):
            fields = [fields]
        if isinstance(fields, (list, tuple)):
            result["fields"] = [str(field) for field in fields]
        return result
    return {"kind": "unknown"}


def _rate_of(node: "IRNode") -> Rate:
    """Legacy presentation field; trigger clocks never receive nominal Hz."""
    clock = _clock_of(node)
    if clock["kind"] in ("rate", "hybrid"):
        return clock.get("hz")
    if clock["kind"] == "trigger":
        fields = clock.get("fields", [])
        return f"trigger({', '.join(fields)})" if fields else "trigger"
    return None


def _sync_of(edge: Any) -> Optional[str]:
    adapter = edge.adapter
    return str(next(iter(adapter))) if isinstance(adapter, Mapping) and adapter else None


def ir_to_graph(
    ir: "IR",
    *,
    labels: Optional[Mapping[str, str]] = None,
    sublabels: Optional[Mapping[str, str]] = None,
    kinds: Optional[Mapping[str, str]] = None,
    layout: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> Dict[str, Any]:
    """Project an IR without renaming implementation types.

    Exact IR type strings are the primary label. Instance IDs remain separate.
    labels/sublabels are secondary annotations only. Missing types fall back to
    the instance ID, without guessing a class name. Unknown kinds use flow
    styling. Partial layouts pin supplied positions after automatic placement.
    """
    labels, sublabels = labels or {}, sublabels or {}
    kinds, layout = kinds or {}, layout or {}
    nodes: List[Dict[str, Any]] = []
    for n in ir.nodes:
        canonical = n.type if isinstance(n.type, str) and n.type else n.id
        node: Dict[str, Any] = {
            "id": n.id,
            "type": canonical,
            "label": canonical,
            "rate": _rate_of(n),
            "clock": _clock_of(n),
            "kind": kinds.get(n.id, "flow"),
        }
        if n.id in labels:
            node["annotation"] = labels[n.id]
        if n.id in sublabels:
            node["sub"] = sublabels[n.id]
        if n.id in layout:
            node.update(x=layout[n.id]["x"], y=layout[n.id]["y"])
        nodes.append(node)
    edges: List[Dict[str, Any]] = []
    for e in ir.edges:
        edge: Dict[str, Any] = {"from": e.source.node, "to": e.destination.node}
        if e.source.port:
            edge["label"] = e.source.port
        sync = _sync_of(e)
        if sync:
            edge["sync"] = sync
        edges.append(edge)
    return _normalize_graph({"nodes": nodes, "edges": edges})


def _normalize_graph(graph: Mapping[str, Any]) -> Dict[str, Any]:
    """Copy and validate externally supplied graph data before serialization."""
    nodes, edges, ids = [], [], set()
    for source in graph.get("nodes", []):
        node = dict(source)
        identity = node.get("id")
        if not isinstance(identity, str) or not identity or identity in ids:
            raise ValueError("Every node needs a unique, nonempty string id.")
        ids.add(identity)
        canonical = node.get("type")
        if not isinstance(canonical, str) or not canonical:
            canonical = identity
        label = node.get("label")
        if isinstance(label, str) and label != canonical:
            node.setdefault("annotation", label)
        node.update(type=canonical, label=canonical)
        if node.get("kind") not in ("flow", "env", "major"):
            node["kind"] = "flow"
        rate = node.get("rate")
        if not isinstance(rate, str):
            node["rate"] = _positive_number(rate)
        for key in ("annotation", "sub"):
            if key in node:
                node[key] = str(node[key])
        if "x" in node or "y" in node:
            for key in ("x", "y"):
                value = node.get(key)
                if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                    raise ValueError("Layout positions require finite numeric x and y.")
        nodes.append(node)
    for source in graph.get("edges", []):
        edge = dict(source)
        if edge.get("from") not in ids or edge.get("to") not in ids:
            raise ValueError("Every edge endpoint must reference an existing node id.")
        for key in ("label", "sync"):
            if key in edge:
                edge[key] = str(edge[key])
        edges.append(edge)
    return {**graph, "nodes": nodes, "edges": edges}


def save_animated_html(
    ir: "IR",
    filename: Union[str, Path] = "pipeline_animated.html",
    **kwargs: Any,
) -> Path:
    """Write a standalone offline diagram; kwargs are forwarded to ir_to_graph."""
    return write_graph_html(ir_to_graph(ir, **kwargs), filename)


def write_graph_html(
    graph: Mapping[str, Any],
    filename: Union[str, Path] = "pipeline_animated.html",
) -> Path:
    """Export a graph dictionary, keeping canonical type identity visible."""
    graph = _normalize_graph(graph)
    html = _inline_vendored(TEMPLATE.read_text(encoding="utf-8"))
    start = html.index(_ANCHOR_OPEN) + len(_ANCHOR_OPEN)
    end = html.index(_ANCHOR_CLOSE, start)
    # Escape HTML-significant characters even though the payload is inert JSON.
    payload = json.dumps(graph, indent=2, allow_nan=False)
    payload = payload.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    out = Path(filename)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html[:start] + "\n" + payload + "\n" + html[end:], encoding="utf-8")
    return out
