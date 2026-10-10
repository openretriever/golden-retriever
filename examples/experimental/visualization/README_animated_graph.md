# Animated pipeline graph

Export Retriever IR as an offline Cytoscape/dagre diagram. Motion is
**illustrative configured-clock animation, not execution replay or measured
telemetry**. Periodic clocks use a compressed visual speed scale; trigger
clocks pulse without assigning a nominal frequency. Hybrid clocks retain both
their periodic and trigger configuration. Unknown clocks remain static.

## Run

From the repository root, in an existing environment with Retriever installed:

```bash
python examples/experimental/visualization/visualize_pipeline_animated.py
```

Expected output: `out/golden_retriever_animated_graph.html`. No pipeline
execution, hardware, model downloads, or network access is needed. The exporter
inlines the Cytoscape/dagre assets bundled with `retriever.ir`; missing assets
raise `RuntimeError` before writing a file. The template itself has CDN
references for development; generated exports do not require a connection.

## Export API

```python
from animated_graph import ir_to_graph, save_animated_html, write_graph_html

ir = pipe.validate()
save_animated_html(ir, "out/graph.html")
graph = ir_to_graph(ir)
write_graph_html(graph, "out/graph.html")
```

Both export functions return the output `Path`. Importing the module does not
execute a pipeline.

### Identity and annotations

- `type` and primary `label` preserve the exact `IRNode.type` string, including
  spelling and capitalization. No humanized class names are generated.
- `id` is the exact instance ID, rendered beneath the class name when different.
  Repeated types remain distinguishable by their instance IDs.
- `labels={node_id: "annotation"}` adds secondary `annotation` text;
  `sublabels` adds `sub` text. Neither replaces the class or instance identity.
- Missing or non-string types fall back to the instance ID. Unknown kinds use
  ordinary `flow` styling. Unknown type strings are displayed unchanged.

```python
save_animated_html(
    ir,
    "out/graph.html",
    labels={node_id: "Optional annotation"},
    kinds={node_id: "major"},
    layout={node_id: {"x": 120, "y": 80}},
)
```

Obtain IDs from `pipe.get_node_id(handle)`. `kinds` accepts `flow`, `env`
(neutral dashed border), or `major` (emphasis).

### Graph dictionary

```json
{
  "nodes": [
    {
      "id": "processor",
      "type": "ProcessorFlow",
      "label": "ProcessorFlow",
      "annotation": "Optional annotation",
      "kind": "flow",
      "rate": "trigger(sample)",
      "clock": {"kind": "trigger", "fields": ["sample"]}
    }
  ],
  "edges": [
    {"from": "processor", "to": "processor", "label": "result", "sync": "Latest"}
  ]
}
```

`clock.kind` is `rate`, `trigger`, `hybrid`, or `unknown`; periodic
configuration carries `hz`, and trigger configuration may carry `fields`.
The legacy `rate` field accepts positive Hz, a trigger string, or null.
Structured `clock` takes precedence. Invalid/nonpositive periodic rates do not
animate. Node IDs must be unique nonempty strings and every edge endpoint must
exist; invalid topology raises `ValueError`. Text is escaped for both JSON
embedding and SVG labels, and nonfinite JSON values are rejected.

## Layout and interaction

**Loops** retains the existing left-to-right spine and nested feedback router.
Feedback channels are ordered by containment; overlapping channels get separate
lanes. Edge labels sit on their longest horizontal run. Node widths include
measurement safety padding; longer text wraps at word or CamelCase boundaries
where possible, without changing its stored spelling.

**Layers** uses dagre's network-simplex layout. **Auto** compares horizontal and
vertical layouts against the available viewport. Vertical mode uses Cytoscape's
native curve routing. Horizontal mode reuses the feedback router. Dense graphs
can still have crossings; pan, zoom, change direction, or supply positions when
a particular topology needs editorial placement.

`layout` maps IDs to finite numeric `x`/`y` centers. A complete layout skips
automatic placement. A partial layout overrides only supplied nodes after
automatic placement; it can therefore introduce overlaps. Routing follows the
resulting positions. Fit includes nodes, edges, and labels, and accounts for the
toolbar and status area even when they wrap.

Nodes can be dragged. Controls provide fit, pause/play, a visual speed multiplier,
graph/layout JSON copying, and PNG export. Port labels are hidden in the initial
overview to avoid fan-out clutter. Hover or tap an individual edge for its port,
sync policy, and endpoint IDs; the Port labels checkbox shows all labels.
When clipboard access is unavailable,
JSON downloads instead. Copied JSON can be passed to `write_graph_html`;
its node coordinates can also populate the `layout` argument. A supplied layout
is preserved on resize. Reduced-motion preferences start the animation paused.
