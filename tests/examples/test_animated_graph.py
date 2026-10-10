from __future__ import annotations

import copy
import importlib.util
import json
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote

import pytest


ROOT = Path(__file__).resolve().parents[2]
VISUALIZATION = ROOT / "examples" / "experimental" / "visualization"


@pytest.fixture
def animated_graph():
    spec = importlib.util.spec_from_file_location(
        "animated_graph", VISUALIZATION / "animated_graph.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _node(node_id, flow_type="SensorFlow", clock=None):
    return SimpleNamespace(
        id=node_id,
        type=flow_type,
        module="example_flows",
        config={"clock": clock} if clock is not None else {},
    )


def _edge(source, target, port="sample", adapter=None):
    return SimpleNamespace(
        source=SimpleNamespace(node=source, port=port),
        destination=SimpleNamespace(node=target, port="input"),
        adapter=adapter,
    )


def _ir(*nodes, edges=()):
    return SimpleNamespace(nodes=list(nodes), edges=list(edges))


class _Document(HTMLParser):
    def __init__(self, html):
        super().__init__(convert_charrefs=False)
        self.scripts = []
        self.tags = []
        self.current_script = None
        self.feed(html)
        self.close()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.tags.append((tag, attrs))
        if tag == "script":
            self.current_script = {"attrs": attrs, "text": ""}
            self.scripts.append(self.current_script)

    def handle_endtag(self, tag):
        if tag == "script":
            self.current_script = None

    def handle_data(self, data):
        if self.current_script is not None:
            self.current_script["text"] += data

    def graph(self):
        blocks = [
            script for script in self.scripts
            if script["attrs"].get("id") == "retriever-graph"
        ]
        assert len(blocks) == 1
        assert blocks[0]["attrs"].get("type") == "application/json"
        return json.loads(blocks[0]["text"])


@pytest.mark.parametrize("flow_type", ["SensorFlow", "RGBDFlow", "PIDControllerFlow"])
def test_flow_class_is_primary_and_instance_id_is_secondary(animated_graph, flow_type):
    ir = _ir(_node("sensor_left", flow_type), _node("sensor_right", flow_type))
    graph = animated_graph.ir_to_graph(ir)

    assert [node["label"] for node in graph["nodes"]] == [flow_type, flow_type]
    assert [node["id"] for node in graph["nodes"]] == ["sensor_left", "sensor_right"]


@pytest.mark.parametrize("flow_type", ["UnregisteredFlow", "example_flows.RGBDFlow"])
def test_unknown_and_qualified_types_do_not_require_imports(animated_graph, flow_type):
    graph = animated_graph.ir_to_graph(_ir(_node("sensor_0", flow_type)))
    assert graph["nodes"][0]["label"] == flow_type
    assert graph["nodes"][0]["id"] == "sensor_0"


def test_instance_id_matching_class_name_is_valid(animated_graph):
    graph = animated_graph.ir_to_graph(_ir(_node("SensorFlow")))
    assert graph["nodes"][0]["label"] == "SensorFlow"
    assert graph["nodes"][0]["id"] == "SensorFlow"


def test_annotations_cannot_replace_flow_identity_or_mutate_ir(animated_graph):
    ir = _ir(_node("sensor_0"), _node("sensor_1"))
    before = copy.deepcopy(ir)
    options = {
        "labels": {"sensor_0": "User-defined label"},
        "sublabels": {"sensor_0": "User-defined subtitle"},
        "kinds": {"sensor_0": "major"},
        "layout": {"sensor_0": {"x": 12.5, "y": -40.0}},
    }
    options_before = copy.deepcopy(options)
    graph = animated_graph.ir_to_graph(ir, **options)
    first, second = graph["nodes"]
    assert first["label"] == "SensorFlow"
    assert first["annotation"] == "User-defined label"
    assert first["sub"] == "User-defined subtitle"
    assert first["kind"] == "major"
    assert (first["x"], first["y"]) == (12.5, -40.0)
    assert second["label"] == "SensorFlow"
    assert "x" not in second and "y" not in second
    assert ir == before
    assert options == options_before


@pytest.mark.parametrize(
    ("clock", "expected"),
    [
        (None, None),
        ({"Rate": {"hz": 200}}, 200.0),
        ({"Rate": {"hz": 0.5}}, 0.5),
        ({"Trigger": {"fields": ["image", "state"]}}, "trigger(image, state)"),
        ({"Trigger": {"ports": ["sample"]}}, "trigger(sample)"),
        ({"Trigger": {}}, "trigger"),
        ({"Hybrid": {"hz": 2.0, "trigger_fields": ["sample"]}}, 2.0),
        ({"CustomClock": {}}, None),
    ],
)
def test_clock_metadata_preserves_rates_and_trigger_labels(animated_graph, clock, expected):
    graph = animated_graph.ir_to_graph(_ir(_node("sensor_0", clock=clock)))
    assert graph["nodes"][0]["rate"] == expected


def test_hybrid_clock_preserves_trigger_fields(animated_graph):
    clock = {"Hybrid": {"hz": 2.0, "trigger_fields": ["sample", "state"]}}
    graph = animated_graph.ir_to_graph(_ir(_node("sensor_0", clock=clock)))
    assert graph["nodes"][0]["clock"] == {
        "kind": "hybrid", "hz": 2.0, "fields": ["sample", "state"]
    }


def test_cycles_parallel_edges_and_isolated_nodes_survive_projection(animated_graph):
    ir = _ir(
        _node("sensor_0"), _node("control_0", "ControlFlow"), _node("isolated_0"),
        edges=[
            _edge("sensor_0", "control_0", adapter={"Latest": {}}),
            _edge("sensor_0", "control_0", "status", {"Zip": {}}),
            _edge("control_0", "sensor_0", "feedback"),
            _edge("control_0", "control_0", ""),
        ],
    )
    graph = animated_graph.ir_to_graph(ir)
    assert [node["id"] for node in graph["nodes"]] == [
        "sensor_0", "control_0", "isolated_0"
    ]
    assert graph["edges"] == [
        {"from": "sensor_0", "to": "control_0", "label": "sample", "sync": "Latest"},
        {"from": "sensor_0", "to": "control_0", "label": "status", "sync": "Zip"},
        {"from": "control_0", "to": "sensor_0", "label": "feedback"},
        {"from": "control_0", "to": "control_0"},
    ]
    assert animated_graph.ir_to_graph(_ir()) == {"nodes": [], "edges": []}


def test_export_embeds_hostile_labels_as_data_only(animated_graph, tmp_path):
    hostile = '</ScRiPt><script id="unexpected">alert(1)</script><img src=x onerror=alert(2)>'
    graph = {
        "nodes": [{
            "id": hostile, "type": hostile, "label": hostile,
            "sub": "<!--<script>", "rate": None, "kind": "flow",
        }],
        "edges": [{"from": hostile, "to": hostile, "label": '<>&"\'\u2028\u2029'}],
    }
    before = copy.deepcopy(graph)
    output = tmp_path / "nested" / "graph.html"
    assert animated_graph.write_graph_html(graph, output) == output
    document = _Document(output.read_text(encoding="utf-8"))
    exported = document.graph()
    assert len(exported["nodes"]) == len(exported["edges"]) == 1
    for field, value in graph["nodes"][0].items():
        assert exported["nodes"][0][field] == value
    for field, value in graph["edges"][0].items():
        assert exported["edges"][0][field] == value
    assert graph == before
    assert not any(attrs.get("id") == "unexpected" for _, attrs in document.tags)
    assert not any("onerror" in attrs for _, attrs in document.tags)
    payload = next(
        script["text"] for script in document.scripts
        if script["attrs"].get("id") == "retriever-graph"
    )
    assert not re.search(r"</script|<!--|<script", payload, re.IGNORECASE)


def test_export_is_offline_when_runtime_assets_are_bundled(animated_graph, tmp_path):
    from retriever.ir import viz

    assets = Path(viz.__file__).parent / "assets"
    names = ("cytoscape.min.js", "dagre.min.js", "cytoscape-dagre.min.js")
    assert all((assets / name).is_file() for name in names)
    output = animated_graph.write_graph_html({"nodes": [], "edges": []}, tmp_path / "graph.html")
    document = _Document(output.read_text(encoding="utf-8"))
    assert not any(script["attrs"].get("src") for script in document.scripts)
    assert not any(
        tag == "link" and attrs.get("rel") == "stylesheet" and attrs.get("href")
        for tag, attrs in document.tags
    )
    assert len([script for script in document.scripts if script["text"].strip()]) >= 5
    assert document.graph() == {"nodes": [], "edges": []}


def test_save_forwards_options_and_preserves_graph(animated_graph, tmp_path):
    ir = _ir(_node("sensor_0", clock={"Rate": {"hz": 30}}))
    options = {"layout": {"sensor_0": {"x": 100, "y": 50}}}
    output = animated_graph.save_animated_html(ir, tmp_path / "graph.html", **options)
    assert _Document(output.read_text(encoding="utf-8")).graph() == animated_graph.ir_to_graph(
        ir, **options
    )


def test_demo_uses_real_flow_names_without_running_pipeline(animated_graph, monkeypatch):
    monkeypatch.syspath_prepend(str(VISUALIZATION))
    spec = importlib.util.spec_from_file_location(
        "visualize_pipeline_animated", VISUALIZATION / "visualize_pipeline_animated.py"
    )
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    ir, ids = demo.build_demo()
    graph = animated_graph.ir_to_graph(ir)
    types = {node.id: node.type for node in ir.nodes}
    assert len(graph["nodes"]) == len(types) == len(ids)
    assert all(node["label"] == types[node["id"]] for node in graph["nodes"])
    assert {node["id"] for node in graph["nodes"]} == set(ids.values())
    assert all(edge["from"] in types and edge["to"] in types for edge in graph["edges"])


def test_clock_animation_is_documented_as_illustrative_not_event_replay():
    readme = (VISUALIZATION / "README_animated_graph.md").read_text(encoding="utf-8")
    template = (VISUALIZATION / "animated_graph_template.html").read_text(encoding="utf-8")
    for text in (readme, template):
        assert "illustrative" in text.lower()
        assert re.search(r"not\b[^.]{0,100}\breplay", text, re.IGNORECASE)


@pytest.mark.parametrize("flow_type", [None, ""])
def test_missing_type_falls_back_to_instance_id_without_inventing_class(animated_graph, flow_type):
    graph = animated_graph.ir_to_graph(_ir(_node("sensor_instance", flow_type)))
    assert graph["nodes"][0]["label"] == "sensor_instance"


@pytest.mark.parametrize(
    "graph",
    [
        {"nodes": [{"id": "sensor"}, {"id": "sensor"}], "edges": []},
        {"nodes": [{"id": ""}], "edges": []},
        {"nodes": [{"id": "sensor"}], "edges": [{"from": "sensor", "to": "missing"}]},
        {"nodes": [{"id": "sensor"}], "edges": [{"from": "missing", "to": "sensor"}]},
        {"nodes": [{"id": "sensor", "x": float("nan"), "y": 1}], "edges": []},
        {"nodes": [{"id": "sensor", "x": 1}], "edges": []},
    ],
)
def test_invalid_graph_is_rejected_before_writing(animated_graph, tmp_path, graph):
    output = tmp_path / "invalid.html"
    with pytest.raises(ValueError):
        animated_graph.write_graph_html(graph, output)
    assert not output.exists()


@pytest.mark.parametrize("rate", [0, -1, float("nan"), float("inf"), True])
def test_invalid_periodic_rates_do_not_become_motion_frequencies(animated_graph, rate):
    graph = animated_graph.ir_to_graph(_ir(_node("sensor_0", clock={"Rate": {"hz": rate}})))
    assert graph["nodes"][0]["rate"] is None


def test_missing_asset_fails_without_writing_partial_export(animated_graph, monkeypatch, tmp_path):
    from retriever.ir import viz

    monkeypatch.setattr(viz, "__file__", str(tmp_path / "viz.py"))
    output = tmp_path / "graph.html"
    with pytest.raises(RuntimeError, match="asset"):
        animated_graph.write_graph_html({"nodes": [], "edges": []}, output)
    assert not output.exists()


@pytest.mark.parametrize("glyph_width", [7.01, 7.1, 7.49])
def test_fractional_text_width_preserves_single_line_class_identity(glyph_width):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the renderer unit test")
    document = _Document(
        (VISUALIZATION / "animated_graph_template.html").read_text(encoding="utf-8")
    )
    scripts = [
        script["text"] for script in document.scripts
        if script["text"].strip() and script["attrs"].get("type") != "application/json"
    ]
    canonical = "ObservationProcessorFlow"
    graph = {
        "nodes": [{"id": "sensor_0", "type": canonical, "rate": 30}],
        "edges": [],
    }
    # Stop at graph mounting; rendering helpers run with deterministic font metrics.
    runner = r"""
const vm = require("node:vm");
const input = JSON.parse(require("node:fs").readFileSync(0, "utf8"));
const stop = new Error("Graph mounting is outside this unit test");
const cytoscape = () => { throw stop; };
cytoscape.use = () => {};
const context = vm.createContext({
  cytoscape, cytoscapeDagre: {},
  document: {
    getElementById: () => ({textContent: JSON.stringify(input.graph)}),
    createElement: () => ({getContext: () => ({
      measureText: text => ({width: String(text).length * input.glyphWidth})
    })})
  }
});
let stopped = false;
try {
  for (const script of input.scripts) vm.runInContext(script, context, {timeout: 1000});
} catch (error) {
  if (error !== stop) throw error;
  stopped = true;
}
if (!stopped) throw new Error("Expected graph mounting boundary");
const result = vm.runInContext(
  "JSON.stringify({box:boxOf(GRAPH.nodes[0]), svg:svgOf(GRAPH.nodes[0])})", context
);
process.stdout.write(result);
"""
    result = subprocess.run(
        [node, "-e", runner],
        input=json.dumps({"scripts": scripts, "graph": graph, "glyphWidth": glyph_width}),
        text=True, capture_output=True, check=True, timeout=10,
    )
    rendered = json.loads(result.stdout)
    assert rendered["box"]["titleLines"] == [canonical]
    assert "sensor_0" in rendered["box"]["detailLines"]
    assert len(canonical) * glyph_width <= rendered["box"]["w"] - 28
    svg = ET.fromstring(unquote(rendered["svg"]["uri"].split(",", 1)[1]))
    text = [element.text for element in svg.findall("{http://www.w3.org/2000/svg}text")]
    assert text[:2] == [canonical, "sensor_0"]
