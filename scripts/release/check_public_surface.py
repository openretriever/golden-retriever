#!/usr/bin/env python3
"""Check GoldenRetriever's current public example surface.

The default mode combines static source/docs checks with short runtime smokes
for the promoted public commands: Hub pack proof, mock-safe robosuite and
RoboCasa, and HTML pipeline visualization. Use ``--static-only`` for a fast
source-tree check.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

REQUIRED_PATHS = (
    "README.md",
    "examples/advanced/robosuite_lift/app.py",
    "examples/advanced/robosuite_lift/README.md",
    "examples/advanced/robocasa_replay/app.py",
    "examples/advanced/robocasa_replay/README.md",
    "examples/experimental/visualization/visualize_pipeline.py",
    "examples/experimental/visualization/README.md",
    "examples/advanced/core_composition/golden_hub_pack_smoke.py",
    "docs-site/astro.config.mjs",
    "docs-site/package.json",
    "docs-site/public/llms.txt",
    "docs-site/public/robots.txt",
    "docs-site/public/assets/logo.svg",
    "docs-site/src/content/docs/index.mdx",
    "docs-site/src/content/docs/examples/index.mdx",
    "docs-site/src/content/docs/examples/golden-hub-proof.mdx",
    "docs-site/src/content/docs/examples/simulation-visualization.mdx",
    "docs-site/src/content/docs/hub/index.mdx",
    "docs-site/src/content/docs/hub/export-catalog.mdx",
    "docs-site/src/content/docs/robot-payloads/index.mdx",
    "docs-site/src/content/docs/robot-payloads/type-catalog.mdx",
)

REMOVED_PATHS: tuple[str, ...] = (
    "docs",
    "mkdocs.yml",
    "examples/experimental/behavior_1k",
)

REQUIRED_TASKS = (
    "demo-golden-hub-pack",
    "demo-robosuite-mock",
    "demo-robocasa-mock",
    "demo-pipeline-html-viz",
    "demo-robotics-data-eventstream",
    "demo-robotics-data-join",
    "demo-robotics-lerobot-bridge",
    "public-surface-check",
)

DOC_MARKERS = {
    "README.md": (
        "GoldenRetriever is the applied reference layer",
        "Start Here",
        "Surface Boundary",
        "pixi run public-surface-check",
    ),
    "docs-site/astro.config.mjs": (
        "starlightThemeNova",
        "GoldenRetriever",
        "Reuse Robot Payloads",
        "Choose a Payload",
    ),
    "docs-site/public/llms.txt": (
        "GoldenRetriever",
        "demo-golden-hub-pack",
        "demo-robosuite-mock",
        "demo-robocasa-mock",
        "demo-pipeline-html-viz",
        "https://golden.retriever.build/examples/simulation-visualization/",
        "Do not treat source examples as Retriever Hub packs unless the Hub manifest exports them.",
    ),
    "docs-site/public/robots.txt": (
        "Sitemap: https://golden.retriever.build/sitemap-index.xml",
        "LLM map: https://golden.retriever.build/llms.txt",
    ),
    "docs-site/src/content/docs/index.mdx": (
        "GoldenRetriever examples start where the core quickstart ends",
        "Run robot-facing examples",
        "GoldenRetriever is not a second runtime",
        "What you should see",
        "Where to go next",
        "How the pieces fit",
        "For AI agents",
    ),
    "docs-site/src/content/docs/examples/index.mdx": (
        "Find an example",
        "Try it first",
        "demo-golden-hub-pack",
        "demo-pipeline-html-viz",
        "Suggested order",
        "Related runtime docs",
        "How mature each example is",
    ),
    "docs-site/src/content/docs/examples/golden-hub-proof.mdx": (
        "Hub Pack Quickstart",
        'hub.use("openretriever/golden-retriever:WorldState")',
        "Arrow round-trip: Action OK",
    ),
    "docs-site/src/content/docs/examples/simulation-visualization.mdx": (
        "Render the graph to HTML",
        "Run these in order",
        "Rerun",
        "Mock robosuite",
        "Connect Retriever to RoboCasa",
        "demo-pipeline-html-viz",
    ),
    "docs-site/src/content/docs/hub/index.mdx": (
        "GoldenRetriever Hub Packs",
        "What's in the pack",
        "Related runtime docs",
        "Pack Rules",
    ),
    "docs-site/src/content/docs/hub/export-catalog.mdx": (
        "GoldenRetriever Export Catalog",
        "RobotState",
        "TaskGoal",
        "Action",
        "Command",
        "Status",
        "convert_to_arrow",
        "convert_from_arrow",
    ),
    "docs-site/src/content/docs/hub/pack-roadmap.mdx": (
        "Pack Maturity Guide",
        "Promotion levels",
        "Maintainer promotion check",
        "Promotion checklist",
    ),
    "docs-site/src/content/docs/robot-payloads/lerobot-export.mdx": (
        "LeRobot Dataset Export",
        "demo-robotics-lerobot-bridge",
        "Canonical rows",
        "LeRobot records",
        "Roundtrip rows",
    ),
    "docs-site/src/content/docs/robot-payloads/index.mdx": (
        "Robot Payload Types",
        "Which payload should this Flow use?",
        "What lives in GoldenRetriever",
        "demo-robotics-typing-catalog",
    ),
    "docs-site/src/content/docs/robot-payloads/type-catalog.mdx": (
        "Choose a Robot Payload",
        "Which type to use",
        "Authoring your own",
        "Validation checklist",
        "Detailed field reference",
    ),
}

SMOKE_CHECKS = (
    (
        "smoke:demo-golden-hub-pack",
        [sys.executable, "examples/advanced/core_composition/golden_hub_pack_smoke.py"],
        ("GoldenRetriever pack exports:", "Arrow round-trip: Action OK"),
    ),
    (
        "smoke:demo-robosuite-mock",
        [
            sys.executable,
            "-m",
            "examples.advanced.robosuite_lift.app",
            "--mode",
            "mock",
            "--steps",
            "4",
            "--dt",
            "0.01",
        ],
        ("[mock step=", "reward="),
    ),
    (
        "smoke:demo-robocasa-mock",
        [
            sys.executable,
            "-m",
            "examples.advanced.robocasa_replay.app",
            "--mode",
            "mock",
            "--steps",
            "14",
        ],
        ("[mock step=0011]", "progress=100.0%", "success=True"),
    ),
    (
        "smoke:demo-pipeline-html-viz",
        [sys.executable, "examples/experimental/visualization/visualize_pipeline.py"],
        ("HTML visualization written to out/golden_retriever_closed_loop_viz.html",),
    ),
    (
        "smoke:demo-robotics-data-eventstream",
        [sys.executable, "examples/advanced/robotics_typing_standard/data_spec_eventstream_demo.py"],
        ("Deterministic merged order:", "Processing-time profile:"),
    ),
    (
        "smoke:demo-robotics-data-join",
        [sys.executable, "examples/advanced/robotics_typing_standard/multi_stream_join_demo.py"],
        ("Exact join:", "Latest-before join", "Window join"),
    ),
    (
        "smoke:demo-robotics-lerobot-bridge",
        [sys.executable, "examples/advanced/robotics_typing_standard/lerobot_bridge_demo.py"],
        ("Canonical rows:", "LeRobot records:", "Roundtrip rows:"),
    ),
)


@dataclass(frozen=True)
class Result:
    name: str
    ok: bool
    detail: str


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def check_required_paths() -> list[Result]:
    results: list[Result] = []
    for path in REQUIRED_PATHS:
        exists = (ROOT / path).exists()
        results.append(Result(f"path:{path}", exists, "exists" if exists else "missing"))
    return results


def git_tracked_under(path: str) -> bool:
    proc = subprocess.run(
        ["git", "ls-files", path],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return bool(proc.stdout.strip())


def check_removed_paths() -> list[Result]:
    results: list[Result] = []
    for path in REMOVED_PATHS:
        exists = (ROOT / path).exists()
        tracked = git_tracked_under(path)
        ok = not exists and not tracked
        detail = "absent" if ok else f"present exists={exists} tracked={tracked}"
        results.append(Result(f"removed:{path}", ok, detail))
    return results


def check_tasks() -> list[Result]:
    pixi = read("pixi.toml")
    results: list[Result] = []
    for task in REQUIRED_TASKS:
        marker = f"{task} ="
        results.append(Result(f"task:{task}", marker in pixi, "declared" if marker in pixi else "missing"))
    return results


def check_doc_markers() -> list[Result]:
    results: list[Result] = []
    for path, markers in DOC_MARKERS.items():
        text = read(path)
        for marker in markers:
            ok = marker in text
            results.append(Result(f"doc:{path}:{marker}", ok, "present" if ok else "missing"))
    return results


def _smoke_env() -> dict[str, str]:
    env = os.environ.copy()
    entries = [str(ROOT)]
    if env.get("PYTHONPATH"):
        entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(entries)
    return env


def check_runtime_smokes(timeout: float) -> list[Result]:
    results: list[Result] = []
    env = _smoke_env()
    for name, command, markers in SMOKE_CHECKS:
        try:
            proc = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            results.append(Result(name, False, f"timeout after {exc.timeout}s"))
            continue
        output = proc.stdout or ""
        missing = [marker for marker in markers if marker not in output]
        ok = proc.returncode == 0 and not missing
        if ok and name == "smoke:demo-pipeline-html-viz":
            artifact = ROOT / "out" / "golden_retriever_closed_loop_viz.html"
            ok = artifact.exists()
            if not ok:
                missing.append(str(artifact.relative_to(ROOT)))
        if ok:
            detail = "ran"
        elif proc.returncode != 0:
            detail = f"exit {proc.returncode}: {output[-500:]}"
        else:
            detail = f"missing markers {missing}: {output[-500:]}"
        results.append(Result(name, ok, detail))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="Skip runtime smoke commands and only check files, tasks, and docs markers.",
    )
    parser.add_argument("--smoke-timeout", type=float, default=30.0)
    args = parser.parse_args()

    results = []
    results.extend(check_required_paths())
    results.extend(check_removed_paths())
    results.extend(check_tasks())
    results.extend(check_doc_markers())
    if not args.static_only:
        results.extend(check_runtime_smokes(args.smoke_timeout))

    if not args.quiet:
        width = max(len(result.name) for result in results)
        for result in results:
            status = "PASS" if result.ok else "FAIL"
            print(f"[{status}] {result.name:<{width}}  {result.detail}")

    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
