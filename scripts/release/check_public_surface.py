#!/usr/bin/env python3
"""Check GoldenRetriever's current public example surface.

The default mode combines static source/docs checks with short runtime smokes
for the promoted public commands: Hub pack proof, mock-safe robosuite, and HTML
pipeline visualization. Use ``--static-only`` for a fast source-tree check.
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
    "examples/advanced/robosuite_lift/app.py",
    "examples/advanced/robosuite_lift/README.md",
    "examples/experimental/visualization/visualize_pipeline.py",
    "examples/experimental/visualization/README.md",
    "examples/advanced/core_composition/golden_hub_pack_smoke.py",
    "docs/examples/README.md",
    "docs/examples/simulation_and_visualization_v1.md",
    "docs/examples/golden_hub_packs_v1.md",
    "docs/hub/README.md",
    "docs/hub/export_catalog_v1.md",
    "docs/hub/module_roadmap_v1.md",
    "docs/llms.txt",
    "docs/robots.txt",
)

REMOVED_PATHS: tuple[str, ...] = ()

REQUIRED_TASKS = (
    "demo-golden-hub-pack",
    "demo-robosuite-mock",
    "demo-pipeline-html-viz",
    "public-surface-check",
)

DOC_MARKERS = {
    "docs/examples/README.md": (
        "demo-golden-hub-pack",
        "demo-robosuite-mock",
        "demo-pipeline-html-viz",
        "out/golden_retriever_closed_loop_viz.html",
    ),
    "docs/examples/simulation_and_visualization_v1.md": (
        "demo-robosuite-mock",
        "demo-pipeline-html-viz",
        "[mock step=...]",
        "out/golden_retriever_closed_loop_viz.html",
    ),
    "docs/hub/README.md": (
        "Retriever Hub Packs",
        "Retriever Hub reference shape",
        "hub.use(\"openretriever/golden-retriever:WorldState\")",
        "Pack Roadmap",
    ),
    "docs/hub/export_catalog_v1.md": (
        "Golden Pack Export Catalog v1",
        "WorldState",
        "convert_to_arrow",
        "convert_from_arrow",
    ),
    "docs/hub/module_roadmap_v1.md": (
        "Retriever Hub Pack Roadmap v1",
        "golden.perception.synthetic_color",
        "import-safe",
        "demo-pipeline-html-viz",
    ),
    "docs/README.md": (
        "Golden reference examples for Retriever",
        "Learn the runtime once in Retriever core",
        "Recommended Path",
        "Demo Gallery",
        "Example Results To Recognize",
        "RobotState",
        "StructuredPlan",
        "ExecutionStatus",
        "Hub-loadable pack",
    ),
    "docs/robots.txt": (
        "Sitemap: https://retriever-space.pages.dev/sitemap.xml",
        "Agent map: https://retriever-space.pages.dev/llms.txt",
    ),
    "docs/llms.txt": (
        "Golden Reference Examples",
        "demo-golden-hub-pack",
        "demo-robosuite-mock",
        "demo-pipeline-html-viz",
        "Do not treat source examples as Retriever Hub packs unless `pyproject.toml` exports them.",
    ),
}

SMOKE_CHECKS = (
    (
        "smoke:demo-golden-hub-pack",
        [sys.executable, "examples/advanced/core_composition/golden_hub_pack_smoke.py"],
        ("Golden pack exports:", "Arrow round-trip: Action OK"),
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
        "smoke:demo-pipeline-html-viz",
        [sys.executable, "examples/experimental/visualization/visualize_pipeline.py"],
        ("HTML visualization written to out/golden_retriever_closed_loop_viz.html",),
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
