from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_repo_manifest_loads_representative_hub_exports() -> None:
    """The public Hub manifest should load from the repo root, including src/."""
    repo_root = Path(__file__).resolve().parents[2]
    script = """
import tomllib
from pathlib import Path
from retriever.hub._loader import load_exports

config = tomllib.loads(Path("pyproject.toml").read_text())
manifest = config["tool"]["retriever"]["module"]
exports = load_exports(
    Path("."),
    manifest["module"],
    manifest["exports"],
    namespace="golden_manifest_smoke",
    hub_meta={"org": "openretriever", "name": "golden-retriever", "commit": "local"},
)
assert exports["WorldState"].__name__ == "WorldState"
assert exports["Plan"].__name__ == "Plan"
assert exports["convert_to_arrow"].__name__ == "convert_to_arrow"
assert exports["convert_from_arrow"].__name__ == "convert_from_arrow"
assert exports["WorldState"].__module__.startswith("_retriever_hub.")
"""
    env = os.environ.copy()
    pythonpath = str(repo_root / "src")
    if env.get("PYTHONPATH"):
        pythonpath = pythonpath + os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
