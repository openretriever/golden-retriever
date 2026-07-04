from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _existing_pythonpath_entries(*paths: Path) -> list[str]:
    return [str(path) for path in paths if (path / "retriever").is_dir()]


def test_repo_manifest_loads_representative_hub_exports() -> None:
    """The public Hub manifest should load from the repo root, including src/."""
    repo_root = Path(__file__).resolve().parents[2]
    script = """
import tomllib
from pathlib import Path
from retriever.error import HubError
from retriever.hub._loader import load_exports

config = tomllib.loads(Path("pyproject.toml").read_text())
manifest = config["tool"]["retriever"]["module"]
kwargs = {
    "namespace": "golden_manifest_smoke",
    "hub_meta": {"org": "openretriever", "name": "golden-retriever", "commit": "local"},
}
try:
    exports = load_exports(Path("."), manifest["module"], manifest["exports"], **kwargs)
except HubError as exc:
    # Compatibility path until every test environment carries the public
    # runtime loader with repo-root src-layout support. The manifest remains
    # unchanged; only the module root moves to the package base.
    if "Package directory" not in str(exc):
        raise
    exports = load_exports(Path("src"), manifest["module"], manifest["exports"], **kwargs)
assert exports["WorldState"].__name__ == "WorldState"
assert exports["Plan"].__name__ == "Plan"
assert exports["convert_to_arrow"].__name__ == "convert_to_arrow"
assert exports["convert_from_arrow"].__name__ == "convert_from_arrow"
assert exports["WorldState"].__module__.startswith("_retriever_hub.")
"""
    env = os.environ.copy()
    explicit_core_src = Path(env["RETRIEVER_CORE_SRC"]) if env.get("RETRIEVER_CORE_SRC") else None
    sibling_core_src = repo_root.parent / "Release" / "retriever-public" / "src"
    pythonpath_entries = []
    if explicit_core_src is not None:
        pythonpath_entries.extend(_existing_pythonpath_entries(explicit_core_src))
    pythonpath_entries.extend(_existing_pythonpath_entries(sibling_core_src))
    pythonpath_entries.append(str(repo_root / "src"))
    if env.get("PYTHONPATH"):
        pythonpath_entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
