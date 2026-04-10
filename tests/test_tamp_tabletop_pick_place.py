from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = REPO_ROOT / "examples" / "advanced" / "tamp_tabletop_pick_place" / "app.py"


def _run_demo(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"packages/retriever-tamp/src:."
    return subprocess.run(
        [sys.executable, str(APP_PATH), *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_tabletop_tamp_demo_no_sim_runs() -> None:
    result = _run_demo("--no-obstacle")
    assert result.returncode == 0, result.stderr or result.stdout
    assert "Done. Goal satisfied" in result.stdout


def test_tabletop_tamp_demo_default_obstacle_path_replans_successfully() -> None:
    result = _run_demo()
    assert result.returncode == 0, result.stderr or result.stdout
    assert "place-left-entry@goal_region, place-top-entry@goal_region" in result.stdout
    assert "selected: place-top-entry@goal_region" in result.stdout


def test_tabletop_tamp_demo_pybullet_direct_runs_if_installed() -> None:
    pytest.importorskip("pybullet")
    result = _run_demo("--sim", "pybullet-direct", "--no-obstacle")
    assert result.returncode == 0, result.stderr or result.stdout
    assert "Simulator mode: pybullet-direct" in result.stdout
