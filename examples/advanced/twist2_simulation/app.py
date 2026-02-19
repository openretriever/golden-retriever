"""
TWIST2 Humanoid Simulation - Retriever Port

Demonstrates:
- Frequency decoupling (1000 Hz physics vs 50 Hz policy)
- @gui_flow for native MuJoCo viewer
- Dora backend for high-performance dataflow
"""
import argparse
import tarfile
import urllib.request
from pathlib import Path

import retriever
from retriever.flow import Latest
from flows import Twist2EnvFlow, Twist2PolicyFlow, MotionPlayerFlow, Twist2VisFlow

TWIST2_ASSETS_TAR_URL = (
    "https://codeload.github.com/amazon-far/TWIST2/tar.gz/refs/heads/master"
)


def _normalize_relative_asset_path(path_value: str) -> Path:
    """
    Normalize old path styles into an assets-relative path.

    Accepted examples:
    - TWIST2/assets/g1/g1_sim2sim_29dof.xml
    - TWIST/assets/g1/g1_sim2sim_29dof.xml
    - assets/g1/g1_sim2sim_29dof.xml
    """
    cleaned = path_value.replace("\\", "/")
    if cleaned.startswith("TWIST2/"):
        cleaned = cleaned[len("TWIST2/") :]
    elif cleaned.startswith("TWIST/"):
        cleaned = cleaned[len("TWIST/") :]
    return Path(cleaned)


def _download_assets_to(asset_root: Path) -> None:
    """
    Download TWIST2 assets into a unified local cache folder.

    Layout created:
      {asset_root}/assets/...
    """
    asset_root.mkdir(parents=True, exist_ok=True)
    archive_path = asset_root / "twist2_master.tar.gz"
    print(f"[assets] Downloading TWIST2 assets -> {archive_path}")
    urllib.request.urlretrieve(TWIST2_ASSETS_TAR_URL, archive_path)

    extracted_files = 0
    with tarfile.open(archive_path, "r:gz") as tf:
        for member in tf.getmembers():
            # Keep only the assets subtree from archive members like:
            # TWIST2-master/assets/...
            marker = "/assets/"
            if marker not in member.name:
                continue

            if member.isdir():
                continue

            rel_from_assets = member.name.split(marker, 1)[1]
            target_path = asset_root / "assets" / rel_from_assets
            target_path.parent.mkdir(parents=True, exist_ok=True)

            src = tf.extractfile(member)
            if src is None:
                continue
            with src, open(target_path, "wb") as dst:
                dst.write(src.read())
            extracted_files += 1

    archive_path.unlink(missing_ok=True)
    print(f"[assets] Downloaded {extracted_files} files into {asset_root / 'assets'}")


def _resolve_asset_path(
    requested_path: str,
    *,
    candidate_roots: list[Path],
) -> Path | None:
    """
    Resolve an asset path against known roots.

    Resolution order:
    1) requested path as-is (absolute or relative to cwd)
    2) normalized assets-relative path under each candidate root
    """
    direct = Path(requested_path).expanduser()
    if direct.exists():
        return direct.resolve()

    rel = _normalize_relative_asset_path(requested_path)
    for root in candidate_roots:
        candidate = (root / rel).expanduser()
        if candidate.exists():
            return candidate.resolve()
    return None


def _ensure_required_assets(
    *,
    xml: str,
    policy: str,
    motion: str,
    asset_root: Path,
    auto_download: bool,
) -> tuple[Path, Path, Path]:
    """
    Resolve required assets, optionally downloading them into the local cache.
    """
    roots = [
        Path.cwd(),
        Path.cwd() / "TWIST2",
        Path.cwd() / "TWIST",
        asset_root,
    ]

    xml_path = _resolve_asset_path(xml, candidate_roots=roots)
    policy_path = _resolve_asset_path(policy, candidate_roots=roots)
    motion_path = _resolve_asset_path(motion, candidate_roots=roots)

    if xml_path and policy_path and motion_path:
        return xml_path, policy_path, motion_path

    # If any are missing, populate local cache once and retry.
    if auto_download:
        print("[assets] Some assets are missing, fetching TWIST2 assets to local cache...")
        _download_assets_to(asset_root)

        roots = [asset_root] + roots
        xml_path = _resolve_asset_path(xml, candidate_roots=roots)
        policy_path = _resolve_asset_path(policy, candidate_roots=roots)
        motion_path = _resolve_asset_path(motion, candidate_roots=roots)

    missing = []
    if xml_path is None:
        missing.append(xml)
    if policy_path is None:
        missing.append(policy)
    if motion_path is None:
        missing.append(motion)

    if missing:
        missing_str = "\n  - ".join(missing)
        raise FileNotFoundError(
            "Required TWIST2 assets not found:\n"
            f"  - {missing_str}\n"
            f"Searched local roots: {[str(r) for r in roots]}\n"
            f"Asset cache root: {asset_root}"
        )

    return xml_path, policy_path, motion_path


def main():
    parser = argparse.ArgumentParser(description="Retriever TWIST2 Port")
    parser.add_argument("--xml", type=str, default="assets/g1/g1_sim2sim_29dof.xml")
    parser.add_argument("--policy", type=str, default="assets/ckpts/twist2_1017_20k.onnx")
    parser.add_argument("--motion", type=str, default="assets/example_motions/0807_yanjie_walk_001.pkl")
    parser.add_argument(
        "--asset-root",
        type=str,
        default="assets/twist2",
        help="Unified local cache folder for TWIST2 assets.",
    )
    parser.add_argument(
        "--no-auto-download",
        action="store_true",
        help="Disable automatic download of missing TWIST2 assets.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--backend", type=str, default="dora")
    parser.add_argument("--no-viewer", action="store_true", help="Disable native viewer")
    args = parser.parse_args()

    asset_root = Path(args.asset_root).expanduser().resolve()
    xml_path, policy_path, motion_path = _ensure_required_assets(
        xml=args.xml,
        policy=args.policy,
        motion=args.motion,
        asset_root=asset_root,
        auto_download=not args.no_auto_download,
    )

    print("[assets] Using:")
    print(f"  xml:    {xml_path}")
    print(f"  policy: {policy_path}")
    print(f"  motion: {motion_path}")

    # Define Flows with Rate Decoupling
    motion = MotionPlayerFlow(motion_file=str(motion_path)) @ retriever.Rate(50)
    policy = Twist2PolicyFlow(policy_path=str(policy_path), device=args.device) @ retriever.Rate(50)
    env = Twist2EnvFlow(xml_path=str(xml_path)) @ retriever.Rate(500)
    vis = None
    if not args.no_viewer:
        vis = Twist2VisFlow(xml_path=str(xml_path)) @ retriever.Rate(30)

    # Build pipeline with .then() chaining
    motion.then(env, sync=Latest())
    motion.then(policy, sync=Latest())
    env.then(policy, sync=Latest())
    policy.then(env, sync=Latest())
    if vis is not None:
        env.then(vis, map={"vis": "vis"}, sync=Latest())

    # Get pipeline from any handle
    pipe = motion.pipeline
    pipe.name = "twist2_demo"

    print("\n=== TWIST2 Simulation (Retriever Port) ===")
    print(f"  Physics: 500 Hz")
    print(f"  Policy:   50 Hz")
    print(f"  Backend: {args.backend}")
    print("==========================================")

    # Run with Rerun visualization enabled
    pipe.run(backend=args.backend, duration=10.0, visualize="rerun")


if __name__ == "__main__":
    main()
