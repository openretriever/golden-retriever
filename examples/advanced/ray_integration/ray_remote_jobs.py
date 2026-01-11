"""
Ray remote jobs demo (local or cluster).

Run (local, auto-start Ray):
  python examples/advanced/ray_integration/ray_remote_jobs.py

Run (connect to existing cluster):
  python examples/advanced/ray_integration/ray_remote_jobs.py --address ray://127.0.0.1:10001
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class JobResult:
    job_id: int
    node_ip: str
    pid: int
    value: float
    elapsed_ms: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ray remote job demonstration.")
    p.add_argument(
        "--address",
        default="",
        help="Ray address (empty means start local Ray). Example: ray://127.0.0.1:10001",
    )
    p.add_argument("--jobs", type=int, default=8, help="Number of remote jobs.")
    p.add_argument("--sleep-ms", type=float, default=20.0, help="Work per job.")
    p.add_argument("--seed", type=int, default=3, help="Deterministic workload seed.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import ray
    except ImportError:
        print("Ray is not installed. Try: pip install ray")
        return

    if args.address:
        ray.init(address=args.address)
    else:
        ray.init()

    print("Ray version:", ray.__version__)
    print("Ray cluster resources:", ray.cluster_resources())
    print("Ray available resources:", ray.available_resources())

    @ray.remote
    def work(job_id: int, sleep_ms: float, seed: int) -> JobResult:
        import os
        import socket

        t0 = time.time()
        time.sleep(sleep_ms / 1000.0)
        value = (job_id + 1) * (seed + 0.5)
        elapsed_ms = (time.time() - t0) * 1000.0
        return JobResult(
            job_id=job_id,
            node_ip=socket.gethostbyname(socket.gethostname()),
            pid=os.getpid(),
            value=value,
            elapsed_ms=elapsed_ms,
        )

    futures = [work.remote(i, args.sleep_ms, args.seed) for i in range(args.jobs)]
    results = ray.get(futures)

    results = sorted(results, key=lambda r: r.job_id)
    for res in results:
        print(
            f"[job {res.job_id}] value={res.value:.2f} node={res.node_ip} "
            f"pid={res.pid} elapsed={res.elapsed_ms:.1f}ms"
        )

    ray.shutdown()


if __name__ == "__main__":
    main()
