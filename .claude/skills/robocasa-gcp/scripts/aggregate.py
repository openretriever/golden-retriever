"""Summarise a run's checkpoints into the numbers that matter.

Reads the per-worker JSONL a sweep writes, so it works on a partial run and on
a run whose VM is long gone.
"""

from __future__ import annotations

import collections
import glob
import json
import sys


def main(root: str) -> int:
    rows = [json.loads(line)
            for path in sorted(glob.glob(f"{root}/w*.jsonl"))
            for line in open(path, errors="replace") if line.strip()]
    if not rows:
        print("  no episodes yet")
        return 0
    by_method = collections.defaultdict(list)
    for row in rows:
        by_method[row["method_id"]].append(row)
    print(f"  {len(rows)} episodes across {len(by_method)} method(s)")
    for method, episodes in sorted(by_method.items()):
        successes = sum(1 for e in episodes if e["success"])
        scores = [e["score"] for e in episodes]
        print(f"\n  {method}")
        print(f"    SR {successes / len(episodes) * 100:5.1f}%  "
              f"({successes}/{len(episodes)})   mean score "
              f"{sum(scores) / len(scores):5.1f}")
        rungs = collections.Counter()
        for episode in episodes:
            rungs.update({k: 1 for k, v in episode["rungs"].items() if v})
        print(f"    rungs {dict(rungs)}")
        steps = sum(e["control_steps"] for e in episodes)
        calls = sum(e["model_calls"] for e in episodes)
        print(f"    control steps {steps:,}   model calls {calls:,}")
        # A refusal becomes a held no-motion action, so a zero score with many
        # refusals is a different finding from a zero score without them.
        refusals = sum(int((e.get("method") or {}).get("refusals") or 0)
                       for e in episodes)
        if refusals:
            print(f"    refusals {refusals} (held no motion - not the same as a weak policy)")
        throttled = sum(int((e.get("method") or {}).get("client_rate_limited") or 0)
                        for e in episodes)
        if throttled:
            print(f"    rate-limited calls {throttled} - the run measured quota, not the model")
        errored = [e for e in episodes if e.get("error")]
        if errored:
            print(f"    episodes ending in an error: {len(errored)}")
            print(f"      e.g. {errored[0]['error'][:100]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
