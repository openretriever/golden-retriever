"""A reference payload for `run.sh`, and the skill's own end-to-end test.

`run.sh` shards by task and requires a module accepting `--task NAME`
(repeatable) and `--checkpoint PATH`. Nothing in the repository satisfied that
contract, which made the contract undemonstrable and the generic path
untestable. This is the smallest thing that does satisfy it.

It measures nothing about any policy: it builds each kitchen, steps it with zero
actions, and records that it worked. That is the point — it exercises the
sharding, the checkpointing, the GCS mirror and the resume without involving a
benchmark, so a failure here is the skill's and not an experiment's.

Copy it as the starting point for a real payload. The two things that matter are
the flags and the per-task JSONL line; everything else is yours.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
import warnings


def finished(checkpoint: pathlib.Path) -> set:
    """Tasks already recorded, so a resumed run skips them.

    This is what makes a lost VM cost one task instead of a run: `run.sh`
    restores the checkpoint from GCS and the payload declines to redo work.
    """
    if not checkpoint.exists():
        return set()
    done = set()
    for line in checkpoint.read_text(errors="replace").splitlines():
        if line.strip():
            done.add(json.loads(line)["task"])
    return done


def run_task(task: str, *, steps: int, seed: int) -> dict:
    import numpy as np
    import robocasa  # noqa: F401  importing it registers the kitchen tasks
    import robosuite
    from robosuite.controllers import load_composite_controller_config

    started = time.perf_counter()
    env = robosuite.make(
        task, robots="PandaOmron",
        controller_configs=load_composite_controller_config(robot="PandaOmron"),
        has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
        camera_names=["robot0_agentview_left"], camera_heights=256, camera_widths=256,
        seed=seed, layout_ids=[1], style_ids=[1])
    observation = env.reset()
    frame = np.asarray(observation["robot0_agentview_left_image"])
    # A blank frame means software GL is not working, which otherwise looks
    # exactly like a policy that cannot act.
    rendered = float(frame.std()) > 1.0
    action = np.zeros(env.action_dim, dtype=np.float32)
    for _ in range(steps):
        env.step(action)
    success = bool(env._check_success())
    env.close()
    return {
        "task": task, "seed": seed, "method_id": "example:zero-action",
        "success": success, "score": 0,
        "rungs": {"reached": False, "grasped": False, "lifted": False,
                  "succeeded": success},
        "control_steps": steps, "model_calls": 0,
        "representation": "native", "horizon": 1,
        "latency_s": round(time.perf_counter() - started, 3),
        "clip": {}, "backend": {"backend": "robocasa", "task": task,
                                "rendered": rendered,
                                "frame_std": round(float(frame.std()), 2)},
        "method": {}, "error": "",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="example_payload")
    parser.add_argument("--task", action="append", dest="tasks", required=True,
                        help="Task to run; repeatable. run.sh supplies one per shard.")
    parser.add_argument("--checkpoint", default="",
                        help="JSONL appended per finished task, and read to skip work.")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")

    checkpoint = pathlib.Path(args.checkpoint) if args.checkpoint else None
    already = finished(checkpoint) if checkpoint else set()
    if already:
        print(f"  skipping {len(already)} task(s) already in the checkpoint")

    records = []
    for task in args.tasks:
        if task in already:
            continue
        try:
            record = run_task(task, steps=args.steps, seed=args.seed)
        except Exception as error:  # noqa: BLE001 - one bad task must not lose the shard
            record = {"task": task, "seed": args.seed,
                      "method_id": "example:zero-action", "success": False,
                      "score": 0, "rungs": {}, "control_steps": 0, "model_calls": 0,
                      "representation": "native", "horizon": 1, "latency_s": 0.0,
                      "clip": {}, "backend": {"backend": "robocasa", "task": task},
                      "method": {}, "error": f"{type(error).__name__}: {error}"[:200]}
        records.append(record)
        print(f"  {task}: rendered="
              f"{record['backend'].get('rendered')} steps={record['control_steps']}"
              f" {('error: ' + record['error']) if record['error'] else ''}")
        if checkpoint:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            with checkpoint.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
    print(json.dumps({"ran": len(records), "skipped": len(already)}, sort_keys=True))
    return 1 if any(r["error"] for r in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
