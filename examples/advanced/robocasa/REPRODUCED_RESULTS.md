# Reproduced Results: Embodied Methods Harness

Date: 2026-08-29

Branch: `feat/robocasa-method-harness`

## Scope

This reproduction validates a cross-platform Retriever harness and browser
visualization for embodied code-as-policy methods. It does not claim to
reproduce a learned policy, generated-code agent, or real-robot result.

The public implementation re-creates these useful patterns:

- explicit benchmark axes for interaction, grounding, abstraction, feedback,
  and examples;
- a simulator-neutral environment lifecycle;
- typed, ordered execution events;
- bounded future-action chunks;
- native task verification; and
- a combined catalog and live simulator dashboard.

Generated Python execution, arbitrary configuration imports, unsafe thread
interruption, and unauthenticated real-robot transport were intentionally not
adopted.

## Environment

| Component | Version |
| --- | --- |
| Tested platform | macOS, arm64 |
| Python | 3.11.15 |
| MuJoCo | 3.3.1 |
| robosuite | 1.5.2 |
| viser | 1.1.0 |

The benchmark-design reference was
[Cap-X at `53e9966`](https://github.com/capgym/cap-x/tree/53e9966d7a8e2fa7494676772bccc35280f5c0ed)
(MIT). GoldenRetriever reimplements the useful harness concepts around its own
typed Flow and action-chunk interfaces; Cap-X is not a runtime dependency and
no generated-code executor was copied.

## Results

| Check | Result |
| --- | --- |
| Dashboard starts on macOS | Pass |
| Real robosuite Lift scene loads in embedded mjviser | Pass |
| One active simulator is replaced and stopped cleanly | Pass |
| Status polling leaves the iframe mounted | Pass |
| Collision and below-floor debug targets are hidden | Pass |
| Typed action safety envelope rejects invalid chunks | Pass |
| Ordered events and trial report | Pass |
| Native verification represented separately from process health | Pass |
| Scripted Lift reaches native task success | Pass (56 actions, repeated twice) |
| Linux x86_64 smoke run | Not run |
| WSL2 smoke run | Not run |

Supported repository suite:

```text
185 passed, 13 skipped in 7.97s
```

Real harness-backed Lift run:

```text
[harness verification step=056] success: robosuite native task success
[harness report] status=success success=True steps=56 reward=1.000
```

The simulator, rendering, typed action-chunk boundary, and native verification
path are reproduced successfully. The same fixed-seed command was run in two
fresh processes and produced the same 56-action result both times. This is a
scripted smoke baseline, not evidence of a learned policy or broad robustness.

## Reproduce

From the repository root, create the pinned simulator environment described in
[`README.md`](README.md), then run:

```bash
python -m examples.advanced.robocasa.launcher --duration 300
```

Open `http://localhost:8084`, select **Scripted privileged cube lift**, and
launch it. The embedded viewer runs at `http://localhost:8085`.

Headless measurement:

```bash
python -m examples.advanced.robocasa.robosuite_lift \
  --mode robosuite --env Lift --seed 0 --steps 300 --dt 0.05 \
  --print-every 100 --harness
```

## Next Reproducible Milestones

1. Report the scripted adapter's success rate across fixed seeds and episodes.
2. Add a reduced RGB-D API worker on Linux with the same safety envelope.
3. Add bounded visual-feedback replanning as typed events, without generated
   code execution.
4. Report success rate across fixed seeds and episodes, not a single video.
