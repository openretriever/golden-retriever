"""The choreography the arm runs, as plain data.

This module is deliberately dependency-free — no MuJoCo, no RoboCasa, no
NumPy — so the mock lane, the tests and the docs can all describe the same
routine without a simulator or an asset pack present. `sequence.py` imports
these phases and executes them against a real model; `app.py` replays them
against either the simulator or a deterministic mock.

The routine:

  1. settle            - arm at its home pose, fingers open
  2. line up           - move to a standoff in front of the handle, squared up
  3. close in          - advance until the open fingers straddle the handle bar
  4. grip handle       - close the fingers onto the bar
  5. pull drawer open  - the hand travels out; the drawer comes with it
  6. push drawer shut  - and back
  7. release           - open the fingers
  8. back off          - withdraw to the standoff
  9. withdraw          - home

Nothing actuates the drawers. Their slide joints are passive and damped, so
steps 5 and 6 only happen if the grasp in step 4 is real: if the fingers miss
the bar or slip off it, the drawer simply stays shut.
"""

from __future__ import annotations

from dataclasses import dataclass

STANDOFF = 0.14  # metres in front of the bar to line up from
STROKE = 0.41  # metres the hand pulls; the slide's own range is 0.45
OPEN = 0.04  # finger opening that clears the bar
SQUEEZE = 0.0  # commanded closed; the bar stops the fingers and they hold

HANDLE_GEOM = "drawer2_door_handle_g1"  # the bar itself
HANDLE_BODY = "drawer2_door_handle_main"  # fallback if the geom is renamed
DRAWER_JOINT = "drawer2_slidejoint"

# Thresholds the run is judged against, shared by `verify.py` and the mock.
OPENED_MIN = 0.35  # metres the drawer must come out under the grasp
SHUT_MAX = 0.02  # metres it may still stand proud after being pushed back


@dataclass(frozen=True)
class Phase:
    label: str
    seconds: float
    mode: str  # home | standoff | engage | carry
    grip: float  # commanded finger opening
    stroke: tuple[float, float] = (0.0, 0.0)  # carry: pull distance start -> end


PHASES: tuple[Phase, ...] = (
    Phase("settle", 0.8, "home", OPEN),
    Phase("line up", 2.5, "standoff", OPEN),
    Phase("close in", 1.5, "engage", OPEN),
    Phase("grip handle", 0.8, "engage", SQUEEZE),
    Phase("pull drawer open", 3.5, "carry", SQUEEZE, (0.0, STROKE)),
    # Overshoot the close by 15 mm: the slide has enough compliance that
    # stopping exactly at the anchor leaves the drawer standing proud.
    Phase("push drawer shut", 3.0, "carry", SQUEEZE, (STROKE, -0.015)),
    Phase("release", 0.8, "engage", OPEN),
    Phase("back off", 1.5, "standoff", OPEN),
    Phase("withdraw", 2.0, "home", OPEN),
)

TOTAL_SECONDS = sum(p.seconds for p in PHASES)

# Phases in which the hand is meant to be holding the bar.
GRASPING = frozenset({"grip handle", "pull drawer open", "push drawer shut"})


def smoothstep(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def phase_at(elapsed: float) -> tuple[int, Phase, float]:
    """Which phase is in force `elapsed` seconds in, and how far through it.

    Returns `(index, phase, blend)` where `blend` is the smoothstepped
    progress through that phase. Past the end of the routine this reports the
    final phase, fully blended, so a caller that overruns simply holds.
    """
    remaining = float(elapsed)
    for index, phase in enumerate(PHASES):
        if remaining < phase.seconds:
            return index, phase, smoothstep(remaining / phase.seconds)
        remaining -= phase.seconds
    return len(PHASES) - 1, PHASES[-1], 1.0
