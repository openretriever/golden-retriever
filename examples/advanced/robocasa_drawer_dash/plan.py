"""The choreography the arm runs, as plain data.

This module is deliberately dependency-free — no MuJoCo, no RoboCasa, no
NumPy — so the mock lane, the tests and the docs can all describe the same
routine without a simulator or an asset pack present. `sequence.py` imports
these phases and executes them against a real model; `app.py` replays them
against either the simulator or a deterministic mock.

The routine:

  1. settle              - arm at its home pose, fingers open
  2. line up             - move to a standoff in front of the handle
  3. close in            - advance until the open fingers straddle the bar
  4. grip handle         - close the fingers onto the bar
  5. pull drawer open    - the hand travels out; the drawer comes with it
  6. let go of handle    - open the fingers, drawer stays put
  7. back off            - withdraw to the standoff
  8. stand clear         - home, so the wrist can turn for a different grasp
  9. reach over jar      - square up above the seasoning on the worktop
  9. down to jar         - descend around its barrel
 10. grip jar            - close on it
 11. lift jar            - straight up, clear of the worktop
 12. carry to drawer     - across to a free spot in the open drawer
 13. lower into drawer   - down into it
 14. let go of jar       - the seasoning is put away
 15. lift clear          - back up out of the drawer
 16. back to handle      - return to the standoff
 17. close in again      - straddle the bar a second time
 18. grip handle again   - close on it
 19. push drawer shut    - and back
 20. release             - open the fingers
 21. withdraw            - home

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

# The seasoning jar is a 48 mm cylinder and the gripper spans 80 mm, so it is
# grasped around the barrel rather than pinched. Commanding 0 the way the
# handle does would stall the fingers 24 mm inside the glass and drive the
# actuators to their 100 N limit; 18 mm leaves a firm grip at about 48 N.
# The Omron torso is driven by the routine rather than parked. The worktop is
# out of reach with the torso down, and the inside of the open drawer is out
# of reach with it up, so the errand raises it to fetch and lowers it to put
# away. `scene.py` bakes the down position into the home keyframe.
TORSO_DOWN = 0.0
TORSO_UP = 0.20

# The errand travels high and in front of the drawer rather than folding back
# through it. The arm's rest pose sits inside the swept volume of the open
# drawer, so going home mid-routine shoves the drawer half shut -- which is
# both untidy and the exact thing this scene claims cannot happen by accident.
TRANSIT_Z = 1.35
TRANSIT_STANDOFF = 0.14

JAR_SQUEEZE = 0.018
JAR_GRASP_Z = 1.15  # on the barrel, below the lid
JAR_CLEAR_Z = 1.28  # approach and lift height, clear of the worktop

# Where the jar is put down, inside the open drawer: x in world, y measured
# back from the drawer's interior centre so it follows the drawer out.
SLOT_X = -0.15
SLOT_Y_FROM_CENTRE = 0.04
SLOT_CLEAR_Z = 1.25  # above the drawer rim, with the jar hanging below the hand
SLOT_PLACE_Z = 0.97  # low enough that the jar is set down, not dropped
# The drawer front carries the handle; its interior centre sits this far
# behind the bar, so the slot travels with the drawer.
HANDLE_TO_INTERIOR = 0.24

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
    mode: str  # home | standoff | engage | carry | jar_* | slot_*
    grip: float  # commanded finger opening
    stroke: tuple[float, float] = (0.0, 0.0)  # carry: pull distance start -> end
    torso: float = 0.0  # commanded Omron torso height, metres


PHASES: tuple[Phase, ...] = (
    Phase("settle", 0.8, "home", OPEN),
    Phase("line up", 2.0, "standoff", OPEN),
    Phase("close in", 1.2, "engage", OPEN),
    Phase("grip handle", 0.8, "engage", SQUEEZE),
    Phase("pull drawer open", 3.0, "carry", SQUEEZE, (0.0, STROKE)),
    # The slides are damped, so an unheld drawer simply stays where it was
    # left. That is what makes it safe to go and fetch the seasoning here.
    Phase("let go of handle", 0.6, "engage", OPEN),
    Phase("back off", 1.2, "standoff", OPEN),
    # Home before the jar, and again before coming back to the handle. The
    # two grasps want wrists a quarter turn apart -- the handle is pinched
    # top to bottom, the jar is taken from above -- and the arm will not make
    # that turn at full stretch: driven straight from one to the other it
    # arrives 60 degrees off square and knocks the jar over instead of
    # closing on it. From the rest pose it arrives square.
    Phase("lift clear of drawer", 3.0, "transit_front", OPEN, torso=TORSO_UP),
    Phase("traverse to worktop", 2.5, "transit_worktop", OPEN, torso=TORSO_UP),
    Phase("reach over jar", 2.5, "jar_clear", OPEN, torso=TORSO_UP),
    Phase("down to jar", 3.0, "jar_grasp", OPEN, torso=TORSO_UP),
    Phase("grip jar", 1.0, "jar_grasp", JAR_SQUEEZE, torso=TORSO_UP),
    Phase("lift jar", 1.8, "jar_clear", JAR_SQUEEZE, torso=TORSO_UP),
    # The torso comes back down here: the inside of the drawer is below what
    # the arm can reach with it raised.
    Phase("carry to drawer", 3.0, "slot_clear", JAR_SQUEEZE, torso=TORSO_DOWN),
    Phase("lower into drawer", 2.0, "slot_place", JAR_SQUEEZE, torso=TORSO_DOWN),
    Phase("let go of jar", 0.8, "slot_place", OPEN),
    Phase("lift clear", 1.6, "slot_clear", OPEN),
    Phase("rise clear", 2.0, "transit_front", OPEN),
    Phase("back to handle", 2.6, "standoff", OPEN),
    Phase("close in again", 1.2, "engage", OPEN),
    Phase("grip handle again", 0.8, "engage", SQUEEZE),
    # Overshoot the close by 15 mm: the slide has enough compliance that
    # stopping exactly at the anchor leaves the drawer standing proud.
    Phase("push drawer shut", 3.0, "carry", SQUEEZE, (STROKE, -0.015)),
    Phase("release", 0.8, "engage", OPEN),
    Phase("withdraw", 3.2, "home", OPEN),
)

TOTAL_SECONDS = sum(p.seconds for p in PHASES)

# Phases in which the hand is meant to be holding the bar.
GRASPING = frozenset({
    "grip handle", "pull drawer open", "grip handle again", "push drawer shut",
})
# ... and the ones in which it is meant to be holding the seasoning jar.
HOLDING_JAR = frozenset({
    "grip jar", "lift jar", "carry to drawer", "lower into drawer",
})
# The phases that command the drawer to move. Between them the drawer stands
# open with nothing touching it, which is when the jar is put away.
MOVING_DRAWER = frozenset({"pull drawer open", "push drawer shut"})


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
