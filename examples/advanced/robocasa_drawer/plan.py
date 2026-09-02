"""The choreography the arm runs, as plain data.

This module is deliberately dependency-free — no MuJoCo, no RoboCasa, no
NumPy — so the mock lane, the tests and the docs can all describe the same
routine without a simulator or an asset pack present. `sequence.py` imports
these phases and executes them against a real model; `app.py` replays them
against either the simulator or a deterministic mock.

The routine:

   1. settle                 - arm at its home pose, fingers open
   2. line up on the handle  - a standoff in front of the top drawer's bar
   3. close in               - advance until the open fingers straddle the bar
   4. grip the handle        - close the fingers onto it
   5. pull the drawer open   - the hand travels out; the drawer comes with it
   6. let go of the handle   - open the fingers
   7. back off the handle    - clear of the bar
   8. rise over the drawer   - up above the open drawer front
   9. over the shaker        - across to a hover above the pepper shaker
  10. down to the shaker     - descend around it, fingers open
  11. grip the shaker        - close onto its waist
  12. lift it out            - straight up out of the drawer
  13. out over the plate     - forward, over the drawer front, in front of it
  14. down to the plate      - down to a working height above the food
  15. tip it over the food   - roll the shaker past horizontal, cap down
  16. shake the seasoning    - oscillate roll and height over the plate
  17. bring it upright       - roll back
  18. lift off the plate     - back up to transit height
  19. back over the drawer   - across, above the drawer again
  20. lower it back in       - down to where it was picked from
  21. let go of the shaker   - open the fingers
  22. lift clear             - straight up, out of the drawer
  23. swing to the front     - over the drawer front again
  24. back to the handle     - down to the standoff, squared up to the bar
  25. close in again         - onto the bar
  26. grip the handle again  - close the fingers
  27. push the drawer shut   - and the drawer goes back
  28. release                - open the fingers
  29. back off               - withdraw to the standoff
  30. withdraw               - home

Nothing actuates the drawers. Their slide joints are passive and damped, so
steps 5 and 27 only happen if the grasps in steps 4 and 26 are real. Nothing
holds the shaker either: it is a free body standing on the drawer floor, so
steps 12-20 only happen if the grasp in step 11 is real.

Two grasps, two orientations. The handle is a cylinder lying along world x
standing about 8 mm proud of the drawer front, so the fingers close *vertically*
across it: the gripper's approach axis points at the drawer, its closing axis
straight up. The shaker stands upright on the drawer floor, so the fingers close
*horizontally* around its waist: the approach axis points straight down. Tipping
the shaker is then a roll about the closing axis — the fingers keep their hold
and the shaker turns over with them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# --- the handle ------------------------------------------------------------
STANDOFF = 0.14      # metres in front of the bar to line up from, drawer shut
CLEARANCE = 0.05     # ... and once it is open and the bar has come to meet you
FRONT_GAP = 0.13     # how far in front of the open drawer the transit lane runs
STROKE = 0.22        # metres the drawer is pulled out; its range is 0.45
OVERSHOOT = 0.015    # push this much past shut: the slide has some compliance

# --- the shaker ------------------------------------------------------------
# Where to take hold of it, measured up from the body origin (its centre). The
# shaker's profile has a waist a little above the middle — 30 mm across against
# 48 mm at the base — so closing there is a form fit, not just friction.
GRIP_RISE = 0.0135
PLACE_RISE = 0.008   # release it this far above where it was picked up
TRANSIT_Z = 1.24     # crossing height: clears the open drawer front and handle
HOVER_RISE = 0.11    # hover this far above the shaker before descending

# --- the plate -------------------------------------------------------------
# Where the shake happens, above the plate. With the torso down it lands a few
# centimetres above the shoulder's own height, and that is not a coincidence:
# the shoulder is what runs out of travel when the shaker is tipped right over,
# and it has the most room to give when the hand is level with it.
WORK_RISE = 0.258
# Roll past horizontal so the cap points down over the food. The sign matters:
# rolling this way swings the wrist down and back towards the robot, clear of
# everything. Rolling the other way swings it up and forwards, straight into the
# drawer the arm has just pulled open, and shoves it shut.
TIP_ANGLE = math.radians(138.0)

SHAKE_ROLL = math.radians(20.0)   # ... and this much either side of it
SHAKE_LIFT = 0.030   # metres of bob, in step with the roll
SHAKE_HZ = 2.0
# The shake is a fast oscillation, and the reach loop's ordinary gains are far
# too slow to follow one — at the default the hand would barely move.
SHAKE_GAIN = 28.0
SHAKE_SPIN_GAIN = 30.0
SHAKE_RATE = 5.0     # rad/s per joint, up from the usual 1.8

# --- the base's torso lift ---------------------------------------------------
# Up for the drawer and down for the plate: the Panda cannot cover a handle a
# metre up and a plate on the tabletop from one shoulder height. It only moves
# while the hand is out in front of the dresser — dropping the shoulder while
# the arm still reaches over the open drawer drags the forearm through it and
# pulls the drawer wide open.
TORSO_HIGH = 0.24    # matches `scene.py`'s TORSO_HEIGHT: where it starts
TORSO_LOW = 0.0
TORSO_SPEED = 0.18   # metres per second

# --- grips -----------------------------------------------------------------
OPEN = 0.04          # finger opening that clears both the bar and the shaker
HANDLE_SQUEEZE = 0.0    # commanded shut; the bar stops the fingers and they hold
SHAKER_SQUEEZE = 0.012  # the same, against the shaker's 30 mm waist

HANDLE_GEOM = "drawer2_door_handle_g1"    # the bar itself
HANDLE_BODY = "drawer2_door_handle_main"  # fallback if the geom is renamed
DRAWER_JOINT = "drawer2_slidejoint"
SHAKER_BODY = "pepper_main"
PLATE_BODY = "plate_main"

# Thresholds the run is judged against, shared by `verify.py` and the mock.
OPENED_MIN = 0.18      # metres the drawer must come out under the grasp
SHUT_MAX = 0.02        # metres it may still stand proud after being pushed back
LIFT_MIN = 0.10        # metres the shaker must rise clear of the drawer floor
TIP_MIN = math.radians(100.0)  # how far past upright it must be tipped to season


@dataclass(frozen=True)
class Phase:
    label: str
    seconds: float
    mode: str            # see sequence.Choreography.target_pose
    grip: float          # commanded finger opening
    goal: str = ""       # waypoint name, for the `move` mode
    open_to: float = 0.0  # carry: how far out to leave the drawer, in metres
    tilt: tuple[float, float] = (0.0, 0.0)    # tilt/shake: roll start -> end
    mark_grip: bool = False  # remember where the shaker was taken hold of
    torso: float = TORSO_HIGH  # where the base's lift should be by the end


PHASES: tuple[Phase, ...] = (
    Phase("settle", 1.2, "home", OPEN),
    Phase("line up on the handle", 2.4, "standoff", OPEN),
    Phase("close in", 1.2, "engage", OPEN),
    Phase("grip the handle", 0.8, "engage", HANDLE_SQUEEZE),
    Phase("pull the drawer open", 3.0, "carry", HANDLE_SQUEEZE, open_to=STROKE),
    Phase("let go of the handle", 0.6, "engage", OPEN),
    Phase("back off the handle", 1.2, "standoff", OPEN),
    Phase("rise over the drawer", 2.2, "move", OPEN, goal="front_high"),
    Phase("over the shaker", 1.8, "move", OPEN, goal="shaker_hover"),
    Phase("down to the shaker", 1.4, "move", OPEN, goal="shaker_grip"),
    Phase("grip the shaker", 0.8, "hold", SHAKER_SQUEEZE),
    Phase("lift it out", 1.5, "move", SHAKER_SQUEEZE, goal="straight_up",
          mark_grip=True),
    Phase("out over the plate", 2.2, "move", SHAKER_SQUEEZE, goal="plate_high"),
    Phase("down to the plate", 2.4, "move", SHAKER_SQUEEZE, goal="work",
          torso=TORSO_LOW),
    Phase("tip it over the food", 2.4, "tilt", SHAKER_SQUEEZE,
          tilt=(0.0, TIP_ANGLE), torso=TORSO_LOW),
    Phase("shake the seasoning", 2.0, "shake", SHAKER_SQUEEZE, torso=TORSO_LOW),
    Phase("bring it upright", 3.4, "tilt", SHAKER_SQUEEZE, tilt=(TIP_ANGLE, 0.0),
          torso=TORSO_LOW),
    Phase("lift off the plate", 2.2, "move", SHAKER_SQUEEZE, goal="plate_high"),
    Phase("back over the drawer", 2.0, "move", SHAKER_SQUEEZE, goal="over_slot"),
    Phase("lower it back in", 1.8, "move", SHAKER_SQUEEZE, goal="place"),
    Phase("let go of the shaker", 0.7, "move", OPEN, goal="place"),
    Phase("lift clear", 1.4, "move", OPEN, goal="straight_up"),
    Phase("swing to the front", 2.0, "move", OPEN, goal="front_high"),
    Phase("back to the handle", 2.0, "standoff", OPEN),
    Phase("close in again", 1.2, "engage", OPEN),
    Phase("grip the handle again", 0.8, "engage", HANDLE_SQUEEZE),
    Phase("push the drawer shut", 2.8, "carry", HANDLE_SQUEEZE, open_to=-OVERSHOOT),
    Phase("release", 0.7, "engage", OPEN),
    Phase("back off", 1.2, "standoff", OPEN),
    Phase("withdraw", 2.0, "home", OPEN),
)

TOTAL_SECONDS = sum(p.seconds for p in PHASES)

# Phases in which the hand is meant to be holding the handle bar.
GRASPING = frozenset({
    "grip the handle", "pull the drawer open",
    "grip the handle again", "push the drawer shut",
})
# ... and the shaker.
CARRYING = frozenset({
    "grip the shaker", "lift it out", "out over the plate", "down to the plate",
    "tip it over the food", "shake the seasoning", "bring it upright",
    "lift off the plate", "back over the drawer", "lower it back in",
})
# The phases that command the drawer to move. Between them the drawer stands
# open with nothing touching it, which is when the seasoning errand happens.
MOVING_DRAWER = frozenset({"pull the drawer open", "push the drawer shut"})
# The phases squared up to the handle; everything else works off PICK_ROTATION.
SQUARE_TO_HANDLE = frozenset({"standoff", "engage", "carry"})
# The phase after which the shaker is back in the drawer.
SHAKER_RELEASED = "let go of the shaker"


# Every commanded motion is eased to its goal in the first EASE of its phase,
# and the rest of the phase is spent standing on it. The reach loop is a
# proportional one, so it trails a moving target by about a finger's width;
# without that settling window each phase would end short of where it says it
# is and the next one would start from the wrong place.
EASE = 0.72


def smoothstep(t: float) -> float:
    t = min(max(t / EASE, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def tip_at(phase: Phase, blend: float) -> float:
    """How far from upright the shaker is being commanded, in radians.

    Plain data, so the mock lane can tell whether the routine actually turns
    the shaker cap-down over the food without a physics engine to ask.
    """
    if phase.mode == "tilt":
        first, last = phase.tilt
        return first + (last - first) * blend
    if phase.mode == "shake":
        wave = math.sin(2.0 * math.pi * SHAKE_HZ * blend * phase.seconds)
        return TIP_ANGLE + SHAKE_ROLL * wave
    return 0.0  # every other phase carries the shaker upright, or not at all


def phase_at(elapsed: float) -> tuple[int, Phase, float]:
    """Which phase is in force `elapsed` seconds in, and how far through it.

    Returns `(index, phase, blend)` where `blend` is the eased progress through
    that phase — except for the shake, which is a wave and is handed its raw
    progress. Past the end of the routine this reports the final phase, fully
    blended, so a caller that overruns simply holds.
    """
    remaining = float(elapsed)
    for index, phase in enumerate(PHASES):
        if remaining < phase.seconds:
            raw = remaining / phase.seconds
            return index, phase, (raw if phase.mode == "shake" else smoothstep(raw))
        remaining -= phase.seconds
    return len(PHASES) - 1, PHASES[-1], 1.0
