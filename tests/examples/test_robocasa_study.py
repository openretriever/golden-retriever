"""Simulator-free tests for the RoboCasa policy-study lane.

Everything here runs without RoboCasa installed: the environment adapter defers
its simulator imports, so its configuration, representation handling and scoring
are testable on their own. The parts that need physics are exercised by the
`study-pnp-baselines` pixi task, which is where real episodes belong.

The properties pinned are the ones that would quietly corrupt a result rather
than break a run — a score credited for progress that did not happen, a clipped
action applied without being recorded, privileged state reaching a policy, or a
representation approximated under a label it does not match.
"""

from __future__ import annotations

import numpy as np
import pytest

from examples.advanced.robocasa.study_methods import RandomMethod, StillMethod
from examples.advanced.robocasa.study_observation import policy_observation, tile_frames
from examples.advanced.robocasa.study_progress import (
    LIFT_HEIGHT, POINTS_PER_RUNG, PROGRESS_RUNGS, REACH_RADIUS, ProgressTracker)
from examples.advanced.robocasa.study_representations import (
    MAX_ROTATION_PER_STEP, MAX_TRANSLATION_PER_STEP, REPRESENTATION_DOF, EefDeltaCodec,
    UnsupportedRepresentation, normalised_dof)
from examples.advanced.robocasa.study_robocasa_env import (
    CAMERA_ROLES, ObservationConfig, RoboCasaPnPEnvironment)


def _frame(value=0, size=8):
    return np.full((size, size, 3), value, dtype=np.uint8)


# --- representations -----------------------------------------------------


def test_every_named_representation_has_a_width():
    from examples.advanced.robocasa.study_representations import REPRESENTATIONS

    assert set(REPRESENTATIONS) == set(REPRESENTATION_DOF)
    assert all(width > 0 for width in REPRESENTATION_DOF.values())


def test_an_unknown_representation_names_the_available_ones():
    with pytest.raises(UnsupportedRepresentation, match="available: eef_delta"):
        normalised_dof("b_spline")


def test_a_joint_representation_is_not_an_end_effector_codec():
    """RoboCasa's operational-space controller cannot express it, and
    approximating it would report it under a label it does not match."""
    with pytest.raises(UnsupportedRepresentation):
        EefDeltaCodec("joint_velocity")


def test_the_translation_cap_is_applied_and_recorded():
    codec = EefDeltaCodec("eef_delta")
    row = codec.to_delta(np.array([1.0, 0, 0, 0, 0, 0, -1.0], dtype=np.float32))
    assert float(np.linalg.norm(row[:3])) == pytest.approx(MAX_TRANSLATION_PER_STEP)
    ledger = codec.ledger.to_dict()
    assert ledger["clipped_rows"] == 1 and ledger["clipped_fraction"] == 1.0
    assert ledger["max_translation_excess_m"] == pytest.approx(1.0 - MAX_TRANSLATION_PER_STEP)


def test_the_rotation_cap_is_applied_and_recorded():
    codec = EefDeltaCodec("eef_delta")
    row = codec.to_delta(np.array([0, 0, 0, 1.0, 0, 0, -1.0], dtype=np.float32))
    assert float(np.linalg.norm(row[3:6])) == pytest.approx(MAX_ROTATION_PER_STEP)
    assert codec.ledger.to_dict()["max_rotation_excess_rad"] > 0


def test_an_action_inside_the_cap_is_not_recorded_as_clipped():
    """Otherwise the ledger could not distinguish a policy that saturates from
    one that never approaches the limit."""
    codec = EefDeltaCodec("eef_delta")
    codec.to_delta(np.array([0.001, 0, 0, 0, 0, 0, -1.0], dtype=np.float32))
    assert codec.ledger.to_dict()["clipped_rows"] == 0


def test_an_absolute_representation_needs_the_current_pose():
    codec = EefDeltaCodec("eef_absolute")
    with pytest.raises(ValueError, match="current pose"):
        codec.to_delta(np.zeros(7, dtype=np.float32))


def test_an_absolute_target_becomes_a_capped_delta_toward_it():
    codec = EefDeltaCodec("eef_absolute")
    row = codec.to_delta(np.array([0.5, 0, 0, 0, 0, 0, -1.0], dtype=np.float32),
                         current_position=np.zeros(3), current_rotation=np.zeros(3))
    assert row[0] == pytest.approx(MAX_TRANSLATION_PER_STEP)


def test_every_codec_emits_one_shape_so_the_environment_never_branches():
    for representation in ("eef_delta", "eef_delta_base"):
        codec = EefDeltaCodec(representation)
        row = codec.to_delta(np.zeros(REPRESENTATION_DOF[representation], dtype=np.float32))
        assert row.shape == (10,)


def test_the_base_segment_passes_through_only_for_the_mobile_representation():
    mobile = EefDeltaCodec("eef_delta_base").to_delta(
        np.array([0, 0, 0, 0, 0, 0, -1, 0.4, -0.2, 0.1], dtype=np.float32))
    assert mobile[7:10] == pytest.approx([0.4, -0.2, 0.1])
    arm_only = EefDeltaCodec("eef_delta").to_delta(np.zeros(7, dtype=np.float32))
    assert arm_only[7:10] == pytest.approx([0.0, 0.0, 0.0])


@pytest.mark.parametrize("row", (
    np.zeros(6, dtype=np.float32), np.zeros(8, dtype=np.float32),
    np.array([np.nan, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    np.array([np.inf, 0, 0, 0, 0, 0, 0], dtype=np.float32),
))
def test_an_unusable_action_row_is_rejected(row):
    with pytest.raises(ValueError):
        EefDeltaCodec("eef_delta").to_delta(row)


# --- the partial score ---------------------------------------------------


def test_no_progress_scores_zero():
    assert ProgressTracker().score == 0


def test_the_ladder_is_monotone_so_a_rung_cannot_be_skipped():
    """A lift reported without a grasp is a trace that never grasped, not
    partial credit for lifting."""
    tracker = ProgressTracker()
    tracker.observe(gripper_position=[9, 9, 9], object_position=[0, 0, 0],
                    grasped=False, success=False)
    tracker.initial_object_height = 0.0
    tracker.observe(gripper_position=[9, 9, 9], object_position=[0, 0, 1.0],
                    grasped=False, success=False)
    assert tracker.score == 0
    assert not any(tracker.rungs.values())


def test_reaching_scores_one_rung():
    tracker = ProgressTracker()
    tracker.observe(gripper_position=[0, 0, REACH_RADIUS / 2], object_position=[0, 0, 0],
                    grasped=False, success=False)
    assert tracker.rungs["reached"] and tracker.score == POINTS_PER_RUNG


def test_a_grasp_implies_a_reach_even_from_a_distance_reading():
    tracker = ProgressTracker()
    tracker.observe(gripper_position=[5, 5, 5], object_position=[0, 0, 0],
                    grasped=True, success=False)
    assert tracker.rungs["reached"] and tracker.rungs["grasped"]
    assert tracker.score == 2 * POINTS_PER_RUNG


def test_a_lift_requires_a_grasp_and_a_real_rise():
    tracker = ProgressTracker()
    tracker.observe(gripper_position=[0, 0, 0], object_position=[0, 0, 0],
                    grasped=True, success=False)
    tracker.observe(gripper_position=[0, 0, 0], object_position=[0, 0, LIFT_HEIGHT / 2],
                    grasped=True, success=False)
    assert not tracker.rungs["lifted"]
    tracker.observe(gripper_position=[0, 0, 0], object_position=[0, 0, LIFT_HEIGHT * 1.5],
                    grasped=True, success=False)
    assert tracker.rungs["lifted"] and tracker.score == 3 * POINTS_PER_RUNG


def test_the_top_rung_is_the_simulator_predicate_and_completes_the_score():
    tracker = ProgressTracker()
    tracker.observe(gripper_position=[0, 0, 0], object_position=[0, 0, 0],
                    grasped=True, success=False)
    tracker.observe(gripper_position=[0, 0, 0], object_position=[0, 0, LIFT_HEIGHT * 2],
                    grasped=True, success=True)
    assert tracker.score == 100
    assert all(tracker.rungs.values())


def test_a_rung_latches_so_a_dropped_object_keeps_its_credit():
    """An episode that grasped and then dropped has still demonstrably
    grasped; a score that fell back would report less than happened."""
    tracker = ProgressTracker()
    tracker.observe(gripper_position=[0, 0, 0], object_position=[0, 0, 0],
                    grasped=True, success=False)
    tracker.observe(gripper_position=[9, 9, 9], object_position=[0, 0, 0],
                    grasped=False, success=False)
    assert tracker.rungs["grasped"]


def test_the_score_scale_matches_the_study():
    assert POINTS_PER_RUNG * len(PROGRESS_RUNGS) == 100


# --- observations --------------------------------------------------------


def test_a_single_view_with_no_history_is_the_frame_itself():
    assert tile_frames([[_frame(3)]]).shape == (8, 8, 3)


def test_views_tile_across_and_history_tiles_down():
    tiled = tile_frames([[_frame(1), _frame(2)], [_frame(3), _frame(4)]])
    assert tiled.shape == (16, 16, 3)
    # Oldest row first, so a policy reads the frames in time order.
    assert tiled[0, 0, 0] == 1 and tiled[8, 0, 0] == 3


@pytest.mark.parametrize("frames", ([], [[]], [[_frame()], [_frame(), _frame()]]))
def test_an_unusable_frame_set_is_rejected(frames):
    with pytest.raises(ValueError):
        tile_frames(frames)


def test_mismatched_view_shapes_are_rejected():
    with pytest.raises(ValueError, match="share one shape"):
        tile_frames([[_frame(size=8), _frame(size=16)]])


def test_an_observation_needs_an_instruction():
    with pytest.raises(ValueError, match="task instruction"):
        policy_observation(frames=[[_frame()]], state=np.zeros(4), prompt="  ")


@pytest.mark.parametrize("state", (np.zeros(0), np.zeros((2, 2)),
                                   np.array([np.nan]), np.array([np.inf])))
def test_an_unusable_state_is_rejected(state):
    with pytest.raises(ValueError):
        policy_observation(frames=[[_frame()]], state=state, prompt="do it")


def test_an_observation_carries_only_what_a_policy_may_see():
    """The scorer reads the object's pose; the policy must not. The Lift demo
    in this folder does put privileged state in its observation, which is fine
    for a heuristic and would invalidate a policy study."""
    observation = policy_observation(frames=[[_frame()]], state=np.arange(11, dtype=np.float32),
                                     prompt="Pick the cube and place it in the cabinet")
    assert set(vars(observation)) == {"image", "state", "prompt"}
    assert observation.state.shape == (11,)


# --- environment configuration (no simulator needed) --------------------


def test_the_camera_roles_mirror_robolabs_registration():
    assert set(CAMERA_ROLES) == {"over_shoulder", "wrist"}
    assert CAMERA_ROLES["wrist"] == "robot0_eye_in_hand"


@pytest.mark.parametrize("kwargs", (
    {"views": ()}, {"views": ("nose_cam",)}, {"views": ("wrist", "wrist")},
    {"history": -1}, {"history": 9}, {"history": 1.0},
    {"image_size": 16}, {"image_size": 1024}, {"image_size": 64.0},
))
def test_an_unusable_observation_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ObservationConfig(**kwargs)


def test_the_observation_config_names_real_cameras():
    config = ObservationConfig(views=("over_shoulder", "wrist"), history=2)
    assert config.camera_names == ["robot0_agentview_left", "robot0_eye_in_hand"]
    assert config.to_dict()["history"] == 2


def test_a_representation_the_backend_cannot_express_is_refused_by_name():
    with pytest.raises(UnsupportedRepresentation, match="RoboLab backend"):
        RoboCasaPnPEnvironment(representation="joint_velocity")


@pytest.mark.parametrize("layout,style", ((0, 1), (1, 0), (-1, 1)))
def test_a_zero_based_layout_is_rejected_before_the_scene_registry_sees_it(layout, style):
    """`layout_ids=0` raises a `KeyError` deep inside RoboCasa, because the
    concrete layouts start at 1 and the negatives are group selectors."""
    with pytest.raises(ValueError):
        RoboCasaPnPEnvironment(layout_id=layout, style_id=style)


def test_the_backend_describes_what_a_run_should_record():
    description = RoboCasaPnPEnvironment(command_base=True).describe()
    assert description["backend"] == "robocasa"
    assert description["command_base"] is True
    assert description["frozen_segments"] == ["torso"]
    assert description["observation"]["views"] == ["over_shoulder"]


def test_freezing_the_base_is_recorded_as_a_frozen_segment():
    """A run that froze the base was measuring an arm against a task that needs
    a mobile manipulator, so the evidence has to say so."""
    assert RoboCasaPnPEnvironment().describe()["frozen_segments"] == ["base", "torso"]


# --- the baselines -------------------------------------------------------


def test_the_still_method_commands_no_motion_and_an_open_gripper():
    chunk = StillMethod(horizon=4).predict(None)
    assert chunk.actions.shape == (4, 7) and chunk.horizon == 4
    assert np.all(chunk.actions[:, :6] == 0.0)
    assert np.all(chunk.actions[:, 6] == -1.0)
    assert chunk.source == "scripted"


def test_the_random_method_replays_from_its_seed():
    first = RandomMethod(horizon=3, seed=11).predict(None).actions
    second = RandomMethod(horizon=3, seed=11).predict(None).actions
    assert np.array_equal(first, second)
    assert not np.array_equal(first, RandomMethod(horizon=3, seed=12).predict(None).actions)


def test_the_random_method_stays_inside_the_safety_envelope():
    from examples.advanced.robocasa.method_harness import SafetyEnvelope

    chunk = RandomMethod(horizon=6, seed=2).predict(None)
    assert SafetyEnvelope().validate(chunk).actions.shape == (6, 7)


def test_a_method_counts_its_own_calls():
    """One call per dispatched chunk is the study's model-call metric."""
    method = StillMethod(horizon=2)
    for _ in range(3):
        method.predict(None)
    assert method.calls == 3


def test_a_mobile_method_emits_the_wider_row():
    chunk = RandomMethod(horizon=2, seed=1, representation="eef_delta_base").predict(None)
    assert chunk.actions.shape == (2, 10) and chunk.dof == 10


@pytest.mark.parametrize("scale", (0.0, -0.1, 1.5))
def test_an_unusable_random_scale_is_rejected(scale):
    with pytest.raises(ValueError):
        RandomMethod(scale=scale)


# --- the RoboLab port ----------------------------------------------------


def test_the_robolab_backend_satisfies_the_same_protocol_as_robocasa():
    """The port is an adapter swap only if both satisfy one contract."""
    from examples.advanced.robocasa.study_robolab_env import RoboLabEnvironment

    for name in ("reset", "step", "verify", "close", "describe"):
        assert callable(getattr(RoboLabEnvironment, name))
        assert callable(getattr(RoboCasaPnPEnvironment, name))


def test_a_missing_robolab_stack_says_which_requirement_is_unmet():
    """Not a bare ImportError three frames deep: the reason this cannot run
    here is a hardware and OS requirement, and the message says so."""
    from examples.advanced.robocasa.study_robolab_env import RoboLabUnavailable, require_robolab

    with pytest.raises(RoboLabUnavailable, match="RTX GPU"):
        require_robolab()
    with pytest.raises(RoboLabUnavailable, match="Apple Silicon"):
        require_robolab()


def test_importing_the_robolab_backend_needs_no_simulator():
    """It is imported by the sweep on every platform, so its module body must
    not touch Isaac."""
    import examples.advanced.robocasa.study_robolab_env as module

    assert module.EXPOSED_DOF == 8


def test_the_robolab_backend_declares_a_joint_action_space():
    """DROID is fixed-base and joint-space; RoboCasa is mobile and
    operational-space. The backends sit at two points on the study's
    action-representation axis, which is the port's whole value."""
    from examples.advanced.robocasa.study_robolab_env import (
        SUPPORTED_REPRESENTATIONS, RoboLabEnvironment)

    assert SUPPORTED_REPRESENTATIONS == ("joint_velocity", "joint_position")
    description = RoboLabEnvironment().describe()
    assert description["fixed_base"] is True and description["exposed_dof"] == 8
    assert description["verified_against_live_instance"] is False


def test_an_end_effector_representation_is_refused_by_the_robolab_backend():
    from examples.advanced.robocasa.study_robolab_env import RoboLabEnvironment

    with pytest.raises(UnsupportedRepresentation, match="RoboCasa backend"):
        RoboLabEnvironment(representation="eef_delta")


def test_the_robolab_camera_roles_match_droids_registration():
    from examples.advanced.robocasa.study_robolab_env import CAMERA_ROLES as ROBOLAB_ROLES

    assert ROBOLAB_ROLES["over_shoulder"] == "over_shoulder_left_camera"
    assert ROBOLAB_ROLES["wrist"] == "wrist_cam"
    # The same role names as the RoboCasa backend, which is what lets one
    # observation config describe either.
    assert set(ROBOLAB_ROLES) == set(CAMERA_ROLES)


# --- the sweep -----------------------------------------------------------


class _FakeEnvironment:
    """Stands in for a backend so the sweep is testable without physics."""

    def __init__(self, task="T", representation="eef_delta", *, succeed_at=None):
        from examples.advanced.robocasa.study_progress import ProgressTracker
        from examples.advanced.robocasa.study_representations import EefDeltaCodec

        self.task = task
        self.representation = representation
        self.progress = ProgressTracker()
        self.codec = EefDeltaCodec(representation)
        self.steps = 0
        self.succeed_at = succeed_at

    def _observation(self):
        return policy_observation(frames=[[_frame()]], state=np.zeros(3), prompt="go")

    def reset(self, request):
        self.steps = 0
        return self._observation()

    def step(self, action):
        from examples.advanced.robocasa.method_harness import Transition

        self.steps += 1
        self.codec.to_delta(np.asarray(action, dtype=np.float32))
        done = self.succeed_at is not None and self.steps >= self.succeed_at
        if done:
            self.progress.observe(gripper_position=[0, 0, 0], object_position=[0, 0, 0],
                                  grasped=True, success=True)
            self.progress.observe(gripper_position=[0, 0, 0], object_position=[0, 0, 1.0],
                                  grasped=True, success=True)
        return Transition(observation=self._observation(), reward=0.0, terminated=done)

    def verify(self):
        from examples.advanced.robocasa.method_harness import Verification

        return Verification(success=self.progress.succeeded)

    def close(self):
        pass

    def describe(self):
        return {"backend": "fake", "task": self.task}


def _sweep(**kwargs):
    from examples.advanced.robocasa.study_sweep import Sweep

    return Sweep(build_environment=lambda task, representation: _FakeEnvironment(
        task=task, representation=representation, **kwargs), **{
            "tasks": ("T1",), "seeds": (0, 1)})


def test_a_sweep_runs_every_cell_and_summarises_per_method():
    from examples.advanced.robocasa.study_sweep import MethodSpec

    specs = (MethodSpec(method_id="still", horizon=4,
                        build=lambda env: StillMethod(horizon=4)),
             MethodSpec(method_id="random", horizon=4,
                        build=lambda env: RandomMethod(horizon=4, seed=0)))
    summary = _sweep().run(specs)
    assert summary["protocol"]["episodes"] == 2
    assert len(summary["episodes"]) == 4
    assert set(summary["summary"]) == {"still", "random"}
    assert summary["errors"] == 0


def test_the_sweep_counts_model_calls_from_the_harness_not_the_method():
    """A method cannot misreport its own call count if the count comes from the
    harness's dispatch events."""
    from examples.advanced.robocasa.study_sweep import MethodSpec

    summary = _sweep().run((MethodSpec(method_id="m", horizon=5,
                                       build=lambda env: StillMethod(horizon=5)),))
    row = summary["episodes"][0]
    assert row["model_calls"] * 5 >= row["control_steps"] > 0
    assert row["horizon"] == 5


def test_control_steps_and_model_calls_are_separate_counts():
    from examples.advanced.robocasa.study_sweep import MethodSpec

    narrow = _sweep().run((MethodSpec(method_id="h1", horizon=1,
                                      build=lambda env: StillMethod(horizon=1)),))
    wide = _sweep().run((MethodSpec(method_id="h8", horizon=8,
                                    build=lambda env: StillMethod(horizon=8)),))
    # Same control steps, fewer calls: that is what a horizon buys, and why
    # the two must never be summed or substituted.
    assert narrow["summary"]["h1"]["control_steps"] == wide["summary"]["h8"]["control_steps"]
    assert wide["summary"]["h8"]["model_calls"] < narrow["summary"]["h1"]["model_calls"]


def test_a_fresh_method_is_built_for_every_trial():
    """Reusing one would leak an episode's phase or generator into the next."""
    from examples.advanced.robocasa.study_sweep import MethodSpec

    built = []
    specs = (MethodSpec(method_id="m", horizon=2,
                        build=lambda env: built.append(1) or StillMethod(horizon=2)),)
    _sweep().run(specs)
    assert len(built) == 2


def test_a_successful_cell_is_scored_and_reported():
    from examples.advanced.robocasa.study_sweep import MethodSpec

    summary = _sweep(succeed_at=3).run((MethodSpec(method_id="m", horizon=1,
                                                   build=lambda env: StillMethod(horizon=1)),))
    assert summary["summary"]["m"]["success_rate"] == 1.0
    assert summary["summary"]["m"]["mean_score"] == 100.0


def test_a_sweep_rejects_duplicate_methods_and_an_empty_set():
    from examples.advanced.robocasa.study_sweep import MethodSpec

    spec = MethodSpec(method_id="same", build=lambda env: StillMethod())
    with pytest.raises(ValueError, match="duplicate method_id"):
        _sweep().run((spec, spec))
    with pytest.raises(ValueError, match="at least one method"):
        _sweep().run(())


def test_a_method_spec_needs_a_factory_not_an_instance():
    from examples.advanced.robocasa.study_sweep import MethodSpec

    with pytest.raises(TypeError, match="callable"):
        MethodSpec(method_id="m", build=StillMethod())


def test_the_rendered_summary_is_sorted_json():
    from examples.advanced.robocasa.study_sweep import MethodSpec, render

    summary = _sweep().run((MethodSpec(method_id="m", horizon=2,
                                       build=lambda env: StillMethod(horizon=2)),))
    text = render(summary)
    assert json_loads_keys_sorted(text)
    assert "episodes" not in json_loads(render(summary, summary_only=True))


def json_loads(text):
    import json

    return json.loads(text)


def json_loads_keys_sorted(text):
    import json

    data = json.loads(text)
    return list(data) == sorted(data)


def test_the_default_seeds_are_the_studys_protocol():
    from examples.advanced.robocasa.study_sweep import DEFAULT_SEEDS

    assert DEFAULT_SEEDS == (0, 1, 2, 3, 4)


# --- the committed task suite -------------------------------------------


def test_the_suite_accounts_for_every_registered_environment():
    """38 registered `PickPlace*`/`Place*` environments, each in exactly one
    bucket, so nothing was dropped silently."""
    from examples.advanced.robocasa.study_tasks import (
        MISSING_ASSET_TASKS, MULTI_OBJECT_TASKS, NON_KITCHEN_TASKS, PNP_TASKS)

    buckets = (tuple(task.task for task in PNP_TASKS), MULTI_OBJECT_TASKS,
               NON_KITCHEN_TASKS, MISSING_ASSET_TASKS)
    names = [name for bucket in buckets for name in bucket]
    assert len(names) == 38
    assert len(set(names)) == 38, "a task appears in two buckets"
    assert len(PNP_TASKS) == 18


def test_every_committed_task_carries_its_measured_reach_and_goal():
    from examples.advanced.robocasa.study_tasks import PNP_TASKS

    for task in PNP_TASKS:
        assert task.goal.strip() and task.goal.endswith(".")
        assert 0.3 <= task.initial_reach_m <= 0.9


def test_no_committed_task_is_arm_reachable():
    """The measurement that bounds the whole lane: a capped Cartesian
    controller stalls near 0.29 m and the nearest object starts at 0.305 m, so
    a policy that does not command the base cannot engage any task."""
    from examples.advanced.robocasa.study_tasks import ARM_STALL_DISTANCE_M, PNP_TASKS

    assert not any(task.arm_reachable for task in PNP_TASKS)
    assert min(task.initial_reach_m for task in PNP_TASKS) > ARM_STALL_DISTANCE_M


def test_an_unscorable_task_explains_itself_rather_than_404ing():
    from examples.advanced.robocasa.study_tasks import get_task

    with pytest.raises(ValueError, match="no single `obj`"):
        get_task("PlaceVegetablesEvenly")
    with pytest.raises(ValueError, match="bin-picking"):
        get_task("PickPlaceCan")
    with pytest.raises(ValueError, match="objs_objaverse"):
        get_task("PickPlaceCounterToOven")
    with pytest.raises(ValueError, match="available:"):
        get_task("NoSuchTask")


def test_the_ablation_subset_spreads_across_the_reach_range():
    """A subset that was accidentally all-near or all-far would measure
    something other than the suite, since reach decides whether a policy can
    engage the object at all."""
    from examples.advanced.robocasa.study_tasks import ablation_subset, get_task

    subset = ablation_subset(6)
    assert len(subset) == len(set(subset)) == 6
    reaches = [get_task(name).initial_reach_m for name in subset]
    assert max(reaches) - min(reaches) > 0.3


@pytest.mark.parametrize("count", (0, -1, 19, 1.0, True))
def test_an_unusable_subset_size_is_rejected(count):
    from examples.advanced.robocasa.study_tasks import ablation_subset

    with pytest.raises(ValueError):
        ablation_subset(count)


def test_the_layout_is_pinned_and_one_based():
    from examples.advanced.robocasa.study_tasks import LAYOUT_ID, STYLE_ID

    assert LAYOUT_ID >= 1 and STYLE_ID >= 1


def test_the_seeds_are_the_studys_protocol():
    from examples.advanced.robocasa.study_tasks import DEFAULT_SEEDS

    assert DEFAULT_SEEDS == (0, 1, 2, 3, 4)


@pytest.mark.parametrize("bad", (
    {"task": "NotATask", "goal": "g.", "initial_reach_m": 0.5},
    {"task": "PickPlaceX", "goal": "   ", "initial_reach_m": 0.5},
    {"task": "PickPlaceX", "goal": "g.", "initial_reach_m": 0.0},
    {"task": "PickPlaceX", "goal": "g.", "initial_reach_m": 5.0},
))
def test_an_unusable_task_entry_is_rejected(bad):
    from examples.advanced.robocasa.study_tasks import PnPTask

    with pytest.raises(ValueError):
        PnPTask(**bad)


# --- telemetry and transcript replay ------------------------------------


def _observation(fill=0, prompt="pick the cube", state=5):
    return policy_observation(frames=[[_frame(fill)]],
                              state=np.arange(state, dtype=np.float32), prompt=prompt)


def _record(**overrides):
    from examples.advanced.robocasa.study_telemetry import CallRecord

    base = {"task": "T", "method_id": "m", "seed": 0, "call_index": 0,
            "prompt_sha256": "a" * 64, "latency_s": 0.5, "parse_ok": True}
    return CallRecord(**{**base, **overrides})


def test_the_prompt_digest_covers_the_image_not_just_the_instruction():
    """An action chunk depends on what the policy saw, so a transcript keyed on
    the instruction alone would return the same answer at every step."""
    from examples.advanced.robocasa.study_transcript import assemble_prompt, prompt_digest

    same = prompt_digest(assemble_prompt(_observation(0)))
    assert same == prompt_digest(assemble_prompt(_observation(0)))
    assert same != prompt_digest(assemble_prompt(_observation(1)))
    assert same != prompt_digest(assemble_prompt(_observation(0, prompt="other"))) 
    assert same != prompt_digest(assemble_prompt(_observation(0, state=6)))


def test_the_image_is_hashed_not_embedded():
    """A 128x128x3 frame is 49 kB; a transcript carrying them would be
    unusable, and a digest over the bytes is exact."""
    from examples.advanced.robocasa.study_transcript import assemble_prompt

    prompt = assemble_prompt(policy_observation(
        frames=[[_frame(size=128)]], state=np.zeros(5), prompt="go"))
    assert len(prompt) < 400
    assert "image_sha256" in prompt


def test_a_configuration_changes_the_digest_so_two_cannot_collide():
    from examples.advanced.robocasa.study_transcript import assemble_prompt, prompt_digest

    plain = prompt_digest(assemble_prompt(_observation()))
    configured = prompt_digest(assemble_prompt(_observation(), config={"variant": "terse"}))
    assert plain != configured


def test_a_recorded_call_replays_exactly():
    from examples.advanced.robocasa.study_transcript import TranscriptMethod, TranscriptRecorder

    recorder = TranscriptRecorder(method=StillMethod(horizon=3))
    original = recorder.predict(_observation())
    replayed = TranscriptMethod(store=recorder.store, dof=7).predict(_observation())
    assert np.allclose(replayed.actions, original.actions)
    assert replayed.horizon == original.horizon == 3


def test_a_recorder_stores_only_what_it_knows():
    """Inventing token or cost figures would fabricate the numbers a study
    measures, and a measured latency would make the file differ every time."""
    from examples.advanced.robocasa.study_transcript import TranscriptRecorder

    recorder = TranscriptRecorder(method=StillMethod(horizon=2))
    recorder.predict(_observation())
    entry = next(iter(recorder.store.entries.values()))
    assert set(entry) == {"response", "model_id"}


def test_a_recorder_can_attach_a_providers_reported_cost():
    from examples.advanced.robocasa.study_transcript import TranscriptRecorder

    recorder = TranscriptRecorder(
        method=StillMethod(horizon=2), model_id="fake:model",
        report_cost=lambda: {"prompt_tokens": 90, "usd": 0.003, "latency_s": 0.4})
    recorder.predict(_observation())
    entry = next(iter(recorder.store.entries.values()))
    assert entry["usd"] == 0.003 and entry["model_id"] == "fake:model"
    assert recorder.ledger.telemetry[0]["recorded_latency_s"] == 0.4


def test_a_transcript_miss_is_refused_and_measured():
    """No fallback: answering from a scripted policy would mix scripted
    behaviour into a replayed arm's results."""
    from examples.advanced.robocasa.study_transcript import (
        InvalidPolicyOutput, TranscriptMethod, TranscriptRecorder)

    recorder = TranscriptRecorder(method=StillMethod(horizon=2))
    recorder.predict(_observation(0))
    method = TranscriptMethod(store=recorder.store, dof=7)
    with pytest.raises(InvalidPolicyOutput, match="no recorded call"):
        method.predict(_observation(1))
    row = method.ledger.telemetry[-1]
    assert row["transcript_hit"] is False and row["parse_ok"] is False


@pytest.mark.parametrize("text,expected", (
    ("", "empty"), ("   ", "empty"), (None, "empty"),
    ("no json here", "no JSON array"),
    ("[]", "non-empty list"),
    ("[[1,2]]", "must each have 7"),
    ("[[0,0,0,0,0,0,0],[0,0,0]]", "must each have 7"),
    ("[[0,0,0,0,0,0,9]]", "normalised action bound"),
    ('[[0,0,0,0,0,0,"x"]]', "non-numeric"),
    ("[[0,0,0,0,0,0,null]]", "non-finite"),
))
def test_an_unusable_chunk_response_is_refused(text, expected):
    from examples.advanced.robocasa.study_transcript import (
        InvalidPolicyOutput, parse_chunk_response)

    with pytest.raises(InvalidPolicyOutput, match=expected):
        parse_chunk_response(text, dof=7)


def test_a_packaged_response_still_parses():
    from examples.advanced.robocasa.study_transcript import parse_chunk_response

    fenced = "Here you go:\n```json\n[[0.1,0,0,0,0,0,-1]]\n```\nDone."
    assert parse_chunk_response(fenced, dof=7).shape == (1, 7)


def test_an_overlong_chunk_is_refused_before_the_envelope_sees_it():
    from examples.advanced.robocasa.study_transcript import (
        MAX_HORIZON, InvalidPolicyOutput, parse_chunk_response)

    rows = json_loads(f"[{','.join(['[0,0,0,0,0,0,0]'] * (MAX_HORIZON + 1))}]")
    import json as _json

    with pytest.raises(InvalidPolicyOutput, match="exceeds the bound"):
        parse_chunk_response(_json.dumps(rows), dof=7)


def test_the_store_rejects_a_malformed_document(tmp_path):
    from examples.advanced.robocasa.study_transcript import TranscriptStore

    with pytest.raises(ValueError, match="not a sha256 digest"):
        TranscriptStore({"short": {"response": "[]"}})
    with pytest.raises(ValueError, match="no response"):
        TranscriptStore({"a" * 64: {"nope": 1}})
    with pytest.raises(FileNotFoundError):
        TranscriptStore.from_path(tmp_path / "absent.json")
    bad = tmp_path / "bad.json"
    bad.write_text('{"schema_version": 99, "calls": {}}')
    with pytest.raises(ValueError, match="unsupported transcript schema"):
        TranscriptStore.from_path(bad)


def test_a_transcript_round_trips_through_a_file(tmp_path):
    from examples.advanced.robocasa.study_transcript import TranscriptRecorder, TranscriptStore

    recorder = TranscriptRecorder(method=StillMethod(horizon=2))
    recorder.predict(_observation())
    path = recorder.store.write(tmp_path / "t.json")
    loaded = TranscriptStore.from_path(path)
    assert len(loaded) == 1 and len(loaded.source_sha256) == 64


def test_an_unset_transcript_path_fails_rather_than_degrading(monkeypatch):
    from examples.advanced.robocasa.study_transcript import (
        TRANSCRIPT_ENV, EnvTranscriptMethod)

    monkeypatch.delenv(TRANSCRIPT_ENV, raising=False)
    with pytest.raises(ValueError, match=TRANSCRIPT_ENV):
        EnvTranscriptMethod()


def test_the_ledger_keeps_the_deterministic_and_measured_halves_apart():
    """A caller that mixed them would produce a record that can never be
    compared across runs."""
    from examples.advanced.robocasa.study_telemetry import NONDETERMINISTIC_FIELDS
    from examples.advanced.robocasa.study_transcript import TranscriptRecorder

    recorder = TranscriptRecorder(method=StillMethod(horizon=2))
    recorder.predict(_observation())
    canonical, = recorder.ledger.canonical
    assert set(canonical) == {"call_index", "prompt_sha256", "horizon"}
    assert all(field not in canonical for field in NONDETERMINISTIC_FIELDS)
    assert "latency_s" in recorder.ledger.telemetry[0]


@pytest.mark.parametrize("field,value", (
    ("task", ""), ("method_id", ""),
    ("call_index", -1), ("call_index", 1.0),
    ("prompt_sha256", "short"), ("latency_s", -0.1), ("latency_s", 1),
    ("prompt_tokens", -1), ("usd", -0.01), ("usd", 1), ("horizon", -1),
))
def test_an_unusable_call_record_is_rejected(field, value):
    with pytest.raises(ValueError):
        _record(**{field: value})


def test_the_sidecar_round_trips_and_joins(tmp_path):
    from examples.advanced.robocasa.study_telemetry import (
        SIDECAR_NAME, read_sidecar, verify_join, write_sidecar)

    write_sidecar(tmp_path, [_record(call_index=0), _record(call_index=1)])
    assert (tmp_path / SIDECAR_NAME).is_file()
    rows = read_sidecar(tmp_path)
    ledger = [{"call_index": i, "prompt_sha256": "a" * 64} for i in (0, 1)]
    assert verify_join(ledger, rows)["joined"]


def test_a_digest_mismatch_breaks_the_join():
    """So a sidecar cannot be silently attached to another run's results."""
    from examples.advanced.robocasa.study_telemetry import verify_join

    verdict = verify_join([{"call_index": 0, "prompt_sha256": "b" * 64}],
                          [_record().to_dict()])
    assert not verdict["joined"]
    assert "prompt digest disagrees" in " ".join(verdict["errors"])


def test_a_gap_in_call_indices_breaks_the_join():
    from examples.advanced.robocasa.study_telemetry import verify_join

    rows = [_record(call_index=0).to_dict(), _record(call_index=2).to_dict()]
    ledger = [{"call_index": 0, "prompt_sha256": "a" * 64},
              {"call_index": 2, "prompt_sha256": "a" * 64}]
    assert "complete ordered sequence" in " ".join(verify_join(ledger, rows)["errors"])


def test_an_absent_sidecar_reads_as_empty(tmp_path):
    from examples.advanced.robocasa.study_telemetry import read_sidecar

    assert read_sidecar(tmp_path) == []


def test_absent_cost_is_reported_absent_rather_than_zero():
    """Summing a missing count as zero quietly understates the cost a study
    exists to measure."""
    from examples.advanced.robocasa.study_telemetry import projected_cost

    summary = projected_cost([_record().to_dict()])
    assert summary["prompt_tokens"] is None
    assert summary["prompt_tokens_reported_by"] == 0


def test_reported_cost_is_totalled_with_its_denominator():
    from examples.advanced.robocasa.study_telemetry import projected_cost

    rows = [_record(call_index=0, prompt_tokens=10, usd=0.002).to_dict(),
            _record(call_index=1).to_dict()]
    summary = projected_cost(rows)
    assert summary["prompt_tokens"] == 10 and summary["prompt_tokens_reported_by"] == 1
    assert summary["usd"] == 0.002 and summary["calls"] == 2


def test_the_two_latencies_are_separate_fields():
    """A replayed call's measured latency is a dictionary lookup; reporting it
    as model latency would understate it by orders of magnitude."""
    row = _record(latency_s=0.001, recorded_latency_s=2.4).to_dict()
    assert row["latency_s"] != row["recorded_latency_s"]


# --- the model boundary and the three policy forms ----------------------


def _config(**overrides):
    from examples.advanced.robocasa.study_model import ModelConfig

    return ModelConfig(**{"model_id": "test:model", **overrides})


def _answers(*texts):
    from examples.advanced.robocasa.study_model import OfflineModelClient

    return OfflineModelClient(answers=list(texts))


def _direct(client, **kwargs):
    from examples.advanced.robocasa.study_model import DirectModelMethod

    return DirectModelMethod(client=client, config=kwargs.pop("config", _config()),
                             dof=7, **kwargs)


@pytest.mark.parametrize("field,value", (
    ("model_id", ""), ("model_id", "  "), ("model_id", 7),
    ("prompt_variant", "nope"), ("reasoning_effort", "maximum"),
    ("horizon_band", "medium"),
    ("temperature", -0.1), ("temperature", 2.1), ("temperature", 0),
    ("timeout_seconds", 0.0), ("timeout_seconds", 60),
    ("max_response_bytes", 0), ("max_output_tokens", -1),
))
def test_an_unusable_model_config_is_rejected(field, value):
    with pytest.raises(ValueError):
        _config(**{field: value})


def test_the_horizon_bands_are_the_studys():
    from examples.advanced.robocasa.study_model import HORIZON_BANDS

    assert HORIZON_BANDS == {"short": (3, 15), "long": (16, 32)}
    assert _config(horizon_band="long").horizon_range == (16, 32)


def test_the_token_cap_is_the_studys():
    assert _config().max_output_tokens == 8_192


@pytest.mark.parametrize("variant", ("terse", "explained", "with_format"))
def test_every_prompt_variant_renders_with_the_action_width(variant):
    text = _config(prompt_variant=variant).instructions(dof=7)
    assert text.strip() and "{" not in text


@pytest.mark.parametrize("field,value", (
    ("prompt_variant", "terse"), ("model_id", "other:model"),
    ("reasoning_effort", "high"), ("horizon_band", "long"),
))
def test_each_prompt_axis_changes_the_digest(field, value):
    """Wired: the configuration is part of the hashed prompt, so two
    configurations cannot collide on one transcript entry."""
    from dataclasses import replace

    from examples.advanced.robocasa.study_model import assemble_model_prompt
    from examples.advanced.robocasa.study_transcript import prompt_digest

    observation = _observation()
    base = prompt_digest(assemble_model_prompt(observation, _config(), dof=7))
    altered = prompt_digest(assemble_model_prompt(
        observation, replace(_config(), **{field: value}), dof=7))
    assert base != altered


def test_carried_history_changes_the_digest():
    """Which is what makes Direct and Agentic distinguishable requests rather
    than the same call under two names."""
    from examples.advanced.robocasa.study_model import assemble_model_prompt
    from examples.advanced.robocasa.study_transcript import prompt_digest

    observation = _observation()
    without = prompt_digest(assemble_model_prompt(observation, _config(), dof=7))
    with_history = prompt_digest(assemble_model_prompt(
        observation, _config(), dof=7, history=({"call": 0, "horizon": 5},)))
    assert without != with_history


# --- every expected failure becomes a refusal ---------------------------


@pytest.mark.parametrize("status", ("timeout", "oversize", "refused", "transport_error"))
def test_a_non_ok_status_is_a_refusal(status):
    from examples.advanced.robocasa.study_model import ModelCallStatus, ModelResponse
    from examples.advanced.robocasa.study_transcript import InvalidPolicyOutput

    method = _direct(_answers(ModelResponse(status=ModelCallStatus(status), detail="why")))
    with pytest.raises(InvalidPolicyOutput, match=f"ended {status}"):
        method.predict(_observation())
    assert method.telemetry()[0]["parse_ok"] is False


def test_an_unparseable_answer_is_a_refusal():
    from examples.advanced.robocasa.study_transcript import InvalidPolicyOutput

    with pytest.raises(InvalidPolicyOutput):
        _direct(_answers("I would rather not.")).predict(_observation())


def test_a_chunk_outside_the_configured_band_is_a_refusal():
    """The horizon band is part of the configuration, so a model that ignores
    it has produced an answer the run did not ask for."""
    import json

    from examples.advanced.robocasa.study_transcript import InvalidPolicyOutput

    one_row = json.dumps([[0.0] * 7])
    with pytest.raises(InvalidPolicyOutput, match="outside the configured short band"):
        _direct(_answers(one_row)).predict(_observation())


def test_a_client_that_raises_is_a_refusal_naming_the_transport():
    from examples.advanced.robocasa.study_model import OfflineModelClient
    from examples.advanced.robocasa.study_transcript import InvalidPolicyOutput

    class _Exploding(OfflineModelClient):
        def complete(self, prompt, config):
            raise ConnectionResetError("peer hung up")

    with pytest.raises(InvalidPolicyOutput, match="transport_error"):
        _direct(_Exploding()).predict(_observation())


def test_an_oversize_answer_is_a_refusal():
    import json

    from examples.advanced.robocasa.study_transcript import InvalidPolicyOutput

    rows = json.dumps([[0.0] * 7] * 5)
    method = _direct(_answers(rows), config=_config(max_response_bytes=10))
    with pytest.raises(InvalidPolicyOutput, match="exceeds its configured bound"):
        method.predict(_observation())


def test_the_base_protocol_refuses_to_be_used_directly():
    from examples.advanced.robocasa.study_model import ModelClient

    with pytest.raises(NotImplementedError):
        ModelClient().complete("x", _config())


def test_a_non_client_reference_is_rejected():
    from examples.advanced.robocasa.study_model import load_model_client

    with pytest.raises(TypeError, match="complete"):
        load_model_client("examples.advanced.robocasa.study_model:PROMPT_VARIANTS")
    with pytest.raises(ValueError, match="module:symbol"):
        load_model_client("examples.advanced.robocasa.study_model.OfflineModelClient")


# --- the three forms ----------------------------------------------------


def test_all_three_forms_produce_a_chunk_with_no_key():
    from examples.advanced.robocasa.study_model import offline_forms

    forms = offline_forms(lambda: StillMethod(horizon=5))
    assert set(forms) == {"direct", "agentic", "orchestrated"}
    for method in forms.values():
        chunk = method.predict(_observation())
        assert chunk.horizon == 5 and chunk.dof == 7


def test_the_three_forms_agree_against_the_offline_client():
    """They answer from the same wrapped method, so a difference would mean the
    harness alters behaviour rather than the form being measured."""
    from examples.advanced.robocasa.study_model import offline_forms

    forms = offline_forms(lambda: StillMethod(horizon=5))
    chunks = [method.predict(_observation()).actions for method in forms.values()]
    assert all(np.allclose(chunks[0], other) for other in chunks[1:])


def test_the_agentic_form_carries_history_across_calls():
    from examples.advanced.robocasa.study_model import AgenticModelMethod, OfflineModelClient

    method = AgenticModelMethod(
        client=OfflineModelClient(method=StillMethod(horizon=5)), config=_config(),
        dof=7, memory=2)
    for _ in range(3):
        method.predict(_observation())
    # Three calls, and the prompts differ because the carried window grew.
    prompts = method.client.calls
    assert len(prompts) == 3 and len(set(prompts)) == 3


def test_the_agentic_form_at_zero_memory_matches_direct():
    """Informationally identical by construction, so a reported difference
    between them at memory 0 would be noise."""
    from examples.advanced.robocasa.study_model import (
        AgenticModelMethod, DirectModelMethod, OfflineModelClient)

    direct = DirectModelMethod(
        client=OfflineModelClient(method=StillMethod(horizon=5)), config=_config(), dof=7)
    agentic = AgenticModelMethod(
        client=OfflineModelClient(method=StillMethod(horizon=5)), config=_config(),
        dof=7, memory=0)
    direct.predict(_observation())
    agentic.predict(_observation())
    assert direct.client.calls == agentic.client.calls


@pytest.mark.parametrize("memory", (-1, 17, 1.0, None))
def test_an_unusable_memory_depth_is_rejected(memory):
    from examples.advanced.robocasa.study_model import AgenticModelMethod, OfflineModelClient

    with pytest.raises(ValueError):
        AgenticModelMethod(client=OfflineModelClient(), config=_config(), memory=memory)


def test_the_orchestrated_form_records_every_review():
    """The intervention rate has to be readable without re-running anything."""
    from examples.advanced.robocasa.study_model import offline_forms

    method = offline_forms(lambda: StillMethod(horizon=5))["orchestrated"]
    method.predict(_observation())
    review, = method.reviews
    assert review["proposed_steps"] == 5 and review["committed_steps"] == 5
    assert review["intervened"] is False


def test_the_orchestrated_form_falls_back_to_the_proposal_on_a_refusal():
    """A reviewer that cannot answer must not fail the episode: the base
    policy's proposal already exists and is what it was reviewing."""
    from examples.advanced.robocasa.study_model import OfflineModelClient, OrchestratedModelMethod

    method = OrchestratedModelMethod(
        client=OfflineModelClient(answers=["nonsense"]), config=_config(), dof=7,
        base=StillMethod(horizon=5))
    chunk = method.predict(_observation())
    assert chunk.horizon == 5
    review, = method.reviews
    assert review["intervened"] is False and review["reason"]


def test_the_orchestrated_form_records_an_intervention():
    import json

    from examples.advanced.robocasa.study_model import OfflineModelClient, OrchestratedModelMethod

    shorter = json.dumps([[0.0] * 7] * 3)
    method = OrchestratedModelMethod(
        client=OfflineModelClient(answers=[shorter]), config=_config(), dof=7,
        base=StillMethod(horizon=8))
    chunk = method.predict(_observation())
    assert chunk.horizon == 3
    review, = method.reviews
    assert review["proposed_steps"] == 8 and review["committed_steps"] == 3
    assert review["intervened"] is True


def test_the_orchestrated_proposal_length_is_the_studys():
    from examples.advanced.robocasa.study_model import ORCHESTRATED_PROPOSAL_STEPS

    assert ORCHESTRATED_PROPOSAL_STEPS == 15


def test_an_orchestrated_method_needs_a_base():
    from examples.advanced.robocasa.study_model import OfflineModelClient, OrchestratedModelMethod

    with pytest.raises(TypeError, match="base with predict"):
        OrchestratedModelMethod(client=OfflineModelClient(), config=_config())


# --- the ledger -----------------------------------------------------------


def test_the_model_ledger_names_the_configuration_and_nothing_measured():
    from examples.advanced.robocasa.study_model import OfflineModelClient
    from examples.advanced.robocasa.study_telemetry import NONDETERMINISTIC_FIELDS

    method = _direct(OfflineModelClient(method=StillMethod(horizon=5)))
    method.predict(_observation())
    entry, = method.canonical_ledger()
    assert entry["model_config"] == _config().to_dict()
    assert all(field not in entry for field in NONDETERMINISTIC_FIELDS)
    assert "latency_s" in method.telemetry()[0]


def test_a_call_can_be_banked_for_free_replay():
    """So one paid sweep replays offline afterwards."""
    from examples.advanced.robocasa.study_model import OfflineModelClient
    from examples.advanced.robocasa.study_transcript import TranscriptStore

    store = TranscriptStore({})
    method = _direct(OfflineModelClient(method=StillMethod(horizon=5)), record_into=store)
    method.predict(_observation())
    assert len(store) == 1


def test_a_refused_call_is_not_banked():
    """A transcript of answers that do not parse would replay as refusals and
    read as a model failure rather than a recording failure."""
    from examples.advanced.robocasa.study_transcript import InvalidPolicyOutput, TranscriptStore

    store = TranscriptStore({})
    method = _direct(_answers("nonsense"), record_into=store)
    with pytest.raises(InvalidPolicyOutput):
        method.predict(_observation())
    assert len(store) == 0


def test_the_prompt_axes_are_wired_but_not_live():
    """Asserting both halves so neither can be read as a finding: the digest
    changes with the configuration, and the offline client's answer does not."""
    from dataclasses import replace

    from examples.advanced.robocasa.study_model import (
        PROMPT_VARIANTS, OfflineModelClient, assemble_model_prompt)
    from examples.advanced.robocasa.study_transcript import prompt_digest

    observation = _observation()
    configs = [replace(_config(), prompt_variant=name) for name in sorted(PROMPT_VARIANTS)]
    digests = {prompt_digest(assemble_model_prompt(observation, config, dof=7))
               for config in configs}
    assert len(digests) == len(configs), "the axis does not reach the model boundary"

    answers = set()
    for config in configs:
        method = _direct(OfflineModelClient(method=StillMethod(horizon=5)), config=config)
        answers.add(method.predict(observation).actions.tobytes())
    assert len(answers) == 1, "the offline client is supposed to ignore the configuration"


# --- the task-specification axis ----------------------------------------


def test_the_default_specification_is_robocasas_own_goal():
    from examples.advanced.robocasa.study_tasks import get_task, specification

    task = get_task("PickPlaceCounterToCabinet")
    spec = specification(task, "default")
    assert spec["text"] == task.goal
    assert spec["derived"] is False and spec["names_object"] is True


def test_the_vague_specification_is_derived_and_withholds_the_object():
    """Derived from the environment name rather than authored, so no
    instruction in this lane is invented text."""
    from examples.advanced.robocasa.study_tasks import specification

    spec = specification("PickPlaceCounterToCabinet", "vague")
    assert spec["text"] == "Move the object from the counter to the cabinet."
    assert spec["derived"] is True and spec["names_object"] is False
    assert spec["fallback"] is False
    # The object identity is exactly what it withholds.
    assert "sugar" not in spec["text"]


def test_the_vague_form_keeps_the_fixtures_for_nearly_every_task():
    """A vague instruction that carried no fixtures would be a different
    condition, so the fallback has to be rare and reported."""
    from examples.advanced.robocasa.study_tasks import PNP_TASKS, specification

    fallbacks = [task.task for task in PNP_TASKS
                 if specification(task, "vague").get("fallback")]
    assert len(fallbacks) == 1 and fallbacks == ["PickPlaceCoffee"]


def test_a_task_without_a_source_and_destination_falls_back_and_says_so():
    from examples.advanced.robocasa.study_tasks import specification

    spec = specification("PickPlaceCoffee", "vague")
    assert spec["fallback"] is True and spec["names_object"] is False


def test_an_unknown_specification_style_is_rejected():
    from examples.advanced.robocasa.study_tasks import specification

    with pytest.raises(ValueError, match="style must be one of"):
        specification("PickPlaceCounterToCabinet", "interpretive_dance")


def test_the_environment_rejects_an_unknown_specification_style():
    with pytest.raises(ValueError, match="specification_style"):
        RoboCasaPnPEnvironment(specification_style="mime")


def test_every_committed_task_has_both_specification_styles():
    from examples.advanced.robocasa.study_tasks import (
        PNP_TASKS, SPECIFICATION_STYLES, specification)

    for task in PNP_TASKS:
        texts = {style: specification(task, style)["text"] for style in SPECIFICATION_STYLES}
        assert all(text.strip() for text in texts.values())
        # The two conditions must differ, or the axis does nothing.
        assert texts["default"] != texts["vague"]


# --- the joint-delta representation --------------------------------------
#
# The reference policy drives joint targets, and the envelope bounds every
# action element to 1.0 while a joint target is in radians up to about 3. So
# the representation has to be a *normalised* delta, and these pin that it is.


def test_joint_delta_is_a_declared_representation():
    from examples.advanced.robocasa.study_representations import (
        REPRESENTATION_DOF, REPRESENTATIONS)

    assert "joint_delta" in REPRESENTATIONS
    assert REPRESENTATION_DOF["joint_delta"] == 8


def test_a_joint_delta_row_scales_into_joint_targets():
    from examples.advanced.robocasa.study_representations import JOINT_DELTA_SCALE, joint_targets

    current = np.zeros(7)
    targets, gripper = joint_targets(current, np.array([1.0, -1.0, 0.0, 0, 0, 0, 0, 1.0]))
    assert targets[0] == pytest.approx(JOINT_DELTA_SCALE)
    assert targets[1] == pytest.approx(-JOINT_DELTA_SCALE)
    assert targets[2] == pytest.approx(0.0)
    assert gripper == 1.0


def test_a_joint_delta_row_is_clipped_before_scaling():
    """An out-of-band row cannot command a larger step than the scale allows."""
    from examples.advanced.robocasa.study_representations import JOINT_DELTA_SCALE, joint_targets

    targets, _ = joint_targets(np.zeros(7), np.array([9.0, -9.0, 0, 0, 0, 0, 0, 0.0]))
    assert targets[0] == pytest.approx(JOINT_DELTA_SCALE)
    assert targets[1] == pytest.approx(-JOINT_DELTA_SCALE)


def test_a_joint_delta_row_is_added_to_the_current_posture():
    """It is a delta, not a target: the same row from elsewhere lands elsewhere."""
    from examples.advanced.robocasa.study_representations import joint_targets

    row = np.array([1.0, 0, 0, 0, 0, 0, 0, 0.0])
    first, _ = joint_targets(np.zeros(7), row)
    second, _ = joint_targets(np.full(7, 0.5), row)
    assert np.allclose(second - first, 0.5)


@pytest.mark.parametrize("row", [np.zeros(7), np.zeros(9)])
def test_a_misshaped_joint_delta_row_is_rejected(row):
    from examples.advanced.robocasa.study_representations import joint_targets

    with pytest.raises(ValueError, match="joint_delta"):
        joint_targets(np.zeros(7), row)


def test_a_non_finite_joint_delta_row_is_rejected():
    from examples.advanced.robocasa.study_representations import joint_targets

    with pytest.raises(ValueError):
        joint_targets(np.zeros(7), np.array([np.nan, 0, 0, 0, 0, 0, 0, 0.0]))


@pytest.mark.parametrize("scale", [0.0, -0.1, 2.0, 1])
def test_an_unusable_joint_delta_scale_is_rejected(scale):
    from examples.advanced.robocasa.study_representations import joint_targets

    with pytest.raises(ValueError, match="scale"):
        joint_targets(np.zeros(7), np.zeros(8), scale=scale)


def test_a_joint_representation_does_not_command_the_base():
    """The joint path drives seven arm joints; a base command has nowhere to go."""
    from examples.advanced.robocasa.study_robocasa_env import RoboCasaPnPEnvironment

    with pytest.raises(ValueError, match="does not command the base"):
        RoboCasaPnPEnvironment(task="PickPlaceCounterToCabinet",
                               representation="joint_delta", command_base=True)


# --- the scripted reference ----------------------------------------------


class _FakeArm:
    """Enough of `ArmModel` to exercise the phase machine without MuJoCo."""

    def __init__(self, env, joints=None):
        self.env = env
        self.posture = np.zeros(7) if joints is None else np.asarray(joints, dtype=float)

    def joints(self):
        return self.posture.copy()

    def hand(self):
        return np.array([0.4, -0.3, 1.0])


class _StubbedWorld:
    """Stands in for `RoboCasaPnPEnvironment`; the reference only reads `.env`."""

    env = object()


def _reference(**kwargs):
    from examples.advanced.robocasa.study_reference import ScriptedPickMethod

    method = ScriptedPickMethod(environment=_StubbedWorld(), **kwargs)
    method._arm = _FakeArm(method.environment.env)
    method._start_height = 0.9
    method._object_position = lambda: np.array([0.4, -0.3, 0.9])  # type: ignore[method-assign]
    method._grasped = lambda: False  # type: ignore[method-assign]
    return method


@pytest.mark.parametrize("horizon", [0, 33, 1.0, "5"])
def test_the_reference_rejects_an_unusable_horizon(horizon):
    from examples.advanced.robocasa.study_reference import ScriptedPickMethod

    with pytest.raises(ValueError, match="horizon"):
        ScriptedPickMethod(environment=_StubbedWorld(), horizon=horizon)


@pytest.mark.parametrize("name", ["grasp_height", "lift_height"])
def test_the_reference_rejects_an_unusable_height(name):
    from examples.advanced.robocasa.study_reference import ScriptedPickMethod

    with pytest.raises(ValueError, match=name):
        ScriptedPickMethod(environment=_StubbedWorld(), **{name: 0.0})


def test_the_reference_stages_above_the_object_then_descends_to_it():
    """The approach target clears the object; the grasp target straddles it."""
    from examples.advanced.robocasa.study_reference import APPROACH_HEIGHT

    method = _reference()
    above = method._target_for("approach")
    grasp = method._target_for("descend")
    assert above[2] - 0.9 == pytest.approx(APPROACH_HEIGHT)
    assert grasp[2] - 0.9 == pytest.approx(method.grasp_height)
    assert above[2] > grasp[2]


def test_the_reference_lifts_straight_up_from_wherever_it_grasped():
    """The lift keeps x and y: dragging a held object sideways drops it."""
    method = _reference()
    target = method._target_for("lift")
    assert target[0] == pytest.approx(0.4)
    assert target[1] == pytest.approx(-0.3)
    assert target[2] == pytest.approx(0.9 + method.lift_height)


def test_the_reference_emits_one_row_per_horizon_step_within_the_bound():
    method = _reference(horizon=5)
    method.phase = "close"
    chunk = method.predict(_observation())
    assert chunk.actions.shape == (5, 8)
    assert chunk.horizon == 5 and chunk.dof == 8
    assert np.abs(chunk.actions).max() <= 1.0


def test_the_reference_output_passes_the_safety_envelope():
    """Whatever the phase, the reference is held to the bound every method is."""
    from examples.advanced.robocasa.method_harness import SafetyEnvelope

    envelope = SafetyEnvelope()
    for phase in ("close", "done"):
        method = _reference(horizon=3)
        method.phase = phase
        envelope.validate(method.predict(_observation()))


def test_the_reference_opens_the_gripper_before_the_grasp_and_closes_after():
    method = _reference()
    method.phase = "descend"
    assert method._row()[7] == -1.0
    method.phase = "lift"
    assert method._row()[7] == 1.0


def test_the_reference_holds_still_once_done():
    method = _reference()
    method.phase = "done"
    row = method._row()
    assert np.allclose(row[:7], 0.0)
    assert row[7] == 1.0


def test_the_reference_closes_for_the_declared_window_then_advances():
    from examples.advanced.robocasa.study_reference import CLOSE_STEPS

    method = _reference()
    method.phase = "close"
    method._plan_for = lambda phase: False  # type: ignore[method-assign]
    for _ in range(CLOSE_STEPS - 1):
        method._row()
    assert method.phase == "close"
    method._row()
    assert method.phase != "close"


def test_a_phase_that_cannot_be_planned_is_skipped_rather_than_retried():
    """An unplannable phase ends the episode on the rung it reached."""
    method = _reference()
    method._plan_for = lambda phase: False  # type: ignore[method-assign]
    method.phase = "close"
    method.closed_for = 10 ** 6
    method._row()
    assert method.phase == "done"


def test_the_reference_gates_the_command_on_the_arm():
    """The command leads the arm by at most the follow tolerance."""
    from examples.advanced.robocasa.study_reference import FOLLOW_INCREMENT, FOLLOW_TOLERANCE

    method = _reference()
    goal = np.full(7, 1.0)
    method._path, method._leg, method._travelled = [np.zeros(7), goal], 1, 0.0
    first = method._commanded_point()
    assert np.linalg.norm(first) == pytest.approx(FOLLOW_INCREMENT)
    # The arm never moves here, so however many times it is asked, the command
    # cannot run away down the leg - it stalls one increment past the tolerance
    # and waits. That bound is the whole mechanism keeping executed motion on
    # the planner's verified line rather than on a route of its own.
    for _ in range(200):
        method._commanded_point()
    assert np.linalg.norm(method.arm.joints()) == 0.0
    assert method._travelled <= FOLLOW_TOLERANCE + FOLLOW_INCREMENT + 1e-9
    assert method._leg == 1


def test_the_reference_settles_at_a_leg_before_starting_the_next():
    """Reaching the end of a leg is not arriving at it.

    Advancing immediately left the hand 2.5 cm above the object and closed the
    fingers on each other, so each leg holds its target for a settle window.
    """
    from examples.advanced.robocasa.study_reference import FOLLOW_SETTLE

    method = _reference()
    target = np.full(7, 0.001)
    method._path, method._leg, method._travelled = [np.zeros(7), target, np.full(7, 0.002)], 1, 0.0
    method._commanded_point()  # travels the whole (tiny) leg in one step
    assert method._travelled > 0
    held = [method._commanded_point() for _ in range(FOLLOW_SETTLE)]
    assert all(np.allclose(point, target) for point in held[:-1])
    assert method._leg == 2


def test_the_reference_reports_its_privilege():
    """It reads the simulator, so it is a reference and must say so."""
    method = _reference()
    assert method.describe()["privileged"] is True


def test_the_reference_records_why_a_plan_failed():
    """A phase that silently does not happen is the hardest kind of bug."""
    method = _reference()
    method._target_for = lambda phase: None  # type: ignore[method-assign]
    assert method._plan_for("lift") is False
    assert method.failures[0]["reason"] == "no target"


def test_the_reference_builds_a_chunk_against_one_posture_by_default():
    """The measured interaction behind the horizon axis.

    Rows are built before any is dispatched, so without a forward model every
    row in a chunk sees the posture the chunk began at. Against an integrating
    representation that compounds: measured, horizon 15 commands 0.750 rad per
    chunk against horizon 1's 0.050, and the score goes from 75 to 0.
    """
    method = _reference(horizon=4)
    method._path, method._leg, method._travelled = [np.zeros(7), np.full(7, 1.0)], 1, 0.0
    rows = method.predict(_observation()).actions
    assert method._predicted is None
    # The gate advances a few increments, then stalls and repeats, because the
    # arm it is gating on never appears to move.
    assert np.allclose(rows[-1], rows[-2])


def test_the_reference_can_predict_its_own_posture_forward():
    """Dead reckoning lets one chunk walk the path instead of stalling on it.

    The difference is the whole interaction: a chunk gated on a posture that
    never appears to move advances one increment and then repeats itself, while
    one gated on its own predictions advances every row. Which of those is
    better depends on the horizon, and neither is right - see the module
    docstring for what each costs.
    """
    from examples.advanced.robocasa.study_reference import FOLLOW_INCREMENT

    stale, predicting = _reference(horizon=4), _reference(horizon=4, dead_reckon=True)
    for method in (stale, predicting):
        method._path, method._leg, method._travelled = [np.zeros(7), np.full(7, 1.0)], 1, 0.0
        method.predict(_observation())
    assert stale._predicted is None
    assert predicting._predicted is not None
    assert predicting._travelled == pytest.approx(4 * FOLLOW_INCREMENT)
    assert predicting._travelled > stale._travelled


def test_dead_reckoning_is_reported_in_the_description():
    """It changes the policy's behaviour, so a run has to record which it was."""
    assert _reference().describe()["dead_reckon"] is False
    assert _reference(dead_reckon=True).describe()["dead_reckon"] is True


@pytest.mark.parametrize("gain", [0.0, -0.5, 1.5, 1])
def test_an_unusable_tracking_gain_is_rejected(gain):
    from examples.advanced.robocasa.study_reference import ScriptedPickMethod

    with pytest.raises(ValueError, match="tracking"):
        ScriptedPickMethod(environment=_StubbedWorld(), tracking=gain)


def test_the_forward_model_gain_is_the_measured_one():
    """A gain of 1.0 assumes a row lands where it was aimed, which it does not.

    Holding a full-scale row delivers 0.148 of the commanded step on the first
    tick and 0.265 in steady state, so the naive model overshoots fourfold and
    the gate advances on arrivals that have not happened.
    """
    from examples.advanced.robocasa.study_reference import TRACKING_FRACTION

    assert 0.0 < TRACKING_FRACTION < 1.0
    assert _reference().tracking == TRACKING_FRACTION


def _travel(horizon, gain):
    """How far down a leg one chunk advances the command."""
    method = _reference(horizon=horizon, dead_reckon=True, tracking=gain)
    method._path, method._leg, method._travelled = [np.zeros(7), np.full(7, 1.0)], 1, 0.0
    method.predict(_observation())
    return method._travelled


def test_a_calibrated_forward_model_advances_less_than_a_naive_one():
    """The gain is what separates an optimistic gate from a truthful one.

    It only bites once the predicted lead has opened past the gate's tolerance,
    which takes about ten rows here - so a short chunk is unaffected and a long
    one is held back, which is the shape of the measured scores.
    """
    assert _travel(8, 0.265) == pytest.approx(_travel(8, 1.0))
    for horizon in (10, 16, 32):
        assert _travel(horizon, 0.265) < _travel(horizon, 1.0)


# --- the released VLA policy ---------------------------------------------
#
# The mapping between the checkpoint's action ordering and this robot's is the
# one place where a mistake would look like success: both are 12 wide, so an
# unpermuted row steps the simulator happily while commanding the base with
# end-effector deltas. These pin it against the environment's own slices.


def test_the_lerobot_permutation_is_the_documented_one():
    from examples.advanced.robocasa.study_vla import LEROBOT_ACTION_DOF, to_native_action

    row = np.arange(LEROBOT_ACTION_DOF, dtype=np.float32)
    assert to_native_action(row).tolist() == [5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4]


def test_each_lerobot_segment_lands_in_the_environment_slice_that_owns_it():
    """Cross-checked against the backend's constants, not restated from them."""
    from examples.advanced.robocasa.study_robocasa_env import _ARM, _BASE, _GRIPPER
    from examples.advanced.robocasa.study_vla import to_native_action

    row = np.zeros(12, dtype=np.float32)
    row[0:3] = [0.1, 0.2, 0.3]        # base x, y, yaw
    row[3] = 0.0                      # torso, frozen in the training data
    row[4] = 1.0                      # base_mode
    row[5:8] = [0.4, 0.5, 0.6]        # eef position
    row[8:11] = [0.7, 0.8, 0.9]       # eef rotation
    row[11] = -1.0                    # gripper
    native = to_native_action(row)
    assert native[_ARM].tolist() == pytest.approx([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    assert native[_GRIPPER].tolist() == pytest.approx([-1.0])
    assert native[_BASE].tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert native[10] == pytest.approx(0.0)   # torso
    assert native[11] == pytest.approx(1.0)   # base_mode


def test_the_permutation_is_a_permutation():
    """Nothing is dropped or duplicated, which a hand-written mapping can do."""
    from examples.advanced.robocasa.study_vla import LEROBOT_ACTION_DOF, NATIVE_FROM_LEROBOT

    assert sorted(NATIVE_FROM_LEROBOT) == list(range(LEROBOT_ACTION_DOF))


@pytest.mark.parametrize("row", [np.zeros(7), np.zeros(13), np.zeros((2, 12))])
def test_a_misshaped_lerobot_row_is_rejected(row):
    from examples.advanced.robocasa.study_vla import to_native_action

    with pytest.raises(ValueError, match="12 wide"):
        to_native_action(row)


def test_a_non_finite_lerobot_row_is_rejected():
    from examples.advanced.robocasa.study_vla import to_native_action

    row = np.zeros(12)
    row[4] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        to_native_action(row)


def _raw_state(**overrides):
    raw = {"robot0_base_pos": np.array([1.0, 2.0, 0.7]),
           "robot0_base_quat": np.array([0.0, 0.0, 0.3, 0.95]),
           "robot0_eef_pos": np.array([0.4, -0.3, 1.0]),
           "robot0_eef_quat": np.array([0.1, 0.2, 0.3, 0.9]),
           "robot0_gripper_qpos": np.array([0.03, -0.03])}
    raw.update(overrides)
    return raw


def test_the_lerobot_state_is_sixteen_wide_in_the_documented_order():
    from examples.advanced.robocasa.study_vla import LEROBOT_STATE_DOF, lerobot_state

    state = lerobot_state(_raw_state())
    assert state.shape == (LEROBOT_STATE_DOF,)
    # base pose passes through unchanged; the hand pose does not (see below).
    assert state[:7].tolist() == pytest.approx([1.0, 2.0, 0.7, 0.0, 0.0, 0.3, 0.95])
    assert state[14:].tolist() == pytest.approx([0.03, -0.03])


def test_the_hand_pose_is_converted_into_the_base_frame():
    """The checkpoint was trained on a base-frame hand pose, not a world one.

    Robosuite reports `robot0_eef_pos` in world coordinates, and feeding that
    through would put the policy's proprioception in a space it never saw while
    raising nothing. With the base at the origin and unrotated, the two frames
    coincide; rotate or translate the base and they must not.
    """
    from examples.advanced.robocasa.study_vla import lerobot_state

    identity = _raw_state(robot0_base_pos=np.zeros(3),
                          robot0_base_quat=np.array([0.0, 0.0, 0.0, 1.0]))
    assert lerobot_state(identity)[7:10].tolist() == pytest.approx([0.4, -0.3, 1.0])
    # A base translated by one metre in x sees the hand one metre nearer.
    shifted = _raw_state(robot0_base_pos=np.array([1.0, 0.0, 0.0]),
                         robot0_base_quat=np.array([0.0, 0.0, 0.0, 1.0]))
    assert lerobot_state(shifted)[7:10].tolist() == pytest.approx([-0.6, -0.3, 1.0])
    # A base yawed 90 degrees swaps the axes rather than leaving them alone.
    turned = _raw_state(robot0_base_pos=np.zeros(3),
                        robot0_base_quat=np.array([0.0, 0.0, np.sin(np.pi / 4),
                                                   np.cos(np.pi / 4)]))
    assert lerobot_state(turned)[7:10].tolist() == pytest.approx([-0.3, -0.4, 1.0], abs=1e-6)


def test_the_relative_rotation_is_canonicalised_to_positive_w():
    """Measured: state dimension 13 never goes negative across 15M frames."""
    from examples.advanced.robocasa.study_vla import lerobot_state

    for sign in (1.0, -1.0):
        raw = _raw_state(robot0_base_quat=np.array([0.0, 0.0, 0.0, 1.0]),
                         robot0_eef_quat=sign * np.array([0.1, 0.2, 0.3, 0.9]))
        state = lerobot_state(raw)
        assert state[13] >= 0.0
    # Both signs encode the same rotation, so both must give the same state.
    first = lerobot_state(_raw_state(robot0_base_quat=np.array([0.0, 0.0, 0.0, 1.0]),
                                     robot0_eef_quat=np.array([0.1, 0.2, 0.3, 0.9])))
    second = lerobot_state(_raw_state(robot0_base_quat=np.array([0.0, 0.0, 0.0, 1.0]),
                                      robot0_eef_quat=-np.array([0.1, 0.2, 0.3, 0.9])))
    assert first.tolist() == pytest.approx(second.tolist(), abs=1e-6)


def test_a_missing_state_observable_is_named():
    from examples.advanced.robocasa.study_vla import lerobot_state

    raw = _raw_state()
    del raw["robot0_eef_quat"]
    with pytest.raises(KeyError, match="robot0_eef_quat"):
        lerobot_state(raw)


def test_a_state_that_assembles_to_the_wrong_width_is_rejected():
    from examples.advanced.robocasa.study_vla import lerobot_state

    with pytest.raises(ValueError, match="expected"):
        lerobot_state(_raw_state(robot0_gripper_qpos=np.array([0.03])))


def test_untiling_inverts_the_observation_tiling_exactly():
    """The policy wants three views; the contract carries one tiled image."""
    from examples.advanced.robocasa.study_observation import tile_frames
    from examples.advanced.robocasa.study_vla import untile

    views = [np.full((64, 64, 3), fill, dtype=np.uint8) for fill in (10, 20, 30)]
    recovered = untile(tile_frames([views]), views=3)
    assert len(recovered) == 3
    for original, back in zip(views, recovered):
        assert np.array_equal(original, back)


@pytest.mark.parametrize("image,match", [
    (np.zeros((64, 100, 3), dtype=np.uint8), "does not divide"),
    (np.zeros((64, 64, 3), dtype=np.uint8), "does not divide"),
    (np.zeros((64, 195, 3), dtype=np.uint8), "must be square"),
    (np.zeros((64, 64), dtype=np.uint8), r"\(H, W, 3\)"),
])
def test_an_untileable_image_is_rejected(image, match):
    from examples.advanced.robocasa.study_vla import untile

    with pytest.raises(ValueError, match=match):
        untile(image, views=3)


def test_the_vla_declares_itself_a_model_and_not_privileged():
    """It sees only images, state and the instruction - no simulator reads."""
    from examples.advanced.robocasa.study_vla import SmolVlaPolicy

    described = SmolVlaPolicy().describe()
    assert described["source"] == "model"
    assert described["privileged"] is False
    assert described["checkpoint"] == "lerobot/smolvla_robocasa"


def test_the_vla_rejects_an_empty_checkpoint():
    from examples.advanced.robocasa.study_vla import SmolVlaPolicy

    with pytest.raises(ValueError, match="checkpoint"):
        SmolVlaPolicy(checkpoint="  ")


def test_the_vla_policy_can_describe_itself_for_reconstruction():
    """`init_config` is what lets a run record name and rebuild the policy."""
    from examples.advanced.robocasa.study_vla import DEFAULT_CHECKPOINT, SmolVlaPolicy

    config = SmolVlaPolicy().init_config()
    assert config == {"checkpoint": DEFAULT_CHECKPOINT, "device": "cpu",
                      "flip_images": True, "seed": 0}
    # The seed belongs in the config because this policy samples: the chunk it
    # returns for one observation depends on it, so a record that omitted the
    # seed would name a policy that cannot be made to produce the run again.
    assert SmolVlaPolicy(seed=7).init_config()["seed"] == 7


def test_the_vla_policy_samples_reproducibly_once_seeded():
    """Two seeded resets must leave the sampler in the same state.

    Unseeded, this policy returned chunk sums of -108.6 and -115.5 for one
    fixed observation on consecutive calls, because flow matching integrates
    from sampled noise. That made an episode unreproducible even though the
    simulator's scene was pinned, and it made validating a port impossible —
    a CPU and a GPU run would differ for two reasons with no way to separate
    them. Asserted on the torch generator rather than on a forward pass so the
    check needs no checkpoint, no network and no GPU.
    """
    torch = pytest.importorskip("torch")
    from examples.advanced.robocasa.study_vla import SmolVlaPolicy

    policy = SmolVlaPolicy(seed=7)
    policy.reset()
    first = torch.randn(4)
    policy.reset()
    assert torch.equal(torch.randn(4), first)
    # And a different seed must actually move it, or the seeding is inert.
    policy.reset(8)
    assert not torch.equal(torch.randn(4), first)
    assert policy.seed == 8


def test_a_model_source_needs_the_envelope_widened_deliberately():
    """The default allowlist is scripted-only, and that is load-bearing.

    The existing model methods declare "scripted" to get through it, which
    mislabels them; this method declares the truth, so a sweep running it has
    to say so in its protocol block.
    """
    from examples.advanced.openpi_policy.common import ActionChunk
    from examples.advanced.robocasa.method_harness import SafetyEnvelope
    from examples.advanced.robocasa.study_sweep import Sweep

    assert Sweep(build_environment=lambda *a, **k: None).allowed_sources == ("scripted",)
    chunk = ActionChunk(actions=np.zeros((1, 12), dtype=np.float32), horizon=1,
                        dof=12, source="model")
    with pytest.raises(ValueError):
        SafetyEnvelope().validate(chunk)
    SafetyEnvelope(allowed_sources=frozenset({"model"})).validate(chunk)


def test_the_lerobot_representation_is_registered_at_twelve_wide():
    from examples.advanced.robocasa.study_representations import (
        REPRESENTATION_DOF, REPRESENTATIONS)

    assert "lerobot_12" in REPRESENTATIONS
    assert REPRESENTATION_DOF["lerobot_12"] == 12


def test_a_passthrough_representation_does_not_derive_the_base_mode():
    """The row carries its own mode flag; deriving one would overwrite it."""
    from examples.advanced.robocasa.study_robocasa_env import RoboCasaPnPEnvironment

    with pytest.raises(ValueError, match="carries its own base command"):
        RoboCasaPnPEnvironment(representation="lerobot_12", command_base=True)


def test_an_unknown_state_mode_is_rejected():
    from examples.advanced.robocasa.study_robocasa_env import RoboCasaPnPEnvironment

    with pytest.raises(ValueError, match="state_mode"):
        RoboCasaPnPEnvironment(state_mode="whatever")


def test_the_third_agentview_is_a_declared_role():
    """The checkpoint was trained with both agentview cameras plus the wrist."""
    from examples.advanced.robocasa.study_robocasa_env import CAMERA_ROLES
    from examples.advanced.robocasa.study_vla import LEROBOT_VIEWS

    from examples.advanced.robocasa.study_robocasa_env import (
        ALL_CAMERA_ROLES, ObservationConfig)

    for view in LEROBOT_VIEWS:
        assert view in ALL_CAMERA_ROLES
    assert [ALL_CAMERA_ROLES[v] for v in LEROBOT_VIEWS] == [
        "robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]
    # The third view is RoboCasa-only, so it must not widen the portable set
    # that the RoboLab backend is held equal to.
    assert "agentview_right" not in CAMERA_ROLES
    assert ObservationConfig(views=LEROBOT_VIEWS, image_size=256).portable is False
    assert ObservationConfig(views=("over_shoulder", "wrist")).portable is True


def test_the_identifier_specification_is_the_task_name():
    """A released checkpoint's language conditioning need not be a sentence.

    The RoboCasa LeRobot dataset stores its `tasks` field as the environment
    name, so a policy trained on it is conditioned on the identifier. Measured:
    the identifier reaches the object and the goal sentence does not.
    """
    from examples.advanced.robocasa.study_tasks import specification

    spec = specification("PickPlaceCounterToCabinet", "identifier")
    assert spec["text"] == "PickPlaceCounterToCabinet"
    assert spec["style"] == "identifier"
    assert spec["derived"] is False
    assert spec["names_object"] is False


def test_every_committed_task_has_an_identifier_specification():
    from examples.advanced.robocasa.study_tasks import list_tasks, specification

    for task in list_tasks():
        assert specification(task, "identifier")["text"] == task


def test_the_identifier_style_is_distinct_from_the_other_two():
    from examples.advanced.robocasa.study_tasks import SPECIFICATION_STYLES, specification

    assert set(SPECIFICATION_STYLES) == {"default", "vague", "identifier"}
    texts = {specification("PickPlaceCounterToCabinet", s)["text"]
             for s in SPECIFICATION_STYLES}
    assert len(texts) == 3


def test_the_vla_flips_frames_to_the_training_convention():
    """Robosuite renders bottom-up; the dataset was converted top-down.

    Settled against a decoded training frame: its bottom quarter is brighter
    than its top while ours is the reverse, and the row-brightness profile
    correlates -0.120 as rendered against +0.570 flipped.
    """
    from examples.advanced.robocasa.study_vla import FLIP_IMAGES, SmolVlaPolicy

    assert FLIP_IMAGES is True
    assert SmolVlaPolicy().flip_images is True
    assert SmolVlaPolicy().describe()["flip_images"] is True
    assert SmolVlaPolicy(flip_images=False).describe()["flip_images"] is False


# --- resumable sweeps ----------------------------------------------------
#
# A full sweep is hours of simulation and the summary prints only at the end,
# so an interruption at 90% used to lose everything.


def _cell_specs():
    from examples.advanced.robocasa.study_sweep import MethodSpec

    return (MethodSpec(method_id="still", horizon=2,
                       build=lambda env: StillMethod(horizon=2)),)


def _build_fake(task, representation):
    return _FakeEnvironment(task=task, representation=representation)


def test_a_sweep_without_a_checkpoint_writes_nothing(tmp_path):
    from examples.advanced.robocasa.study_sweep import Sweep

    sweep = Sweep(build_environment=_build_fake, tasks=("T",), seeds=(0,))
    sweep.run(_cell_specs())
    assert list(tmp_path.iterdir()) == []
    assert sweep.resumed == 0


def test_a_checkpointed_sweep_records_every_finished_episode(tmp_path):
    from examples.advanced.robocasa.study_sweep import Sweep

    path = tmp_path / "runs" / "cells.jsonl"
    sweep = Sweep(build_environment=_build_fake, tasks=("T", "U"), seeds=(0, 1),
                  checkpoint=str(path))
    sweep.run(_cell_specs())
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    assert len(lines) == len(sweep.results) == 4
    import json as _json

    assert all(_json.loads(line)["task"] in ("T", "U") for line in lines)


def test_a_resumed_sweep_reuses_finished_cells_and_runs_only_the_rest(tmp_path):
    """The point of the checkpoint: an interrupted sweep does not start over."""
    from examples.advanced.robocasa.study_sweep import Sweep

    path = tmp_path / "cells.jsonl"
    first = Sweep(build_environment=_build_fake, tasks=("T",), seeds=(0, 1),
                  checkpoint=str(path))
    first.run(_cell_specs())
    assert first.resumed == 0

    built = []

    def counting(task, representation):
        built.append(task)
        return _build_fake(task=task, representation=representation)

    second = Sweep(build_environment=counting, tasks=("T",), seeds=(0, 1, 2),
                   checkpoint=str(path))
    summary = second.run(_cell_specs())
    # Seeds 0 and 1 came from disk; only seed 2 built an environment.
    assert second.resumed == 2
    assert len(built) == 1
    assert len(second.results) == 3
    assert summary["protocol"]["resumed_episodes"] == 2


def test_a_resumed_cell_keeps_its_original_measurements(tmp_path):
    from examples.advanced.robocasa.study_sweep import Sweep

    path = tmp_path / "cells.jsonl"
    first = Sweep(build_environment=_build_fake, tasks=("T",), seeds=(0,),
                  checkpoint=str(path))
    first.run(_cell_specs())
    second = Sweep(build_environment=_build_fake, tasks=("T",), seeds=(0,),
                   checkpoint=str(path))
    second.run(_cell_specs())
    assert second.results[0].to_dict() == first.results[0].to_dict()


def test_a_corrupt_checkpoint_line_is_reported_rather_than_skipped(tmp_path):
    """Being killed mid-write is normal; silently dropping a cell is not."""
    from examples.advanced.robocasa.study_sweep import Sweep

    path = tmp_path / "cells.jsonl"
    path.write_text('{"task": "T", "seed": 0, "method_id": "still"\n')
    sweep = Sweep(build_environment=_build_fake, tasks=("T",), seeds=(0,),
                  checkpoint=str(path))
    with pytest.raises(ValueError, match="readable episode record"):
        sweep.run(_cell_specs())


def test_blank_checkpoint_lines_are_tolerated(tmp_path):
    from examples.advanced.robocasa.study_sweep import Sweep

    path = tmp_path / "cells.jsonl"
    first = Sweep(build_environment=_build_fake, tasks=("T",), seeds=(0,),
                  checkpoint=str(path))
    first.run(_cell_specs())
    path.write_text("\n" + path.read_text() + "\n\n")
    second = Sweep(build_environment=_build_fake, tasks=("T",), seeds=(0,),
                   checkpoint=str(path))
    second.run(_cell_specs())
    assert second.resumed == 1


def test_the_replan_cadence_is_the_leaderboards_own_value():
    """Read from the official evaluation client, not chosen by us.

    `robocasa-benchmark/openpi`, `examples/robocasa/main.py` declares
    `replan_steps: int = 5` and applies `action_chunk[: replan_steps]`, so a
    checkpoint predicting fifty steps is evaluated closed-loop every fifth one.
    Matching it is a condition of comparability.
    """
    from examples.advanced.robocasa.study_vla import REPLAN_STEPS

    assert REPLAN_STEPS == 5


def test_the_vla_arm_is_a_flow_behind_the_official_cadence():
    """The arm is the adapter plus the policy, not a bespoke method.

    Truncation lives in `FlowMethod.take` and is tested there once, for every
    policy, rather than reimplemented per checkpoint.
    """
    from examples.advanced.robocasa.study_flow_method import FlowMethod
    from examples.advanced.robocasa.study_vla import (
        REPLAN_STEPS, SmolVlaPolicy, smolvla_arm)

    arm = smolvla_arm()
    assert isinstance(arm, FlowMethod)
    assert isinstance(arm.policy, SmolVlaPolicy)
    assert arm.take == REPLAN_STEPS
    assert arm.describe()["policy_ref"].endswith(":SmolVlaPolicy")
    # The open-loop condition this study used before reading the protocol.
    assert smolvla_arm(take=None).take is None


def test_the_vla_policy_returns_its_whole_proposal():
    """The policy proposes; the adapter decides how much of it runs.

    Keeping those apart is what makes the model-call count right: with the
    chunk held in a private queue and one row handed out per call, the harness
    saw one dispatch per control step and reported 1,130 model calls for an
    episode with 23 inferences.
    """
    from examples.advanced.robocasa.study_vla import LEROBOT_ACTION_DOF, SmolVlaPolicy

    policy = SmolVlaPolicy()
    policy._infer = lambda observation: [  # type: ignore[method-assign]
        np.full(LEROBOT_ACTION_DOF, float(i)) for i in range(50)]
    chunk = policy.step(_observation())
    assert chunk.actions.shape == (50, LEROBOT_ACTION_DOF)
    assert chunk.horizon == 50
    assert policy.calls == 1


def test_the_arm_consumes_only_the_leading_actions_of_each_proposal():
    """End to end through the adapter: 5 kept, 45 discarded, then re-inferred."""
    from examples.advanced.robocasa.study_vla import LEROBOT_ACTION_DOF, smolvla_arm

    arm = smolvla_arm()
    inferences = []
    arm.policy._infer = lambda observation: (  # type: ignore[method-assign]
        inferences.append(1) or [np.full(LEROBOT_ACTION_DOF, float(i)) for i in range(50)])
    first = arm.predict(_observation())
    assert first.actions[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert arm.rows_dispatched == 5 and arm.rows_discarded == 45
    assert len(inferences) == 1
    # The harness dispatches those five, then asks again - a fresh inference.
    arm.predict(_observation())
    assert len(inferences) == 2


def test_the_atomic_seen_split_is_eighteen_tasks_with_five_pick_and_place():
    """The leaderboard's split, not ours.

    Reconstruction check: the mean of REKI's 18 per-task rates is 65.8, exactly
    the atomic_seen_success it reports, which is what confirms the membership.
    """
    from examples.advanced.robocasa.study_tasks import (
        ATOMIC_SEEN_TASKS, VLA_COMPARABLE_TASKS)

    assert len(ATOMIC_SEEN_TASKS) == 18
    assert len(set(ATOMIC_SEEN_TASKS)) == 18
    pnp = [t for t in ATOMIC_SEEN_TASKS if t.startswith("PickPlace")]
    assert len(pnp) == 5
    # Our comparable set is exactly the pick-and-place part of their split.
    assert sorted(pnp) == sorted(VLA_COMPARABLE_TASKS)


def test_the_official_trial_count_is_recorded():
    from examples.advanced.robocasa.study_tasks import DEFAULT_SEEDS, OFFICIAL_TRIALS

    assert OFFICIAL_TRIALS == 50
    # Our own protocol is cheaper and must not be mistaken for theirs.
    assert len(DEFAULT_SEEDS) < OFFICIAL_TRIALS


# --- running a Retriever Flow as a study method --------------------------


class _FakeFlow:
    """A minimal stand-in for Flow[PolicyObservation, ActionChunk]."""

    def __init__(self, horizon=10, dof=7, source="mock"):
        self.horizon, self.dof, self.source = horizon, dof, source
        self.reset_calls = 0
        self.steps = 0

    def init_config(self):
        return {"horizon": self.horizon, "dof": self.dof}

    def reset(self):
        self.reset_calls += 1

    def step(self, observation):
        from examples.advanced.openpi_policy.common import ActionChunk

        self.steps += 1
        rows = np.arange(self.horizon * self.dof, dtype=np.float32)
        rows = (rows.reshape(self.horizon, self.dof) % 1.0)
        rows[:, 0] = np.arange(self.horizon, dtype=np.float32)
        return ActionChunk(actions=rows, horizon=self.horizon, dof=self.dof,
                           source=self.source)


def test_a_flow_policy_runs_as_a_study_method():
    from examples.advanced.robocasa.study_flow_method import FlowMethod

    flow = _FakeFlow()
    method = FlowMethod(policy=flow)
    assert flow.reset_calls == 1  # reset once at construction, not per predict
    chunk = method.predict(_observation())
    assert chunk.actions.shape == (10, 7)
    assert chunk.horizon == 10 and chunk.dof == 7
    assert chunk.source == "mock"
    assert flow.reset_calls == 1


def test_take_truncates_a_chunk_the_way_the_official_client_does():
    """`action_plan.extend(action_chunk[: replan_steps])` — the rest is dropped."""
    from examples.advanced.robocasa.study_flow_method import FlowMethod

    method = FlowMethod(policy=_FakeFlow(horizon=50), take=5)
    chunk = method.predict(_observation())
    assert chunk.actions.shape == (5, 7)
    assert chunk.horizon == 5
    # Rows 0-4 of the proposal, in order; rows 5-49 discarded.
    assert chunk.actions[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert method.rows_dispatched == 5
    assert method.rows_discarded == 45


def test_an_untruncated_chunk_keeps_the_policys_own_horizon():
    from examples.advanced.robocasa.study_flow_method import FlowMethod

    method = FlowMethod(policy=_FakeFlow(horizon=32))
    assert method.predict(_observation()).horizon == 32
    assert method.rows_discarded == 0


def test_the_adapter_records_the_policys_identity_and_config():
    """A cell reproducible only from live Python objects is not reproducible."""
    from examples.advanced.robocasa.study_flow_method import FlowMethod

    described = FlowMethod(policy=_FakeFlow(horizon=4, dof=8), take=2).describe()
    assert described["policy_ref"].endswith(":_FakeFlow")
    assert described["policy_config"] == {"horizon": 4, "dof": 8}
    assert described["take"] == 2


def test_a_policy_that_cannot_describe_itself_is_refused():
    from examples.advanced.robocasa.study_flow_method import FlowContractError, FlowMethod

    class Anonymous:
        def step(self, observation):
            return None

    with pytest.raises(FlowContractError, match="init_config"):
        FlowMethod(policy=Anonymous())


def test_a_non_flow_is_refused():
    from examples.advanced.robocasa.study_flow_method import FlowContractError, FlowMethod

    class NotAFlow:
        def init_config(self):
            return {}

    with pytest.raises(FlowContractError, match="no step"):
        FlowMethod(policy=NotAFlow())


def test_a_misshaped_chunk_from_a_flow_is_refused():
    from examples.advanced.openpi_policy.common import ActionChunk
    from examples.advanced.robocasa.study_flow_method import FlowContractError, FlowMethod

    class Broken(_FakeFlow):
        def step(self, observation):
            return ActionChunk(actions=np.zeros(7, dtype=np.float32), horizon=1,
                               dof=7, source="mock")

    method = FlowMethod(policy=Broken())
    with pytest.raises(FlowContractError, match=r"\(horizon, dof\)"):
        method.predict(_observation())


def test_the_adapter_does_not_relabel_a_policys_source_by_default():
    """A method must not be able to launder its own provenance."""
    from examples.advanced.robocasa.study_flow_method import FlowMethod

    assert FlowMethod(policy=_FakeFlow(source="mock")).predict(
        _observation()).source == "mock"
    assert FlowMethod(policy=_FakeFlow(source="mock"),
                      source="model").predict(_observation()).source == "model"


@pytest.mark.parametrize("take", [0, 65, 5.0])
def test_an_unusable_take_is_rejected(take):
    from examples.advanced.robocasa.study_flow_method import FlowMethod

    with pytest.raises(ValueError, match="take"):
        FlowMethod(policy=_FakeFlow(), take=take)


def test_the_real_openpi_mock_policy_works_through_the_bridge():
    """The lane's own CI policy, unmodified, as a study arm."""
    from examples.advanced.openpi_policy.common import MockPi05Policy
    from examples.advanced.robocasa.method_harness import SafetyEnvelope
    from examples.advanced.robocasa.study_flow_method import FlowMethod

    # A bounded state: this mock scales its output straight off the state, so
    # arange(7) would hand the envelope actions of magnitude 6.
    observation = policy_observation(frames=[[_frame()]],
                                     state=np.full(7, 0.2, dtype=np.float32),
                                     prompt="pick the cube")
    method = FlowMethod(policy=MockPi05Policy(), take=5)
    chunk = method.predict(observation)
    assert chunk.actions.shape == (5, 7)
    assert method.describe()["policy_ref"] == (
        "examples.advanced.openpi_policy.common:MockPi05Policy")
    assert method.describe()["policy_config"] == {"horizon": 10, "dof": 7}
    SafetyEnvelope(allowed_sources=frozenset({"mock"})).validate(chunk)


def test_the_reported_dof_is_measured_not_claimed():
    """A policy's declared width and its emitted width can disagree.

    `MockPi05Policy.step` slices `base[None, : self.dof]`, so a state narrower
    than `dof` yields narrower actions while the chunk still declares the wider
    `dof`. The envelope and the scorer must see the array that will actually be
    dispatched, so the adapter recomputes the width rather than trusting the
    label.
    """
    from examples.advanced.openpi_policy.common import MockPi05Policy
    from examples.advanced.robocasa.study_flow_method import FlowMethod

    method = FlowMethod(policy=MockPi05Policy(dof=7))
    chunk = method.predict(_observation(state=5))
    assert chunk.actions.shape[1] == 5     # what will be dispatched
    assert chunk.dof == 5                  # not the policy's claim of 7
    assert method.describe()["policy_config"]["dof"] == 7


def test_per_task_horizons_override_the_flat_cap(tmp_path):
    """The official protocol gives each task its own budget."""
    from examples.advanced.robocasa.study_sweep import Sweep

    seen = {}

    class _Recording(_FakeEnvironment):
        def reset(self, request):
            seen[self.task] = request.max_steps
            return super().reset(request)

    sweep = Sweep(
        build_environment=lambda task, representation: _Recording(
            task=task, representation=representation),
        tasks=("T", "U"), seeds=(0,), max_steps=400,
        horizons={"T": 750})
    summary = sweep.run(_cell_specs())
    assert seen == {"T": 750, "U": 400}   # named task overridden, other falls back
    assert summary["protocol"]["horizons"] == {"T": 750}


def test_the_official_horizons_are_per_task_and_in_range():
    """Read from robocasa's registry, which is where the official client reads."""
    from examples.advanced.robocasa.study_tasks import ATOMIC_SEEN_TASKS

    pytest.importorskip("robocasa")
    from examples.advanced.robocasa.study_tasks import official_horizon

    values = {t: official_horizon(t) for t in ATOMIC_SEEN_TASKS}
    assert all(450 <= v <= 1050 for v in values.values())
    # Not a flat cap - a single number would starve the long tasks.
    assert len(set(values.values())) > 1


# --- the prompt is part of the interface ---------------------------------
#
# Every failure this section guards against was found by running a sweep, not
# by a test: a prompt whose example was 7 wide beside a stated dof of 10, and a
# prompt stating [-1, 1] when the codec's cap was 0.02 m. Both produced valid,
# parseable, entirely useless actions — the shape of bug that looks like a weak
# policy. The prompt is as much a part of the policy/environment contract as the
# action width, and it is asserted here against the codec rather than by eye.


@pytest.mark.parametrize("dof", [7, 8, 10])
def test_the_prompt_example_has_exactly_the_stated_width(dof):
    """A model answers with the example's width when the two disagree."""
    import json as _json

    from examples.advanced.robocasa.study_model import example_rows

    rows = _json.loads(example_rows(dof))
    assert rows, "an example with no rows teaches nothing"
    assert all(len(row) == dof for row in rows), [len(r) for r in rows]


@pytest.mark.parametrize("variant", ["with_format", "with_layout"])
@pytest.mark.parametrize("dof", [7, 8, 10])
def test_every_rendered_prompt_contains_a_correctly_sized_example(variant, dof):
    import json as _json
    import re

    from examples.advanced.robocasa.study_model import ModelConfig

    text = ModelConfig(model_id="m", prompt_variant=variant).instructions(dof=dof)
    found = re.search(r"\[\[.*?\]\]", text, re.S)
    assert found, "the prompt shows no example at all"
    for row in _json.loads(found.group(0)):
        assert len(row) == dof, f"{variant} example row is {len(row)} wide, stated {dof}"


@pytest.mark.parametrize("dof", [7, 10])
def test_the_prompt_states_the_codecs_real_limits(dof):
    """The numbers in the prompt must be the codec's, not a second copy.

    Measured consequence of them disagreeing: requests exceeding the cap by
    0.84 m, 75-93% of rows clipped, and zero successes on all eighteen tasks.
    """
    from examples.advanced.robocasa.study_model import layout_text
    from examples.advanced.robocasa.study_representations import (
        MAX_ROTATION_PER_STEP, MAX_TRANSLATION_PER_STEP)

    text = layout_text(dof)
    assert str(MAX_TRANSLATION_PER_STEP) in text
    assert str(MAX_ROTATION_PER_STEP) in text
    assert "metre" in text and "radian" in text


def test_the_prompt_example_is_inside_the_codecs_limits():
    """An example above the cap teaches a magnitude that gets clipped away."""
    import json as _json

    import numpy as _np

    from examples.advanced.robocasa.study_model import example_rows
    from examples.advanced.robocasa.study_representations import (
        MAX_TRANSLATION_PER_STEP)

    for row in _json.loads(example_rows(10)):
        assert _np.linalg.norm(row[:3]) <= MAX_TRANSLATION_PER_STEP + 1e-9


def test_an_undescribed_action_width_refuses_rather_than_prompting_vaguely():
    """A prompt that cannot describe the action space must not be sent."""
    from examples.advanced.robocasa.study_model import layout_text

    with pytest.raises(ValueError, match="no action-space description"):
        layout_text(12)


def test_a_prompt_at_the_codecs_scale_does_not_saturate_the_clip_ledger():
    """The canary: a correctly-scaled action passes the codec unclipped.

    If the prompt teaches the right magnitude, a policy following it is not
    clipped. This pins the direction of the relationship, so a future prompt
    regression shows up here rather than as a 75-minute sweep scoring zero.
    """
    import json as _json

    import numpy as _np

    from examples.advanced.robocasa.study_model import example_rows
    from examples.advanced.robocasa.study_representations import EefDeltaCodec

    codec = EefDeltaCodec("eef_delta_base")
    for row in _json.loads(example_rows(10)):
        out = codec.to_delta(_np.asarray(row, dtype=_np.float32),
                             current_position=_np.zeros(3),
                             current_rotation=_np.zeros(3))
        assert out[:3].tolist() == pytest.approx(row[:3], abs=1e-6), "clipped"
    assert codec.ledger.to_dict()["clipped_rows"] == 0


def test_a_full_scale_action_does_saturate_and_is_visible():
    """The other half: the ledger must actually notice over-scale requests.

    Without this the canary above could pass against a ledger that never
    reports anything.
    """
    import numpy as _np

    from examples.advanced.robocasa.study_representations import EefDeltaCodec

    codec = EefDeltaCodec("eef_delta_base")
    row = _np.zeros(10, dtype=_np.float32)
    row[0] = 1.0
    codec.to_delta(row, current_position=_np.zeros(3), current_rotation=_np.zeros(3))
    summary = codec.ledger.to_dict()
    assert summary["clipped_rows"] == 1
    assert summary["max_translation_excess_m"] > 0.9


def test_the_model_arms_dof_matches_its_representation():
    """The arm's dof and its representation's width must be the same number."""
    from examples.advanced.robocasa.study_representations import REPRESENTATION_DOF

    # The app builds the Agentic arm as eef_delta_base with dof=10.
    assert REPRESENTATION_DOF["eef_delta_base"] == 10


def test_a_cell_records_what_the_method_reports_about_itself():
    """A zero score and a run of held no-motion actions look identical without this."""
    from examples.advanced.robocasa.study_sweep import MethodSpec, Sweep

    class _Reporting(StillMethod):
        refusals = 7
        last_refusal = "model call ended timeout"

        def describe(self):
            return {"method": "reporting", "refusals": self.refusals,
                    "nested": {"dropped": True}}

    sweep = Sweep(build_environment=_build_fake, tasks=("T",), seeds=(0,))
    summary = sweep.run((MethodSpec(method_id="m", horizon=2,
                                    build=lambda env: _Reporting(horizon=2)),))
    recorded = summary["episodes"][0]["method"]
    assert recorded["refusals"] == 7
    assert recorded["last_refusal"] == "model call ended timeout"
    assert recorded["method"] == "reporting"
    # Only scalars survive, so a record stays JSON-safe and byte-stable.
    assert "nested" not in recorded


def test_a_methods_broken_describe_does_not_lose_the_cell():
    """Reporting is not worth an episode."""
    from examples.advanced.robocasa.study_sweep import MethodSpec, Sweep

    class _Broken(StillMethod):
        def describe(self):
            raise RuntimeError("no")

    sweep = Sweep(build_environment=_build_fake, tasks=("T",), seeds=(0,))
    summary = sweep.run((MethodSpec(method_id="m", horizon=2,
                                    build=lambda env: _Broken(horizon=2)),))
    assert summary["errors"] == 0
    assert summary["episodes"][0]["method"] == {}


# --- preflight: catch an interface disagreement before paying for it ------


def test_preflight_catches_a_prompt_that_omits_the_action_scale():
    """The exact failure that cost 75 minutes and scored zero on 18 tasks.

    `with_format` states the envelope bound and not the codec's limits, so a
    model following it asks for a metre where the cap is two centimetres. This
    must fail the check; `with_layout`, which states the limits, must pass it.
    """
    from examples.advanced.robocasa.study_model import ModelConfig
    from examples.advanced.robocasa.study_preflight import Preflight, check_prompt_contract

    lax = Preflight()
    check_prompt_contract(lax, config=ModelConfig(model_id="m",
                                                  prompt_variant="with_format"), dof=10)
    assert not lax.ok
    assert "prompt states the codec's per-step limits" in lax.to_dict()["failed"]

    strict = Preflight()
    check_prompt_contract(strict, config=ModelConfig(model_id="m",
                                                     prompt_variant="with_layout"), dof=10)
    assert strict.ok, strict.to_dict()["failed"]


def test_preflight_catches_a_width_mismatch():
    from examples.advanced.robocasa.study_preflight import Preflight, check_action_width

    bad = Preflight()
    check_action_width(bad, representation="eef_delta_base", dof=7)
    assert not bad.ok
    good = Preflight()
    check_action_width(good, representation="eef_delta_base", dof=10)
    assert good.ok


def test_preflight_catches_a_blind_policy():
    """A blank frame is what a policy handed a placeholder actually receives."""
    from examples.advanced.robocasa.study_preflight import Preflight, check_observation

    class _Obs:
        def __init__(self, image, state, prompt):
            self.image, self.state, self.prompt = image, state, prompt

    blank = Preflight()
    check_observation(blank, observation=_Obs(np.zeros((8, 8, 3), np.uint8),
                                              np.zeros(4), "go"))
    assert "image is not blank" in blank.to_dict()["failed"]

    real = Preflight()
    check_observation(real, observation=_Obs(
        (np.random.default_rng(0).random((8, 8, 3)) * 255).astype(np.uint8),
        np.zeros(4), "go"))
    assert real.ok, real.to_dict()["failed"]


def test_preflight_catches_a_saturated_policy():
    """75-93% clipped was the measured signature of the scale bug."""
    from examples.advanced.robocasa.study_preflight import Preflight, check_saturation

    hot = Preflight()
    check_saturation(hot, clip={"clipped_fraction": 0.83, "max_translation_excess_m": 0.84})
    assert not hot.ok
    cool = Preflight()
    check_saturation(cool, clip={"clipped_fraction": 0.0, "max_translation_excess_m": 0.0})
    assert cool.ok
    # A ledger that reports nothing must not read as a pass.
    silent = Preflight()
    check_saturation(silent, clip={})
    assert not silent.ok


def test_preflight_catches_a_policy_that_mostly_refuses():
    """A refusal becomes a held no-motion action, indistinguishable from failure."""
    from examples.advanced.robocasa.study_preflight import Preflight, check_calls

    class _Method:
        def __init__(self, calls, refusals):
            self.calls, self.refusals = calls, refusals

    bad = Preflight()
    check_calls(bad, method=_Method(10, 9), steps=20)
    assert not bad.ok
    good = Preflight()
    check_calls(good, method=_Method(10, 1), steps=20)
    assert good.ok
    never = Preflight()
    check_calls(never, method=_Method(0, 0), steps=20)
    assert "the policy was called" in never.to_dict()["failed"]


def test_a_preflight_report_names_what_failed():
    from examples.advanced.robocasa.study_preflight import Preflight

    p = Preflight()
    p.note("a", True)
    p.note("b", False, "why")
    out = p.to_dict()
    assert out["ok"] is False
    assert out["failed"] == ["b"]
    assert out["findings"][1]["detail"] == "why"


def test_preflight_finds_the_call_count_wherever_it_lives():
    """A Flow adapter counts on itself; a model method counts on its client.

    Reading the wrong attribute reports a working policy as never called, which
    is exactly what happened the first time this check ran live.
    """
    from examples.advanced.robocasa.study_preflight import Preflight, check_calls

    class _OnClient:
        class client:  # noqa: N801 - a stand-in for the provider client
            calls, refusals = 12, 0

    on_client = Preflight()
    check_calls(on_client, method=_OnClient(), steps=12)
    assert on_client.ok, on_client.to_dict()["failed"]

    class _OnLedger:
        class ledger:  # noqa: N801
            canonical = [{}] * 9

    on_ledger = Preflight()
    check_calls(on_ledger, method=_OnLedger(), steps=12)
    assert on_ledger.ok, on_ledger.to_dict()["failed"]


def test_a_commanded_delta_reaches_the_arm_in_controller_units():
    """A metre delta must be scaled into the controller's normalised range.

    The operational-space controller takes [-1, 1] and multiplies by its own
    `output_max`, which is 0.05 m per step here. Writing the codec's metre value
    straight into the action vector therefore understated every command fifty
    times over: forty steps of a full-cap 0.02 m — 0.800 m asked for — moved the
    end-effector 0.009 m, and every end-effector-representation arm was scored
    against a robot that could not reach anything. The regression is silent in
    the records, because an episode of ineffective actions and an episode of
    bad actions both end with the predicate unmet and no rung credited.

    Asserted on the scaling rather than on a displacement, so it needs no
    simulator step budget and cannot flake on contact dynamics.
    """
    pytest.importorskip("robocasa")
    import numpy as np

    from examples.advanced.robocasa.study_app import model_observation
    from examples.advanced.robocasa.study_representations import (
        MAX_ROTATION_PER_STEP, MAX_TRANSLATION_PER_STEP)
    from examples.advanced.robocasa.study_robocasa_env import RoboCasaPnPEnvironment

    class Request:
        task = "PickPlaceCounterToStove"
        seed = 0
        episode = 0
        max_steps = 5
        method_id = "scaling"

    environment = RoboCasaPnPEnvironment(
        task=Request.task, representation="eef_delta_base", command_base=True,
        observation=model_observation())
    try:
        environment.reset(Request())
        scale = environment._arm_output_max()
        assert scale.shape == (6,) and np.all(scale > 0)

        # A full-cap translation must arrive as a sizeable fraction of the
        # controller's range, not as the raw metre value.
        delta = np.zeros(6, dtype=np.float32)
        delta[0] = MAX_TRANSLATION_PER_STEP
        normalised = environment._normalise_arm(delta)
        assert normalised[0] == pytest.approx(MAX_TRANSLATION_PER_STEP / scale[0])
        assert normalised[0] > 10 * MAX_TRANSLATION_PER_STEP, (
            "the metre value is reaching the controller unscaled")

        rotation = np.zeros(6, dtype=np.float32)
        rotation[3] = MAX_ROTATION_PER_STEP
        assert environment._normalise_arm(rotation)[3] == pytest.approx(
            MAX_ROTATION_PER_STEP / scale[3])

        # And nothing may leave the controller's admissible range.
        assert np.all(np.abs(environment._normalise_arm(np.full(6, 10.0))) <= 1.0)
    finally:
        environment.close()
