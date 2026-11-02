"""A released vision-language-action policy as a study method.

Every method in this study so far was written here: two baselines that score
zero and a privileged scripted reference that scores 75. Neither tells you what
a *trained* policy does on these tasks, and that is the number the study needs
to be read against. `lerobot/smolvla_robocasa` is a SmolVLA checkpoint
fine-tuned on `lerobot/robocasa_target_human_unified` — 25,307 episodes over 50
RoboCasa tasks — and five of its tasks are in this study's committed suite.

**Why it fits without reinterpretation.** The policy emits a 12-wide action and
this robot takes a 12-wide action, but the orderings differ, and feeding one to
the other unpermuted would look like it was working while commanding the base
with end-effector deltas. The mapping below is not taken on trust from a
README: it was derived from the dataset's own per-dimension statistics and then
cross-checked against this environment's action slices.

The giveaways were the degenerate dimensions. In 15 million frames, action
dimension 3 is *exactly* zero — a frozen torso. Dimensions 4 and 11 are exactly
binary: a ±1 variable satisfies `std = sqrt(1 - mean**2)`, and dimension 4's
measured −0.780 / 0.626 and dimension 11's −0.172 / 0.985 both match to three
decimals, which no continuous signal would. That identifies the mode flag and
the gripper, and the remaining translation/rotation split falls out of the
magnitudes: dimensions 5–7 have standard deviations near 0.37 against 0.11 for
8–10, and a position delta moves more than a rotation delta.

The state vector was settled the same way and then confirmed by physics: the
16 values are this robot's `base_pos`, `base_quat`, `eef_pos`, `eef_quat` and
`gripper_qpos` concatenated in that order, which is why state dimensions 3 and 4
are identically zero (a base that only yaws has no quaternion x or y) and
dimension 2 sits at 0.70 with a standard deviation of 0.002 (a base of fixed
height).

**What this measures and what it does not.** This is a comparable number on the
same task, not a published one: the checkpoint's model card reports no scores,
so what comes out is our measurement of a public policy under our scorer. It is
honest about being a model — `source` is `"model"`, not `"scripted"` — so the
safety envelope has to be widened deliberately to admit it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from examples.advanced.openpi_policy.common import ActionChunk, PolicyObservation

#: The checkpoint this module is written against.
DEFAULT_CHECKPOINT = "lerobot/smolvla_robocasa"
#: The dataset it was fine-tuned on, recorded so a run says what it faced.
TRAINING_DATASET = "lerobot/robocasa_target_human_unified"
#: Camera views the policy was trained with, in the order its config names them.
#: `agentview_right` is a RoboCasa-only role, so an observation configured for
#: this policy reports `portable: False` — DROID has no second agentview.
LEROBOT_VIEWS = ("over_shoulder", "agentview_right", "wrist")
#: The feature keys the *model* declares, in view order.
#:
#: The released preprocessor renames the dataset's `robot0_*` keys to these, and
#: the pinned lerobot has no such step, so the batch is built under the model's
#: own names. The rename map in the checkpoint fixes the order and it is not a
#: guess: `robot0_agentview_left` is camera1, `robot0_agentview_right` is
#: camera2, `robot0_eye_in_hand` is camera3.
LEROBOT_IMAGE_KEYS = ("observation.images.camera1",
                      "observation.images.camera2",
                      "observation.images.camera3")
#: The dataset keys those correspond to, in the same order.
LEROBOT_DATASET_IMAGE_KEYS = ("observation.images.robot0_agentview_left",
                              "observation.images.robot0_agentview_right",
                              "observation.images.robot0_eye_in_hand")
#: Robosuite observables the 16-wide state is built from, in order.
#:
#: The end-effector pose is **not** used as robosuite reports it. The checkpoint
#: family documents this vector as "base pose (position 3 + quaternion 4), EE
#: pose *in base frame* (position 3 + quaternion 4), gripper finger positions
#: (2)", and robosuite's `robot0_eef_pos` is in world coordinates. The
#: distinction is not cosmetic and the dataset's own statistics settle it: state
#: dimensions 7-9 have mean (0.287, -0.021, 0.483), while at reset this robot's
#: world-frame hand is at (0.658, -0.575, 1.312) and its base-frame hand is at
#: (0.240, -0.009, 0.612). Feeding the world frame would put the policy's
#: proprioception in a space it never saw, and nothing would raise.
LEROBOT_STATE_KEYS = ("robot0_base_pos", "robot0_base_quat", "robot0_eef_pos",
                      "robot0_eef_quat", "robot0_gripper_qpos")
#: Width of that state, asserted rather than assumed.
LEROBOT_STATE_DOF = 16
#: Width of one action row.
LEROBOT_ACTION_DOF = 12
#: Native image side the checkpoint's config declares.
LEROBOT_IMAGE_SIZE = 256
#: Actions taken from each predicted chunk before re-planning.
#:
#: This is the RoboCasa365 leaderboard's own value, read from the official
#: evaluation client (`robocasa-benchmark/openpi`,
#: `examples/robocasa/main.py`): `replan_steps: int = 5`, applied as
#: `action_plan.extend(action_chunk[: replan_steps])`. The remainder of the
#: chunk is discarded. A checkpoint that predicts 50 steps is therefore
#: evaluated closed-loop at every fifth step, not open-loop across all fifty,
#: and matching that is a condition of comparability rather than a tuning
#: choice.
REPLAN_STEPS = 5
#: Whether to flip rendered frames vertically before the policy sees them.
#:
#: Robosuite's `IMAGE_CONVENTION` defaults to `opengl`, which returns frames
#: bottom-up; the dataset was converted under the opencv convention. Settled
#: against a decoded training frame rather than assumed: that frame's
#: bottom quarter is brighter than its top (138.1 against 86.1) while ours is
#: the reverse (73.6 against 116.8), and the row-brightness profile correlates
#: -0.120 as rendered against +0.570 flipped.
FLIP_IMAGES = True

#: Where each native action element comes from in a LeRobot row.
#:
#: Native layout is `eef_pos(3) | eef_rot(3) | gripper | base(3) | torso | mode`
#: and the LeRobot row is `base(3) | torso | mode | eef_pos(3) | eef_rot(3) |
#: gripper`, so this permutation is the whole conversion.
NATIVE_FROM_LEROBOT = (5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4)
#: Dimensions the training data never moved. Commanding them would be acting
#: outside the distribution the policy was fitted on, so a run reports whether
#: the policy tried.
FROZEN_LEROBOT_DIMS = (3,)


#: Config fields the released checkpoint carries that this pinned lerobot does
#: not accept — it was saved by a newer version. Each was checked to be inert
#: for inference before being dropped: `use_peft` is false so there are no
#: adapters folded into the weights, `rtc_config` is null, `compile_model` is
#: false (which makes `compile_mode` moot), and `pretrained_path` records
#: provenance. Dropping an *architecture* field would silently build the wrong
#: model, so this list is narrow on purpose and the loader refuses anything it
#: has not been told about.
INERT_CONFIG_FIELDS = ("use_peft", "pretrained_path", "rtc_config",
                       "compile_model", "compile_mode")


class VlaUnavailable(RuntimeError):
    """The checkpoint or its dependencies could not be loaded."""


#: Which normalisation statistic feeds which buffer inside the policy.
#:
#: The released checkpoint keeps these in a separate processor safetensors — the
#: newer lerobot's pipeline format — so loading the model alone leaves the
#: normaliser at its uninitialised sentinel. State and action are MEAN_STD
#: normalised, which means an uninitialised normaliser does not fail loudly: it
#: produces plausible-looking nonsense. Images are IDENTITY normalised and so
#: need no statistics, which is why only two features appear here.
NORMALIZATION_SOURCES = (
    ("normalize_inputs", "buffer_observation_state.mean", "observation.state.mean"),
    ("normalize_inputs", "buffer_observation_state.std", "observation.state.std"),
    ("unnormalize_outputs", "buffer_action.mean", "action.mean"),
    ("unnormalize_outputs", "buffer_action.std", "action.std"),
)


def install_normalization(policy, directory) -> dict:
    """Load the released normalisation statistics into `policy`.

    Returns what was installed, so a run can show the normaliser was populated
    rather than leaving it to be assumed. Raises if any statistic is missing,
    misshaped or non-finite — all three would otherwise surface as a policy
    that runs and is simply wrong.
    """
    from pathlib import Path

    import torch
    from safetensors.torch import load_file

    stats: dict = {}
    for path in sorted(Path(directory).glob("*processor*.safetensors")):
        stats.update(load_file(str(path)))
    if not stats:
        raise VlaUnavailable(f"no processor statistics found in {directory}")
    installed = {}
    for module_name, buffer_key, stat_key in NORMALIZATION_SOURCES:
        module = getattr(policy, module_name, None)
        if module is None:
            raise VlaUnavailable(f"policy has no {module_name}")
        current = module.state_dict()
        if buffer_key not in current:
            raise VlaUnavailable(f"{module_name} has no {buffer_key}")
        if stat_key not in stats:
            raise VlaUnavailable(f"statistics are missing {stat_key}")
        value = stats[stat_key].reshape(current[buffer_key].shape).to(
            current[buffer_key].dtype)
        if not bool(torch.isfinite(value).all()):
            raise VlaUnavailable(f"{stat_key} is not finite")
        module.load_state_dict({buffer_key: value}, strict=False)
        installed[f"{module_name}.{buffer_key}"] = stat_key
    for module_name, buffer_key, _ in NORMALIZATION_SOURCES:
        tensor = getattr(policy, module_name).state_dict()[buffer_key]
        if not bool(torch.isfinite(tensor).all()):
            raise VlaUnavailable(
                f"{module_name}.{buffer_key} is still uninitialised after loading")
    return installed


def prepare_checkpoint(checkpoint: str = DEFAULT_CHECKPOINT, *, cache_dir=None) -> str:
    """Fetch `checkpoint` and drop only the config fields known to be inert.

    Returns a local directory the pinned lerobot can load. The rewrite is
    recorded in the returned directory so a run can show what was changed.
    """
    import json
    import shutil
    from pathlib import Path

    from huggingface_hub import snapshot_download

    source = Path(snapshot_download(checkpoint))
    target = Path(cache_dir) if cache_dir else source.parent / f"{source.name}-compat"
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.is_file() and not (target / item.name).exists():
            shutil.copy2(item, target / item.name)
    config = json.loads((source / "config.json").read_text())
    dropped = {key: config.pop(key) for key in INERT_CONFIG_FIELDS if key in config}
    (target / "config.json").write_text(json.dumps(config, indent=2))
    (target / "retriever_compat.json").write_text(
        json.dumps({"checkpoint": checkpoint, "dropped": dropped}, indent=2, sort_keys=True))
    return str(target)


def to_native_action(row) -> np.ndarray:
    """Permute one LeRobot action row into this environment's ordering."""
    values = np.asarray(row, dtype=np.float32)
    if values.shape != (LEROBOT_ACTION_DOF,):
        raise ValueError(f"a lerobot action row must be {LEROBOT_ACTION_DOF} wide")
    if not np.isfinite(values).all():
        raise ValueError("a lerobot action row contains a non-finite value")
    return values[list(NATIVE_FROM_LEROBOT)]


def _rotation(quaternion) -> np.ndarray:
    """Rotation matrix for an `(x, y, z, w)` quaternion.

    Written out rather than imported so the state assembly stays testable
    without a simulator; cross-checked against robosuite's own
    `transform_utils` in the simulator-backed suite.
    """
    x, y, z, w = (float(v) for v in quaternion)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _relative(base, hand) -> np.ndarray:
    """The rotation taking the base frame to the hand frame, `(x, y, z, w)`."""
    bx, by, bz, bw = (float(v) for v in base)
    # Inverse of a unit quaternion is its conjugate.
    bx, by, bz = -bx, -by, -bz
    hx, hy, hz, hw = (float(v) for v in hand)
    return np.array([
        bw * hx + bx * hw + by * hz - bz * hy,
        bw * hy - bx * hz + by * hw + bz * hx,
        bw * hz + bx * hy - by * hx + bz * hw,
        bw * hw - bx * hx - by * hy - bz * hz,
    ])


def _canonical(quaternion) -> np.ndarray:
    """Flip a quaternion to the `w >= 0` half, the convention in the data.

    Measured: state dimension 13, the relative rotation's `w`, spans 0 to 0.999
    across 15 million frames and never goes negative. A quaternion and its
    negation are the same rotation, so without this the policy would sometimes
    receive the numerically opposite encoding of a pose it knows.
    """
    values = np.asarray(quaternion, dtype=np.float64)
    return -values if values[3] < 0.0 else values


def lerobot_state(raw) -> np.ndarray:
    """Build the policy's 16-wide state from this robot's observables.

    The end-effector pose is converted into the base frame, which is what the
    checkpoint was trained on; see `LEROBOT_STATE_KEYS` for how that was
    established.
    """
    for key in LEROBOT_STATE_KEYS:
        if key not in raw:
            raise KeyError(f"observation is missing {key!r}")
    base_position = np.asarray(raw["robot0_base_pos"], dtype=np.float64).reshape(-1)
    base_quaternion = _canonical(
        np.asarray(raw["robot0_base_quat"], dtype=np.float64).reshape(-1))
    hand_position = np.asarray(raw["robot0_eef_pos"], dtype=np.float64).reshape(-1)
    hand_quaternion = np.asarray(raw["robot0_eef_quat"], dtype=np.float64).reshape(-1)
    gripper = np.asarray(raw["robot0_gripper_qpos"], dtype=np.float64).reshape(-1)
    if base_quaternion.shape != (4,) or hand_quaternion.shape != (4,):
        raise ValueError("base and hand quaternions must each be 4 wide")
    relative_position = _rotation(base_quaternion).T @ (hand_position - base_position)
    relative_quaternion = _canonical(_relative(base_quaternion, hand_quaternion))
    state = np.concatenate([base_position, base_quaternion, relative_position,
                            relative_quaternion, gripper]).astype(np.float32)
    if state.shape != (LEROBOT_STATE_DOF,):
        raise ValueError(f"state assembled to {state.shape}, expected "
                         f"({LEROBOT_STATE_DOF},)")
    if not np.isfinite(state).all():
        raise ValueError("assembled state contains a non-finite value")
    return state


def untile(image, *, views: int = len(LEROBOT_VIEWS)) -> list:
    """Recover the separate camera views from one tiled observation image.

    `tile_frames` lays views out along the width, so slicing the width into
    equal parts inverts it exactly. Going through the tiled image rather than
    around it keeps the policy on the same observation contract every other
    method uses, instead of reaching into the environment for raw frames.
    """
    frame = np.asarray(image)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("a tiled observation must be (H, W, 3)")
    if type(views) is not int or views < 1:
        raise ValueError("views must be a positive int")
    height, width = frame.shape[:2]
    if width % views:
        raise ValueError(f"a width of {width} does not divide into {views} views")
    side = width // views
    if side != height:
        raise ValueError(f"each view must be square, got {height}x{side}")
    return [np.ascontiguousarray(frame[:, i * side:(i + 1) * side]) for i in range(views)]


@dataclass
class SmolVlaPolicy:
    """`lerobot/smolvla_robocasa` as a `Flow[PolicyObservation, ActionChunk]`.

    This is the shape Retriever expects of a policy, so the study measures a
    policy written against the framework's contract rather than a class that
    happens to live here. `step` returns the checkpoint's **whole** proposal —
    `n_action_steps` is 50 in the released config — and truncating it to the
    benchmark's re-planning cadence is the adapter's job, not the policy's. See
    `study_flow_method.FlowMethod`.

    Keeping the two apart is what makes the study's model-call count correct.
    With the chunk held in a private queue and one row handed out per call, the
    harness saw one dispatch per control step and reported 1,130 "model calls"
    for an episode in which the policy was queried 23 times. The paper's metric
    is calls per episode; this returns the real one.
    """

    checkpoint: str = DEFAULT_CHECKPOINT
    source: str = "model"
    device: str = "cpu"
    #: Flip frames to the convention the checkpoint was trained under.
    flip_images: bool = FLIP_IMAGES
    policy: object = None
    calls: int = 0
    inferences: int = 0
    frozen_commands: int = 0
    #: Whether the checkpoint had to be loaded through the field-dropping
    #: shim above, so a run records that it was not loaded as published.
    compat_shim: bool = False
    #: Which released statistic went into which normaliser buffer, recorded so
    #: a run shows the normaliser was populated rather than assumed.
    normalization: dict = field(default_factory=dict)
    #: Seed for the sampling this policy does at inference time.
    #:
    #: SmolVLA is a flow-matching policy: `predict_action_chunk` integrates from
    #: sampled noise, so two calls on one observation return *different* chunks.
    #: Measured on a fixed observation, the same build on the same machine gave
    #: chunk sums of -108.6 and -115.5 on consecutive calls. Nothing seeded it,
    #: so the episode seed reached the simulator's scene but not the policy, and
    #: an episode was not reproducible even though the environment was. That also
    #: makes a port impossible to validate: a CPU and a GPU run would differ for
    #: two reasons at once, with no way to tell which.
    seed: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint, str) or not self.checkpoint.strip():
            raise ValueError("checkpoint must be a non-empty string")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty string")

    def init_config(self) -> dict:
        """Enough to rebuild this policy from a serialised run record."""
        return {"checkpoint": self.checkpoint, "device": self.device,
                "flip_images": self.flip_images, "seed": self.seed}

    def load(self) -> None:
        """Build the policy. Separate from `__init__` so import stays cheap."""
        if self.policy is not None:
            return
        try:
            import torch
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        except ImportError as error:  # pragma: no cover - environment-dependent
            raise VlaUnavailable(f"lerobot/transformers unavailable: {error}") from error
        try:
            policy = SmolVLAPolicy.from_pretrained(self.checkpoint)
        except Exception as direct:
            # The released config carries fields a newer lerobot writes. Retry
            # against a copy with only the inert ones removed, and say so.
            try:
                local = prepare_checkpoint(self.checkpoint)
                policy = SmolVLAPolicy.from_pretrained(local)
                self.compat_shim = True
            except Exception as error:  # pragma: no cover - network-dependent
                raise VlaUnavailable(
                    f"could not load {self.checkpoint}: {direct}; "
                    f"compatibility retry also failed: {error}") from error
        policy.eval()
        policy.to(torch.device(self.device))
        # Must happen after loading: the model file carries no statistics.
        self.normalization = install_normalization(
            policy, prepare_checkpoint(self.checkpoint))
        self.policy = policy

    def reset(self, seed: int | None = None) -> None:
        """Clear the policy's own state and re-seed its sampling.

        Takes an optional seed so a sweep can tie an episode's sampling to the
        episode, matching the scripted methods' `reset(seed)` convention. Seeds
        the global torch generator rather than threading a `Generator` through,
        because the sampling happens inside lerobot's own forward pass where
        there is nowhere to pass one.
        """
        if seed is not None:
            self.seed = int(seed)
        try:
            import torch
        except ImportError:  # pragma: no cover - environment-dependent
            pass
        else:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():  # pragma: no cover - device-dependent
                torch.cuda.manual_seed_all(self.seed)
        if self.policy is not None and hasattr(self.policy, "reset"):
            self.policy.reset()

    # --- the harness contract ------------------------------------------

    def step(self, observation: PolicyObservation) -> ActionChunk:
        """One inference, returning the checkpoint's whole action proposal."""
        self.calls += 1
        rows = self._infer(observation)
        actions = np.asarray(rows, dtype=np.float32)
        self.frozen_commands += int(
            np.count_nonzero(actions[:, list(FROZEN_LEROBOT_DIMS)]))
        return ActionChunk(actions=actions, horizon=int(actions.shape[0]),
                           dof=LEROBOT_ACTION_DOF, source=self.source)

    def _infer(self, observation: PolicyObservation) -> list:
        """One forward pass, returning the policy's whole action chunk."""
        import torch

        self.load()
        self.inferences += 1
        batch = self._batch(observation)
        with torch.no_grad():
            if hasattr(self.policy, "predict_action_chunk"):
                chunk = self.policy.predict_action_chunk(batch)
            else:  # pragma: no cover - depends on the lerobot version
                chunk = self.policy.select_action(batch).unsqueeze(1)
        values = np.asarray(chunk.squeeze(0).float().cpu().numpy(), dtype=np.float32)
        if values.ndim != 2 or values.shape[1] < LEROBOT_ACTION_DOF:
            raise ValueError(f"policy returned {values.shape}, expected "
                             f"(steps, >={LEROBOT_ACTION_DOF})")
        # The checkpoint pads to `max_action_dim`; only the declared width is
        # the robot's, and the padding is not ours to command.
        return [np.clip(row[:LEROBOT_ACTION_DOF], -1.0, 1.0) for row in values]

    def _batch(self, observation: PolicyObservation) -> dict:
        """Shape one observation the way the policy's preprocessor wants it."""
        import torch

        views = untile(observation.image)
        if self.flip_images:
            views = [np.ascontiguousarray(view[::-1]) for view in views]
        state = np.asarray(observation.state, dtype=np.float32)
        if state.shape != (LEROBOT_STATE_DOF,):
            raise ValueError(f"the policy needs a {LEROBOT_STATE_DOF}-wide state, "
                             f"got {state.shape}; is the environment in lerobot mode?")
        # Every tensor goes to the policy's device. `load` moves the module and
        # its normaliser buffers there, so leaving the inputs on the host raises
        # "Expected all tensors to be on the same device" from inside the
        # normaliser's subtraction — the first thing the forward pass does, and
        # a failure that reads as a lerobot bug rather than as a missing `.to()`
        # here. Harmless when the device is cpu, which is why running only on
        # CPU never revealed it.
        device = torch.device(self.device)
        batch = {"observation.state": torch.from_numpy(state).unsqueeze(0).to(device),
                 "task": [observation.prompt]}
        for key, view in zip(LEROBOT_IMAGE_KEYS, views):
            # Channels-first float in [0, 1], which is what the normaliser's
            # IDENTITY mapping for visual features assumes.
            image = torch.from_numpy(view.astype(np.float32) / 255.0)
            batch[key] = image.permute(2, 0, 1).unsqueeze(0).to(device)
        return batch

    def describe(self) -> dict:
        return {"method": "smolvla", "checkpoint": self.checkpoint,
                "training_dataset": TRAINING_DATASET, "privileged": False,
                "device": self.device, "flip_images": self.flip_images,
                "source": self.source, "calls": self.calls,
                "inferences": self.inferences,
                "frozen_commands": self.frozen_commands,
                "compat_shim": self.compat_shim,
                "normalization": dict(self.normalization)}


def smolvla_arm(*, take: int = REPLAN_STEPS, **kwargs):
    """The released checkpoint as a study arm, at the benchmark's cadence.

    `take` is the official client's `replan_steps`; pass `None` to let the
    policy's whole 50-step proposal run open-loop, which is what this study
    did before the protocol was read properly.
    """
    from .study_flow_method import FlowMethod

    return FlowMethod(policy=SmolVlaPolicy(**kwargs), take=take)
