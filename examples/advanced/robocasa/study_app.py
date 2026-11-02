"""Entry point for the RoboCasa pick-and-place policy study.

Prints one sorted JSON object and returns a meaningful exit code, following the
other runnable demos in this folder.

    pixi run -e robocasa study-pnp-baselines

The default sweep is the two baselines the harness has to be read against, over
the study's five fixed seeds. It is expected to report a zero success rate; the
numbers that matter from it are the control-step and model-call counts, the
partial-score floor, and the clipped-action fraction.
"""

from __future__ import annotations

import argparse

from .study_methods import RandomMethod, StillMethod
from .study_reference import ScriptedPickMethod
from .study_representations import REPRESENTATION_DOF
from .study_sweep import MethodSpec, Sweep, render
from .study_tasks import (
    ATOMIC_SEEN_TASKS, DEFAULT_SEEDS, ablation_subset, get_task, list_tasks,
    official_horizon, vla_comparable)


def baseline_specs(horizon: int, representation: str) -> tuple:
    return (
        MethodSpec(method_id="still", horizon=horizon, representation=representation,
                   label="no motion; the floor every method is read against",
                   build=lambda env, h=horizon, r=representation: StillMethod(
                       horizon=h, representation=r)),
        MethodSpec(method_id="random", horizon=horizon, representation=representation,
                   label="bounded seeded random motion",
                   build=lambda env, h=horizon, r=representation: RandomMethod(
                       horizon=h, representation=r, seed=0)),
    )


def reference_specs(horizon: int) -> tuple:
    """The scripted reference, which is privileged and drives joint deltas.

    It reads the object's pose from the simulator, so it is a ceiling to read
    policies against rather than an entry in the comparison. Its representation
    is fixed: the pipeline behind it commands joint targets, and asking for an
    end-effector representation would silently reinterpret its rows.
    """
    return (
        MethodSpec(method_id="scripted-pick", horizon=horizon, representation="joint_delta",
                   label="privileged reach, grasp and lift; the ceiling, not a policy",
                   build=lambda env, h=horizon: ScriptedPickMethod(environment=env, horizon=h)),
    )


def vla_specs(horizon: int, *, take: int | None = None, device: str = "cpu") -> tuple:
    """A released RoboCasa VLA checkpoint, in its own representation.

    Its observation and action spaces are fixed by the checkpoint — three views
    at 256 px, a 16-wide state, a 12-wide action in its own ordering — so none
    of them follow `--representation`.
    """
    from .study_vla import REPLAN_STEPS, smolvla_arm

    # `horizon` does not apply: the checkpoint fixes its own 50-step proposal
    # and the benchmark fixes how much of it runs before re-planning.
    chosen = REPLAN_STEPS if take is None else take
    return (
        MethodSpec(method_id="smolvla", horizon=chosen, representation="lerobot_12",
                   label="lerobot/smolvla_robocasa as a Flow; trained, not privileged",
                   build=lambda env, t=chosen, d=device: smolvla_arm(take=t, device=d)),
    )


def model_specs(*, model_id: str, memory: int, dof: int) -> tuple:
    """A frontier model as the policy, in the paper's Agentic form.

    Agentic is the form the Astranomy study reports its best result from, which
    is why it is the one wired first. The representation is `eef_delta_base`:
    a delta end-effector pose plus gripper plus base, which is the paper's own
    action representation extended with the base this robot needs — several
    atomic-seen tasks (`NavigateKitchen` most obviously) cannot be done without
    it. The environment derives the control-mode flag, so the model is never
    asked for a protocol detail it has no way to reason about.
    """
    from .study_gemini import GeminiClient
    from .study_model import AgenticModelMethod, ModelConfig

    def build(environment, _m=model_id, _mem=memory, _dof=dof):
        return AgenticModelMethod(
            client=GeminiClient(model_id=_m,
                                # Published rates, so a run records real cost
                                # per episode rather than leaving it absent.
                                usd_per_1m_input=0.30, usd_per_1m_output=2.50),
            # The most explicit prompt in the study's content axis: a row width
            # the example contradicts is answered with the example's width.
            config=ModelConfig(model_id=_m, horizon_band="short",
                               prompt_variant="with_layout"),
            dof=_dof, memory=_mem, source="model",
            # A provider hiccup must not void an episode; the refusal is counted.
            refusal_action="hold")

    return (
        MethodSpec(method_id=f"agentic:{model_id}", horizon=15,
                   representation="eef_delta_base",
                   label=f"{model_id} as an Agentic policy; memory={memory}",
                   build=build),
    )


def model_observation():
    """Two views at 256 px for a model that has to actually see.

    The 128 px single view the baselines use is enough for a scripted policy
    reading privileged state and far too little for a vision-language model.
    Two views is also the study's prompt-geometry axis at its documented
    setting.
    """
    from .study_robocasa_env import ObservationConfig

    return ObservationConfig(views=("over_shoulder", "wrist"), history=0, image_size=256)


def vla_observation():
    from .study_robocasa_env import ObservationConfig
    from .study_vla import LEROBOT_IMAGE_SIZE, LEROBOT_VIEWS

    return ObservationConfig(views=LEROBOT_VIEWS, history=0,
                             image_size=LEROBOT_IMAGE_SIZE)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="study-pnp")
    parser.add_argument("--task", action="append", dest="tasks", default=None,
                        help="Committed task name; repeatable. Defaults to a reach-spread subset.")
    parser.add_argument("--all-tasks", action="store_true",
                        help="Run all 18 committed tasks instead of the subset.")
    parser.add_argument("--subset", type=int, default=4,
                        help="Size of the reach-spread subset when no task is named.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
                        help="Episode seeds. The study uses 0-4.")
    parser.add_argument("--horizon", type=int, default=5,
                        help="Actions proposed per model call.")
    parser.add_argument("--representation", default="eef_delta",
                        help="Action representation; eef_delta_base commands the mobile base.")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--specification", choices=("default", "vague", "identifier"),
                        default="default",
                        help="Task-specification axis: RoboCasa's goal, or the derived "
                             "instruction that withholds the object's identity.")
    parser.add_argument("--vla-tasks", action="store_true",
                        help="Run only the tasks the released VLA checkpoint also knows.")
    parser.add_argument("--atomic-seen", action="store_true",
                        help="Run the leaderboard's 18-task atomic-seen split, which is "
                             "the column its published success rates are reported in. "
                             "Thirteen of them are not pick-and-place, so the partial "
                             "score does not apply and binary success is the metric.")
    parser.add_argument("--official-horizons", action="store_true",
                        help="Give each task the episode budget the official client "
                             "gives it, instead of a flat --max-steps.")
    parser.add_argument("--take", type=int, default=None,
                        help="Actions consumed per inference by a Flow policy; the "
                             "official client uses 5. Omit to use that default.")
    parser.add_argument("--model-id", default="gemini-2.5-flash",
                        help="Model for the Agentic arm. 2.5-flash measured 6.5 s per "
                             "call against 2.5-pro's 35.4 s, and parsed more reliably.")
    parser.add_argument("--memory", type=int, default=3,
                        help="Previous calls described in the prompt; 0 is the Direct form.")
    parser.add_argument("--methods", choices=("baselines", "reference", "vla", "model", "both"),
                        default="baselines",
                        help="Which arms to run. The reference is privileged and always "
                             "uses joint_delta, whatever --representation says.")
    parser.add_argument("--checkpoint", default="",
                        help="Append finished episodes to this JSONL and resume from it. "
                             "A full sweep is hours long and the summary prints only at "
                             "the end, so an interruption otherwise loses everything.")
    parser.add_argument("--preflight", action="store_true",
                        help="Probe the policy against the environment for a few steps "
                             "and stop unless every interface check passes. Every "
                             "expensive mistake in this study was an interface "
                             "disagreement that produced valid, useless behaviour.")
    parser.add_argument("--device", default="cpu",
                        help="Torch device for a checkpoint-backed policy. "
                             "SmolVLA inference is 98%% of an episode's cost on "
                             "CPU (2.58 s of 2.63 s per control step, measured), "
                             "so this is the single setting that decides whether "
                             "a full-protocol sweep is affordable.")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args(argv)

    from .study_robocasa_env import RoboCasaPnPEnvironment

    def build(task, representation):
        if representation == "eef_delta_base" and args.methods == "model":
            return RoboCasaPnPEnvironment(
                task=task, representation=representation, command_base=True,
                observation=model_observation(),
                specification_style=args.specification)
        if representation == "lerobot_12":
            # The checkpoint fixes its own observation and state format, and its
            # language conditioning is the task identifier unless asked
            # otherwise - handing it prose is out of distribution.
            style = args.specification if args.specification != "default" else "identifier"
            return RoboCasaPnPEnvironment(
                task=task, representation=representation,
                observation=vla_observation(), state_mode="lerobot",
                specification_style=style)
        return RoboCasaPnPEnvironment(
            task=task, representation=representation,
            command_base=representation == "eef_delta_base",
            specification_style=args.specification)

    try:
        if args.tasks:
            # A named task may be one of this study's committed pick-and-place
            # entries or one of the leaderboard's atomic-seen tasks, which are
            # not all pick-and-place and so have no entry here.
            tasks = tuple(name if name in ATOMIC_SEEN_TASKS else get_task(name).task
                          for name in args.tasks)
        elif args.atomic_seen:
            tasks = ATOMIC_SEEN_TASKS
        elif args.vla_tasks:
            tasks = vla_comparable()
        elif args.all_tasks:
            tasks = list_tasks()
        else:
            tasks = ablation_subset(args.subset)
        specs = ()
        if args.methods in ("baselines", "both"):
            specs += baseline_specs(args.horizon, args.representation)
        if args.methods in ("reference", "both"):
            specs += reference_specs(args.horizon)
        if args.methods == "vla":
            specs += vla_specs(args.horizon, take=args.take, device=args.device)
        if args.methods == "model":
            specs += model_specs(model_id=args.model_id, memory=args.memory, dof=10)
        # A model's output is admitted only when a model arm is actually in
        # the sweep, so the default stays scripted-only.
        sources = (("scripted", "model") if args.methods in ("vla", "model")
                   else ("scripted",))
        horizons = ({task: official_horizon(task) for task in tasks}
                    if args.official_horizons else {})
        if args.preflight:
            from . import study_preflight

            spec = specs[0]
            environment = build(tasks[0], spec.representation)
            config = getattr(spec.build(environment), "config", None)
            report = study_preflight.run(
                environment=environment, method=spec.build(environment),
                task=tasks[0], representation=spec.representation,
                dof=REPRESENTATION_DOF[spec.representation], config=config)
            try:
                environment.close()
            except Exception:  # noqa: BLE001 - cleanup must not mask the report
                pass
            print(render(report.to_dict()))
            return 0 if report.ok else 1
        sweep = Sweep(build_environment=build, tasks=tasks,
                      seeds=tuple(args.seeds), max_steps=args.max_steps,
                      horizons=horizons, allowed_sources=sources,
                      checkpoint=args.checkpoint)
        summary = sweep.run(specs)
    except (ValueError, TypeError, OSError) as error:
        print(render({"error": f"{type(error).__name__}: {error}"}))
        return 1
    print(render(summary, summary_only=args.summary_only))
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
