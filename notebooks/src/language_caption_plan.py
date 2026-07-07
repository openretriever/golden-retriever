# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Language → Plan (GoldenRetriever)
#
# Turn a scene **caption** into a small primitive **plan** using three plain
# Flows. There is no LLM in the loop — the planner is deterministic rules, so the
# notebook runs anywhere and shows the real point: the **typed data contract**
# between language and planning, and where a model would later slot in.

# %% [markdown]
# > **Running in Colab?** The next cell installs `retriever-core`. From a source
# > checkout (or once it's already installed) the install is skipped.

# %%
# Colab setup: install retriever-core only if it isn't importable yet.
try:
    import retriever  # noqa: F401
except ImportError:  # pragma: no cover
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "retriever-core"], check=True
    )

# %% [markdown]
# ## The Flows
#
# Each stage is a `Flow[In, Out]` whose payloads are standard language types from
# `retriever.types.language`. A Flow wakes on its clock and returns one typed
# record — it knows nothing about buffering or how it's wired.

# %%
from retriever.flow import Flow
from retriever.types.language import Caption, PlanStepText, PlanText


class CaptionSource(Flow[None, Caption]):
    """Emits one caption per tick (stands in for a captioner / VLM)."""

    def __init__(self, *, captions=None):
        super().__init__()
        self.captions = tuple(captions or (
            "red cube on the left near the blue cylinder",
            "pick the red cube and place it at the goal",
            "inspect the blue object before moving",
        ))

    def reset(self):
        self._i = 0

    def step(self, _):
        text = self.captions[self._i % len(self.captions)]
        self._i += 1
        return Caption(text=text, source="golden.language_examples")


class CaptionPlanner(Flow[Caption, PlanText]):
    """Caption -> ordered primitive steps. Rules here; swap for an LLM later."""

    def step(self, caption: Caption) -> PlanText:
        text = caption.text.lower()
        steps = [PlanStepText(index=0, text=f"inspect scene: {caption.text}", action_label="inspect")]
        if "red" in text:
            steps.append(PlanStepText(index=len(steps), text="focus on the red object", action_label="focus"))
        if "pick" in text or "grasp" in text:
            steps.append(PlanStepText(index=len(steps), text="pick the target object", action_label="pick"))
        if "place" in text or "goal" in text:
            steps.append(PlanStepText(index=len(steps), text="place the target at the goal", action_label="place"))
        return PlanText(steps=tuple(steps), summary=caption.text, source="caption_planner")


class PlanTextPrinter(Flow[PlanText, None]):
    def step(self, plan: PlanText) -> None:
        print("plan=" + str([f"{s.index}:{s.action_label}:{s.text}" for s in plan.steps]))

# %% [markdown]
# `CaptionPlanner` only reads `caption.text` and returns a typed `PlanText`. That
# isolation is the whole point: replace the rule body with a real model and the
# graph is unchanged.

# %% [markdown]
# ## Wire it and run
#
# Clocks say *when* each Flow runs; `sync=` says *which* input record it consumes.
# The source ticks on a `Rate`; the planner and printer are `Trigger`ed by new
# arrivals on the port named in the type.

# %%
from retriever.flow import Latest, Pipeline, Rate, Trigger

pipe = Pipeline("language_caption_plan")
with pipe:
    source = CaptionSource() @ Rate(hz=10)
    planner = CaptionPlanner() @ Trigger("text")     # wakes on a new Caption.text
    printer = PlanTextPrinter() @ Trigger("summary")  # wakes on a new PlanText.summary
    pipe.connect(source, planner, sync=Latest())
    pipe.connect(planner, printer, sync=Latest())

for _ in range(3):
    pipe.step(dt=0.1)  # in-process; set a breakpoint in any step()
pipe.close_stepper()

# %% [markdown]
# ## Why it's shaped this way
#
# `Caption`, `PlanText`, and `PlanStepText` are **standard payloads** — the same
# types perception, memory, and planning Flows hand each other. Because the
# contract is typed and explicit, you can swap `CaptionPlanner`'s rule body for a
# real VLM without touching the graph, step the pipeline in-process and
# breakpoint inside any `step()`, and record then replay the exact stream later.
