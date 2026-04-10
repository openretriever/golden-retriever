# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: "1.3"
# --

# %% [markdown]
# # Retriever notebook demo
#
# This notebook keeps the source of truth in `py:percent` format so it stays
# reviewable in git while remaining convertible to a normal `.ipynb` file.

# %%
from retriever.flow import Flow, Pipeline, Rate, Trigger, io


@io
class Observation:
    value: float


@io
class Belief:
    estimate: float


class SyntheticObservation(Flow[None, Observation]):
    def __init__(self):
        super().__init__()
        self.count = 0

    def reset(self) -> None:
        self.count = 0

    def step(self, _input: None) -> Observation:
        self.count += 1
        return Observation(value=float(self.count))


class BeliefUpdater(Flow[Observation, Belief]):
    def __init__(self, alpha: float = 0.4):
        super().__init__()
        self.alpha = alpha
        self.estimate = 0.0

    def reset(self) -> None:
        self.estimate = 0.0

    def step(self, input: Observation) -> Belief:
        self.estimate = (1.0 - self.alpha) * self.estimate + self.alpha * input.value
        return Belief(estimate=self.estimate)


class Printer(Flow[Belief, None]):
    def step(self, input: Belief) -> None:
        print(f"belief={input.estimate:.3f}")


with Pipeline("notebook_demo") as pipe:
    source = SyntheticObservation() @ Rate(hz=10)
    updater = BeliefUpdater(alpha=0.5) @ Trigger("value")
    printer = Printer() @ Trigger("estimate")
    source >> updater >> printer

print("Pipeline built.")

# %%
pipe.reset()
for step_idx in range(6):
    print(f"--- step {step_idx} ---")
    pipe.step(dt=0.1)
