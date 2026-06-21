from dataclasses import dataclass

from retriever.flow import Flow, Pipeline, Rate
from retriever.flow.adapter import Window
from retriever.flow import io


@io
@dataclass
class FastOutput:
    val: float


class FastSource(Flow[None, FastOutput]):
    def step(self, _):
        if not hasattr(self, 'count'):
            self.count = 0.0
        out = FastOutput(val=self.count)
        self.count += 1.0
        return out


@io
@dataclass
class WindowInput:
    val: float


@io
@dataclass
class WindowFusionOut:
    mean: float


class WindowFusion(Flow[WindowInput, WindowFusionOut]):
    def step(self, input: WindowInput) -> WindowFusionOut:
        if hasattr(input, 'val'):
            print(f"Fusion received window mean: {input.val}")
            return WindowFusionOut(mean=input.val)
        return WindowFusionOut(mean=0.0)


def main():
    print("Wiring with Window(duration=0.5, agg='mean')...")
    pipe = Pipeline('sync_policies_window')
    with pipe:
        fast = FastSource() @ Rate(10.0)
        fusion = WindowFusion() @ Rate(1.0)
        pipe.connect(fast, fusion, map={'val': 'val'}, sync=Window(buffer_size=20, duration=0.5, agg='mean'))

    print("Running pipeline for 2.2s...")
    pipe.run(backend='multiprocessing', duration=2.2)


if __name__ == "__main__":
    main()
