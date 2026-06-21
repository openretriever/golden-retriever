"""
FRP Combinators Example - Single-Expression Graph Building

Demonstrates Arrow-style combinators for building dataflow graphs in one expression:
- `>>` (compose): sequential connection
- `&` (fanout): split signal to multiple destinations
"""
from dataclasses import dataclass

from retriever.flow import Flow, Pipeline, Rate
from retriever.flow import io


@io
@dataclass
class SensorData:
    value: float
    seq: int


@io
@dataclass
class SensorIn:
    value: float
    seq: int


@io
@dataclass
class AlertResult:
    alert: bool


class Sensor(Flow[None, SensorData]):
    def step(self, _) -> SensorData:
        import random
        if not hasattr(self, 'seq'):
            self.seq = 0
        self.seq += 1
        value = random.uniform(0, 100)
        print(f"Sensor: {value:.1f} (seq={self.seq})")
        return SensorData(value=value, seq=self.seq)


class Detector(Flow[SensorIn, AlertResult]):
    def __init__(self, threshold: float = 50.0):
        self.threshold = threshold

    def init_config(self):
        return {"threshold": self.threshold}

    def step(self, input: SensorIn) -> AlertResult:
        if input.value is None:
            return AlertResult(alert=False)
        detected = input.value > self.threshold
        if detected:
            print(f"  Detector: ALERT! {input.value:.1f} > {self.threshold}")
        return AlertResult(alert=detected)


class Logger(Flow[SensorIn, None]):
    def step(self, input: SensorIn) -> None:
        if input.value is not None:
            print(f"  Logger: recorded seq={input.seq}")
        return None


class Recorder(Flow[SensorIn, None]):
    def step(self, input: SensorIn) -> None:
        if input.value is not None:
            print(f"  Recorder: saved {input.value:.1f}")
        return None


def main():
    print("=== FRP Combinators Example ===")
    print("Demonstrating single-expression graph building\n")

    pipe = Pipeline('combinators_demo')
    with pipe:
        sensor = Sensor() @ Rate(2.0)
        detector = Detector(60.0) @ Rate(2.0)
        logger = Logger() @ Rate(2.0)
        recorder = Recorder() @ Rate(2.0)
        print("Graph: sensor >> (detector & logger & recorder)")
        print("       sensor splits to all 3 destinations\n")
        sensor >> (detector & logger & recorder)

    print("Running pipeline for 2s...")
    pipe.run(backend='multiprocessing', duration=2.0)


if __name__ == "__main__":
    main()
