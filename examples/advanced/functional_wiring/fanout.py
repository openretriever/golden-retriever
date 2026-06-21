"""
Fan-Out Example - Signal Splitting

Demonstrates one source feeding multiple destinations.
This is the dual of fan-in (fusion).
"""
from dataclasses import dataclass

from retriever.flow import Flow, Pipeline, Rate
from retriever.flow import io


@io
@dataclass
class SensorReading:
    value: float | None = None
    timestamp: float | None = None


@io
@dataclass
class DetReading:
    value: float | None = None
    timestamp: float | None = None


@io
@dataclass
class LogReading:
    value: float | None = None
    timestamp: float | None = None


@io
@dataclass
class DetectionResult:
    detected: bool
    confidence: float


class Sensor(Flow[None, SensorReading]):
    def step(self, _) -> SensorReading:
        import random
        import time
        value = random.uniform(0, 100)
        ts = time.time()
        print(f"Sensor: {value:.1f}")
        return SensorReading(value=value, timestamp=ts)


class Detector(Flow[DetReading, DetectionResult]):
    def __init__(self, threshold: float = 50.0):
        self.threshold = threshold

    def init_config(self):
        return {"threshold": self.threshold}

    def step(self, input: DetReading) -> DetectionResult:
        if input.value is None:
            return DetectionResult(detected=False, confidence=0.0)
        detected = input.value > self.threshold
        confidence = min(1.0, input.value / 100.0)
        if detected:
            print(f"Detector: ALERT! {input.value:.1f} > {self.threshold}")
        return DetectionResult(detected=detected, confidence=confidence)


class DataLogger(Flow[LogReading, None]):
    def step(self, input: LogReading) -> None:
        if input.value is None or input.timestamp is None:
            return None
        print(f"Logger: recorded {input.value:.1f} at {input.timestamp:.3f}")
        return None


def main():
    print("=== Fan-Out Example ===")
    print("Pattern: Sensor --> Detector (for alerts)")
    print("                --> Logger (for recording)\n")

    pipe = Pipeline('fanout_demo')
    with pipe:
        sensor = Sensor() @ Rate(2.0)
        detector = Detector(60.0) @ Rate(2.0)
        logger = DataLogger() @ Rate(2.0)
        sensor.then(detector)
        sensor.then(logger)

    print("Running pipeline for 3s...")
    pipe.run(backend='multiprocessing', duration=3.0)


if __name__ == "__main__":
    main()
