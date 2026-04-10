"""
Fan-In: Multiple sources → single destination port.

Key insight: All sources share ONE buffer. Adapter determines sampling.

Demo 1: Latest - receives A1, B1, C1, A2, B2... from any source
Demo 2: Window(mean) - proves shared buffer (mean of 10,20,30 ≈ 20)
"""

from dataclasses import dataclass
from retriever.flow import Flow, flow_io, Rate, Trigger, Pipeline, Latest, Window


# =============================================================================
# Demo 1: Fan-in with Latest (string readings: A1, B1, C1...)
# =============================================================================

@flow_io
@dataclass
class SensorData:
    reading: str


class Sensor(Flow[None, SensorData]):
    """Produces readings: A1, A2, A3..."""
    def __init__(self, label: str):
        self._label = label
        self._count = 0

    def init_config(self):
        return {"label": self._label}

    def step(self, _) -> SensorData:
        self._count += 1
        return SensorData(reading=f"{self._label}{self._count}")


class Monitor(Flow[SensorData, None]):
    """Receives latest from any sensor."""
    def step(self, inp: SensorData):
        print(f"  Monitor <- {inp.reading}")


def demo_latest():
    print("\n" + "=" * 50)
    print("Demo 1: Fan-in + Latest")
    print("  A, B, C sensors → Monitor")
    print("  Output: interleaved A1, B1, C1, A2, B2...")
    print("=" * 50)

    pipe = Pipeline("fanin_latest")
    with pipe:
        a = Sensor("A") @ Rate(hz=1)
        b = Sensor("B") @ Rate(hz=1)
        c = Sensor("C") @ Rate(hz=1)
        monitor = Monitor() @ Trigger("reading")

        a.then(monitor, sync=Latest())
        b.then(monitor, sync=Latest())
        c.then(monitor, sync=Latest())

    pipe.run(duration=4.0)


# =============================================================================
# Demo 2: Fan-in with Window(mean) - proves shared buffer
# =============================================================================

@flow_io
@dataclass
class NumericData:
    value: float


class NumericSensor(Flow[None, NumericData]):
    """Produces constant value."""
    def __init__(self, value: float):
        self._value = value

    def init_config(self):
        return {"value": self._value}

    def step(self, _) -> NumericData:
        return NumericData(value=self._value)


class Aggregator(Flow[NumericData, None]):
    """Shows mean of shared buffer."""
    def step(self, inp: NumericData):
        print(f"  Aggregator <- mean = {inp.value:.1f}")


def demo_mean():
    print("\n" + "=" * 50)
    print("Demo 2: Fan-in + Window(mean)")
    print("  A=10, B=20, C=30 → Aggregator")
    print("  Shared buffer → mean ≈ 20")
    print("=" * 50)

    pipe = Pipeline("fanin_mean")
    adapter = Window(buffer_size=10, duration=5.0, agg="mean")

    with pipe:
        a = NumericSensor(10) @ Rate(hz=1)
        b = NumericSensor(20) @ Rate(hz=1)
        c = NumericSensor(30) @ Rate(hz=1)
        agg = Aggregator() @ Rate(hz=0.5)

        a.then(agg, sync=adapter)
        b.then(agg, sync=adapter)
        c.then(agg, sync=adapter)

    pipe.run(duration=5.0)


if __name__ == "__main__":
    demo_latest()
    demo_mean()
