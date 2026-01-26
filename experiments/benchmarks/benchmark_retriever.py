"""
Benchmarking script for Retriever.

Backends are configurable with the --backend command-line argument.
"""

import argparse
from dataclasses import dataclass
import time
import os

import csv
import numpy as np

from retriever.flow import Flow, flow_io, Rate, Trigger, Pipeline, Latest

SIZES = [
    8,
    64,
    512,
    10 * 512,
    100 * 512,
    1000 * 512,
    10000 * 512,
    8,
]
NUM_POINTS_PER_SIZE = 100
DATA_RATE_S = 0.05  # seconds

NAME = "Retriever"
PLATFORM = "COMPUTER_PERF"
LATENCY = True


@flow_io
@dataclass
class RandomSequence:
    data: np.ndarray


class SourceFlow(Flow[None, RandomSequence]):
    def __init__(self):
        super().__init__()
        self.i = 0
        self.j = 0

    def run(self, _):
        random_data = np.array(
            np.random.randint(255, size=SIZES[self.i], dtype=np.uint64)
        )
        random_data[0] = time.perf_counter_ns()
        if self.j == NUM_POINTS_PER_SIZE:
            self.i += 1
            self.j = 0
        else:
            self.j += 1

        return RandomSequence(data=random_data)


class SinkFlow(Flow[RandomSequence, None]):
    def __init__(self, backend: str = "dora"):
        super().__init__()
        self.backend = backend
        self.latencies = []
        self.current_size = 0
        self.n = 0

    def run(self, input: RandomSequence):
        t_received = time.perf_counter_ns()
        length = len(input.data) * 8  # As it is Uint64
        if length != self.current_size:
            if self.n > 0:
                self.record_results([], self.current_size, self.latencies, LATENCY, self.backend)
            self.current_size = length
            self.n = 0
            self.latencies = []
        t_send = int(input.data[0])
        self.latencies.append((t_received - t_send) / 1000.0)
        self.n += 1

    def record_results(self, start, current_size, latencies, latency, backend):
        avg_latency = np.array(latencies).mean()

        csv_file = f"experiments/benchmarks/results/retriever_{backend}_benchmark_results.csv"
        append = os.path.isfile(csv_file)
        log_header = ["name", "platform", "size", "latency_ns"]
        log_row = [NAME, PLATFORM, current_size, avg_latency]
        if append:
            with open(csv_file, "a", encoding="utf-8") as f:
                w = csv.writer(f, lineterminator="\n")
                w.writerow(log_row)
        else:
            with open(csv_file, "w+", encoding="utf-8") as f:
                w = csv.writer(f, lineterminator="\n")
                w.writerow(log_header)
                w.writerow(log_row)


def main():
    p = argparse.ArgumentParser(
        description="Perception demo (camera -> detection -> display)"
    )
    p.add_argument("--backend", default="dora", choices=["dora", "multiprocessing"])
    p.add_argument("--duration", type=float, default=120.0)
    args = p.parse_args()

    pipe = Pipeline("retriever_benchmarking_dora")
    with pipe:
        source = SourceFlow() @ Rate(hz=1.0 / DATA_RATE_S)
        sink = SinkFlow(backend=args.backend) @ Trigger("data")
        source >> sink  # TODO: Why does this go back to using dora backend?

    pipe.run(backend=args.backend, duration=args.duration, blocking=True)


if __name__ == "__main__":
    main()
