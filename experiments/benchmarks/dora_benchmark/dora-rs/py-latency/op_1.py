#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
from typing import Callable

import numpy as np
import pyarrow as pa
from dora import DoraStatus

pa.array([])

SIZES = [
    8,
    64,
    512,
    2048,
    4 * 512,
    10 * 512,
    100 * 512,
    1000 * 512,
    10000 * 512,
    8,
]


class Operator:
    """
    Template docstring
    """

    def __init__(self) -> None:
        self.size_index = 0
        self.i = 0

    def on_event(
        self,
        dora_event: dict,
        send_output: Callable[[str, bytes], None],
    ) -> DoraStatus:
        if dora_event["type"] == "INPUT":
            return self.on_input(dora_event, send_output)
        return DoraStatus.CONTINUE

    def on_input(
        self,
        dora_input: dict,
        send_output: Callable[[str, bytes], None],
    ):
        """

        Args:
            dora_input (dict): Input dict containing an `id`, `data` and `metadata`.
            send_output (Callable[[str, bytes], None]): Send output to the dataflow

        Returns:
            DoraStatus:
                CONTINUE means that the operator will
                    keep listening for further inputs.
                STOP means that the operator stop listening for inputs.

        """
        if self.i == 100:
            self.size_index += 1
            self.i = 0
            if self.size_index == len(SIZES):
                return DoraStatus.STOP
        else:
            self.i += 1

        random_data = np.random.randint(
            255,
            size=SIZES[self.size_index],
            dtype=np.uint64,  # <- Can be replaced with int8
        )
        random_data[0] = np.array([time.perf_counter_ns()])  # <- Counter starts here

        send_output("latency", pa.array(random_data), dora_input["metadata"])

        return DoraStatus.CONTINUE
