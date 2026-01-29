#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time

import numpy as np
import pyarrow as pa
from dora import Node

# NOTE: SIZES defines the number of uint64 elements in the payload.
# Since each uint64 is 8 bytes, the actual payload size in bytes is: Element Count * 8.
# Examples:
#   i=6  -> 2^6  = 64 elements   -> 512 Bytes
#   i=20 -> 2^20 = 1M elements   -> 8 MB
#   i=24 -> 2^24 = 16M elements  -> 128 MB
SIZES = [2**i for i in range(6, 25)]


node = Node()
pa.array([])

# test latency first
for size in SIZES:
    for _ in range(0, 100):
        now = time.time()
        random_data = np.random.randint(1000, size=size, dtype=np.uint64)
        random_data[0] = time.perf_counter_ns()

        node.send_output("latency", pa.array(random_data))
        time.sleep(max(0, 0.05 - (time.time() - now)))

node.send_output("latency", pa.array([], type=pa.uint64()))
