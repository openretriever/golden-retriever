#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time

import pyarrow as pa
from dora import Node
from helper import record_results

pa.array([])
node = Node()


current_size = 8
n = 0
i = 0
latencies = []

NAME = "dora Node"

while True:
    event = node.next()
    if event["type"] == "INPUT":
        data = event["value"]
    else:
        break
    
    if len(data) == 0:
        break

    # frame = np.frombuffer(data, dtype="uint8")
    # frame = cv2.imdecode(frame, -1)
    t_received = time.perf_counter_ns()
    length = len(data) * 8
    if length != current_size:
        if n > 0:
            record_results(NAME, current_size, latencies)
        current_size = length
        n = 0
        start = time.perf_counter_ns()
        latencies = []

    ## Arrow Test: Using PyBytes
    # t_send = np.frombuffer(data[:8], np.uint64)[0]
    #
    ## Arrow Test: Using Arrow
    t_send = data[0].as_py()
    latencies.append((t_received - t_send))  # Manual modification: Log as nanoseconds

    n += 1
    i += 1

record_results(NAME, current_size, latencies)
