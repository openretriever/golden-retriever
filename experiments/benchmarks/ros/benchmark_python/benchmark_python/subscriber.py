"""
Subscriber node for ROS 2 Python benchmarking.

NOTE: This file was copied and slightly modified from the dora-rs benchmarks in 
https://github.com/dora-rs/dora-benchmark/blob/main/ros2/py_pubsub/setup.py
"""

import csv
import os
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt64MultiArray


NAME = "ROS 2 Python"
PLATFORM = "COMPUTER_PERF"
LATENCY = True


def record_results(start, current_size, latencies, latency):
    csv_file = "experiments/benchmarks/results/ros_python_benchmark_results.csv"
    append = os.path.isfile(csv_file)
    log_header = ["name", "platform", "size", "latency_ns"]
    log_row = [NAME, PLATFORM, current_size, latencies]
    if append:
        with open(csv_file, "a", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(log_row)
    else:
        with open(csv_file, "w+", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(log_header)
            w.writerow(log_row)


class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__("minimal_subscriber")
        self.subscription = self.create_subscription(
            UInt64MultiArray, "topic", self.listener_callback, 10,
        )
        self.subscription  # prevent unused variable warning
        self.current_size = 0
        self.latencies = []
        self.n = 0

    def listener_callback(self, msg: UInt64MultiArray):

        t_received = time.perf_counter_ns()
        length = len(msg.data) * 8  # As it is Uint64
        if length != self.current_size:
            if self.n > 0:
                record_results([], self.current_size, self.latencies, LATENCY)
            self.current_size = length
            self.n = 0
            self.latencies = []
        t_send = msg.data[0]
        self.latencies.append(t_received - t_send)
        self.n += 1
        # self.get_logger().info('I heard: "%s"' % msg.data)


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
