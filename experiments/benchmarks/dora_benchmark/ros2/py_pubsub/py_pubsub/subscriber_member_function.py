# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import csv
import os
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt64MultiArray

LATENCY = True

NAME = os.getenv("NAME") or "ROS 2"
PLATFORM = "COMPUTER_PERF"
current_size = 8
n = 0
latencies = []
save_x = []


def record_results(start, current_size, latencies, latency: bool):

    avg_latency = np.array(latencies).mean()

    csv_file = os.getenv("CSV_TIME_FILE", "time.csv")
    append = os.path.isfile(csv_file)
    log_header = ["name", "platform", "size", "latency"]
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


class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__("minimal_subscriber")
        self.subscription = self.create_subscription(
            UInt64MultiArray, "topic", self.listener_callback, 10
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
        self.latencies.append((t_received - t_send) / 1000)
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
