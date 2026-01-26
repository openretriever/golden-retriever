"""
Publisher node for ROS 2 Python benchmarking.

NOTE: This file was copied and slightly modified from the dora-rs benchmarks in 
https://github.com/dora-rs/dora-benchmark/blob/main/ros2/py_pubsub/setup.py
"""

import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt64MultiArray

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


class MinimalPublisher(Node):
    def __init__(self):
        super().__init__("minimal_publisher")
        self.publisher_ = self.create_publisher(UInt64MultiArray, "topic", 10)
        timer_period = DATA_RATE_S
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0
        self.j = 0

    def timer_callback(self):
        random_data = np.array(
            np.random.randint(255, size=SIZES[self.i], dtype=np.uint64)
        )
        random_data[0] = time.perf_counter_ns()

        msg = UInt64MultiArray(data=random_data.tolist())
        self.publisher_.publish(msg)
        # self.get_logger().info('Publishing: "%s"' % msg.data)
        if self.j == NUM_POINTS_PER_SIZE:
            self.i += 1
            self.j = 0
        else:
            self.j += 1


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
