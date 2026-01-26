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


class MinimalPublisher(Node):
    def __init__(self):
        super().__init__("minimal_publisher")
        self.publisher_ = self.create_publisher(UInt64MultiArray, "topic", 10)
        timer_period = 0.05  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0
        self.j = 0

    def timer_callback(self):
        msg = UInt64MultiArray()
        random_data = np.array(
            np.random.randint(255, size=SIZES[self.i], dtype=np.uint64)
        )

        random_data[0] = np.array([time.perf_counter_ns()])

        random_data = (
            random_data.tobytes()
        )  # <- Converting to bytes is required as there is no easy way to pass the memoryview to ROS2 Messages
        msg.data.frombytes(random_data)
        self.publisher_.publish(msg)
        # self.get_logger().info('Publishing: "%s"' % msg.data)
        if self.j == 100:
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
