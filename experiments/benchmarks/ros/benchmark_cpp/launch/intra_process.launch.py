"""Run publisher + subscriber in a single process with intra-process comms.

Both nodes are loaded as components into a multi-threaded container. With
`use_intra_process_comms: True`, the publisher"s `publish(std::unique_ptr<T>)`
hands the message directly to the subscriber"s `UniquePtr` callback in the
same process — no serialization, no DDS hop, no copy.

Launch from a workspace where `benchmark_cpp` is installed:

    ros2 launch benchmark_cpp intra_process.launch.py

The subscriber writes results to
`experiments/benchmarks/results/ros_cpp_benchmark_results.csv` relative to
the working directory, so invoke from the repo root.
"""

from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    container = ComposableNodeContainer(
        name="benchmark_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container_mt",
        output="screen",
        composable_node_descriptions=[
            ComposableNode(
                package="benchmark_cpp",
                plugin="MinimalPublisher",
                name="publisher",
                extra_arguments=[{"use_intra_process_comms": True}],
            ),
            ComposableNode(
                package="benchmark_cpp",
                plugin="MinimalSubscriber",
                name="subscriber",
                parameters=[{
                    "benchmark_name": "ROS 2 C++ (Components)",
                    "output_file":
                        "experiments/benchmarks/results/ros_cpp_components_benchmark_results.csv",
                }],
                extra_arguments=[{"use_intra_process_comms": True}],
            ),
        ],
    )

    return LaunchDescription([container])
