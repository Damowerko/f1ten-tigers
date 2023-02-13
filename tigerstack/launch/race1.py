from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="tigerstack",
                executable="safety_node",
            ),
            Node(
                package="tigerstack",
                executable="gap_follow",
            ),
        ]
    )
