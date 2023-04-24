#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="tigerstack",
                executable="mpc_node",
                name="mpc_node",
                parameters=[
                    {"sim": True},
                ],
            ),
            Node(
                package="tigerstack",
                executable="mpc_node",
                name="mpc_node_opp",
                parameters=[
                    {"sim": True},
                    {"speed_factor": 0.8},
                ],
                remappings=[
                    ("/drive", "/opp_drive"),
                    ("/ego_racecar/odom", "/opp_racecar/odom"),
                    ("/path", "/opp_path"),
                ],
            ),
        ]
    )
