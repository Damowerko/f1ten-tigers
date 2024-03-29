#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    sim_arg = DeclareLaunchArgument(
        "sim",
        default_value="False",
        description="Whether to run in simulation or not.",
    )

    return LaunchDescription(
        [
            sim_arg,
            Node(
                package="tigerstack",
                executable="mpc_node",
                name="mpc_node",
                parameters=[
                    {"sim": LaunchConfiguration("sim")},
                ],
            ),
            Node(
                package="planner",
                executable="planner_node",
                name="planner_node",
                parameters=[
                    {"sim": LaunchConfiguration("sim")},
                ],
            ),
        ]
    )
