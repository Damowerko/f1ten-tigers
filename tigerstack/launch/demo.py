#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    sim_arg = DeclareLaunchArgument(
        "sim",
        default_value="True",
        description="Whether to run in simulation or not.",
    )    
    map_arg = DeclareLaunchArgument(
        "skir",
        default_value="False",
        description="False to use levine map",

    )
    return LaunchDescription(
        [
            sim_arg,
            map_arg,
            Node(
                package="tigerstack",
                executable="mpc_node",
                name="mpc_node",
                parameters=[
                    {"sim": LaunchConfiguration("sim")},
                    {"skir": LaunchConfiguration("skir")},
                ],
            ),
            Node(
                package="planner",
                executable="planner_node",
                name="planner_node",
                parameters=[
                    {"sim": LaunchConfiguration("sim")},
                ]
            ),
        ]
    )
