#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare the launch arguments
    sim_arg = DeclareLaunchArgument(
        "sim",
        default_value="False",
        description="Whether to run in simulation or not.",
    )
    waypoint_arg = DeclareLaunchArgument(
        "waypoints_file",
        default_value="/maps/skir.csv",
        description="Waypoints file location",
    )

    return LaunchDescription(
        [
            sim_arg,
            waypoint_arg,
            Node(
                package="tigerstack",
                executable="mpc_rrt_node",
                name="mpc_rrt_node",
                parameters=[
                    {"sim": LaunchConfiguration("sim")},
                ],
            ),
            Node(
                package="tigerstack",
                executable="pure_pursuit",
                name="pure_pursuit",
                parameters=[
                    {"sim": LaunchConfiguration("sim"),
                    "waypoints_file": LaunchConfiguration("waypoints_file")},
                ],
            ),
            Node(
                package="tigerstack",
                executable="rrt_node",
                name="rrt_node",
                parameters=[
                    {"sim": LaunchConfiguration("sim"),
                    "waypoints_file": LaunchConfiguration("waypoints_file")},
                ],
            ),
            Node(
                package="tigerstack",
                executable="mpc_oppo",
                name="mpc_oppo",
                parameters=[
                    {"sim": LaunchConfiguration("sim"),
                    "waypoints_file": LaunchConfiguration("waypoints_file")},
                ],
            ),
        ]
    )
