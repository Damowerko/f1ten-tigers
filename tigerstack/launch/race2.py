#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    sim_arg = DeclareLaunchArgument(
        "sim",
        default_value="False",
        description="Whether to run in simulation or not.",
    )
    return LaunchDescription([sim_arg, *generate_localization_nodes()])


def generate_localization_nodes():
    common_localization_parameters = [
        {"frequency": 50.0},
        {"two_d_mode": True},
        {"publish_tf": False},
        {"imu0": "/sensors/imu/raw"},
        {
            "imu0_config": [
                False,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                True,
                True,
                False,
                False,
            ]
        },
        {
            "odom0_config": [
                True,
                True,
                False,
                False,
                False,
                True,
                True,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
            ]
        },
    ]
    return [
        Node(
            package="robot_localization",
            condition=UnlessCondition(LaunchConfiguration("sim")),
            executable="ekf_node",
            name="localization",
            parameters=[
                *common_localization_parameters,
                {"odom0": "/odom"},
                {"world_frame": "odom"},
                {"odom_frame": "odom"},
            ],
        ),
        Node(
            package="robot_localization",
            condition=IfCondition(LaunchConfiguration("sim")),
            executable="ekf_node",
            name="localization",
            parameters=[
                *common_localization_parameters,
                {"odom0": "/ego_racecar/odom"},
                {"world_frame": "map"},
                {"odom_frame": "map"},
                {"map_frame": "none"},
                {"base_link_frame": "ego_racecar/base_link"},
            ],
        ),
    ]
