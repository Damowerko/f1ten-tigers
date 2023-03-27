# MIT License

# Copyright (c) 2020 Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # config and args
    localize_config = os.path.join(
        get_package_share_directory("tigerstack"), "config", "localize.yaml"
    )
    localize_config_dict = yaml.safe_load(open(localize_config, "r"))
    map_name = localize_config_dict["map_server"]["ros__parameters"]["map"]
    localize_la = DeclareLaunchArgument(
        "localize_config",
        default_value=localize_config,
        description="Localization configs",
    )
    ld = LaunchDescription([localize_la])

    # nodes
    pf_node = Node(
        package="particle_filter",
        executable="particle_filter",
        name="particle_filter",
        parameters=[LaunchConfiguration("localize_config")],
    )
    map_server_node = Node(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        parameters=[
            {
                "yaml_filename": os.path.join(
                    get_package_share_directory("tigerstack"),
                    "maps",
                    map_name + ".yaml",
                )
            },
            {"topic": "map"},
            {"frame_id": "map"},
            {"output": "screen"},
            {"use_sim_time": True},
        ],
    )
    nav_lifecycle_node = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_localization",
        output="screen",
        parameters=[
            {"use_sim_time": True},
            {"autostart": True},
            {"node_names": ["map_server"]},
        ],
    )
    imu_framer = Node(
        package="tigerstack",
        executable="imu_framer",
        name="imu_framer",
    )
    ekf_node = Node(
        package="robot_localization",
        executable="ekf_node",
        name="localization",
        parameters=[
            {"imu0": "/sensors/imu/raw_framed"},
            {"odom0": "/odom"},
            {"world_frame": "odom"},
            {"odom_frame": "odom"},
            {"frequency": 50.0},
            {"two_d_mode": True},
            {"publish_tf": False},
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
        ],
    )

    # finalize
    ld.add_action(ekf_node)
    ld.add_action(nav_lifecycle_node)
    ld.add_action(map_server_node)
    # ld.add_action(imu_framer)
    # ld.add_action(pf_node)

    return ld
