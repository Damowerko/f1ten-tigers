#!/usr/bin/env python3
import math
import time

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from graph_ltpl.Graph_LTPL import Graph_LTPL
from nav_msgs.msg import Odometry
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray
from tigerstack.mpc.visualize import points_to_arrow_markers
from visualization_msgs.msg import Marker, MarkerArray

# track_param = configparser.ConfigParser()
# if not track_param.read(toppath + "/params/driving_task.ini"):
#     raise ValueError('Specified online parameter config file does not exist or is empty!')


toppath = get_package_share_directory("planner")

# define all relevant paths
path_dict = {
    "globtraj_input_path": toppath + "/config/traj_ltpl_cl.csv",
    "graph_store_path": toppath + "/stored_graph.pckl",
    "ltpl_offline_param_path": toppath + "/config/ltpl_config_offline.ini",
    "ltpl_online_param_path": toppath + "/config/ltpl_config_online.ini",
}


class Planner(Node):
    """
    Implement Pure Pursuit on the car
    This is just a template, you are free to implement your own node!
    """

    def __init__(self):
        super().__init__("planner_node")  # type: ignore

        self.sim = bool(self.declare_parameter("sim", True).value)

        self.timer = self.create_timer(0.05, self.timer_callback)
        self.publisher_array = self.create_publisher(Float32MultiArray, "/path", 1)

        odom_topic = "/ego_racecar/odom" if self.sim else "/pf/pose/odom"
        self.sub_odom = self.create_subscription(
            Odometry, odom_topic, self.odom_callback, 1
        )

        self.sub_scan = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, 1
        )
        self.pub_visualize = self.create_publisher(MarkerArray, "~/visualize", 1)

        # intialize graph_ltpl-class
        self.ltpl_obj = Graph_LTPL(
            path_dict=path_dict, visual_mode=False, log_to_file=False
        )

        # calculate offline graph
        self.get_logger().info("Calculating offline graph...")
        self.ltpl_obj.graph_init()
        self.get_logger().info("Graph calculated!")

        # set start pose based on first point in provided reference-line
        self.selected_action = "straight"

        self.position = None

        # set start pos
        self.initialized = False

        self.filter_width = 20
        self.max_range = 2.0

    def odom_callback(self, odom_msg: Odometry):
        position = odom_msg.pose.pose.position
        orientation = odom_msg.pose.pose.orientation
        self.position = np.array([position.x, position.y])
        self.velocity = odom_msg.twist.twist.linear.x
        self.heading = (
            R.from_quat(
                [orientation.x, orientation.y, orientation.z, orientation.w]
            ).as_euler("xyz")[2]
            - np.pi / 2
        )

    def scan_callback(self, data: LaserScan):
        if self.position is None:
            return

        # Find angles between self.angle_min and self.angle_max
        angles = np.arange(data.angle_min, data.angle_max, data.angle_increment)
        ranges = np.asarray(data.ranges)
        ranges = np.nan_to_num(ranges)
        ranges = np.clip(ranges, 0, data.range_max)

        ranges = np.convolve(
            ranges, np.ones(self.filter_width) / self.filter_width, "same"
        )

        t = np.array([self.position[0], self.position[1]])
        angle_offset = self.heading + np.pi / 2
        laser_positions = (
            t
            + np.array(
                [
                    ranges * np.cos(angles + angle_offset),
                    ranges * np.sin(angles + angle_offset),
                ]
            ).T
        )

        self.obstacles = []
        for i in range(0, len(laser_positions), self.filter_width):
            self.obstacles += [
                {
                    "id": 0,  # integer id of the object
                    "type": "physical",  # type 'physical' (only class implemented so far)
                    "X": laser_positions[i, 0],  # x coordinate
                    "Y": laser_positions[i, 1],  # y coordinate
                    "theta": 0.0,  # orientation (north = 0.0)
                    "v": 0.0,  # velocity along theta
                    "length": 0.1,  # length of the object
                    "width": 0.1,  # width of the object
                }
            ]

        # occupied = ranges < self.max_range
        # # pad ranges to find edges at start and end
        # ranges = np.pad(occupied, 1, mode="constant", constant_values=False)

        # diff = np.diff(1 * ranges)
        # rising_edge = (diff > 0).nonzero()[0]
        # falling_edge = (diff < 0).nonzero()[0]

        # self.obstacles = []
        # for i in range(len(rising_edge)):
        #     start_idx = rising_edge[i]
        #     end_idx = min(falling_edge[i], len(ranges) - 1)

        #     center = np.mean(laser_positions[start_idx:end_idx], axis=0)

        #     self.obstacles += [
        #         {
        #             "id": 0,  # integer id of the object
        #             "type": "physical",  # type 'physical' (only class implemented so far)
        #             "X": center[0],  # x coordinate
        #             "Y": center[1],  # y coordinate
        #             "theta": 0.0,  # orientation (north = 0.0)
        #             "v": 0.0,  # velocity along theta
        #             "length": 0.3,  # length of the object
        #             "width": 0.3,  # width of the object
        #         }
        #     ]

    def select_action(self, trajectory_set):
        for selected_action in [
            "right",
            "straight",
            "left",
            "follow",
        ]:
            if selected_action in trajectory_set.keys():
                return selected_action
        raise RuntimeError("No action found.")

    def get_objects(self):
        return self.obstacles

    def timer_callback(self):
        if not self.initialized and self.position is not None:
            self.initialized = self.ltpl_obj.set_startpos(
                pos_est=self.position, heading_est=self.heading, vel_est=self.velocity
            )
        if not self.initialized or self.position is None:
            return

        self.ltpl_obj.calc_paths(
            prev_action_id=self.selected_action,
            object_list=self.get_objects(),
        )

        # -- CALCULATE VELOCITY PROFILE AND RETRIEVE TRAJECTORIES ----------------------------------------------------------
        # pos_est:[x, y]
        # vel_est:float
        traj_set = self.ltpl_obj.calc_vel_profile(
            pos_est=self.position,
            vel_est=self.velocity,
            vel_max=20.0,
            gg_scale=1.0,
            local_gg=(10.0, 6.0),
            ax_max_machines=np.array([[0.0, 10.0], [20.0, 10.0]]),
            safety_d=0.5,
            incl_emerg_traj=False,
        )[0]

        self.selected_action = self.select_action(traj_set)

        # [s, x, y, heading, curvature, vx, ax]
        trajectory = traj_set[self.selected_action][0]

        array_msg = Float32MultiArray()
        array_msg.data = list(trajectory.flatten())
        self.publisher_array.publish(array_msg)

        # -- LIVE PLOT (if activated) --------------------------------------------------------------------------------------
        self.ltpl_obj.visual()


def main(args=None):
    rclpy.init(args=args)
    planner_node = Planner()
    print("Planner Initialized")
    rclpy.spin(planner_node)
    planner_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
