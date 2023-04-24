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
        self.opponent_topic = self.create_subscription(
            Odometry, "/ego_racecar/opp_odom", self.opponent_callback, 1
        )

        self.sub_scan = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, 1
        )

        # intialize graph_ltpl-class
        self.ltpl_obj = Graph_LTPL(
            path_dict=path_dict, visual_mode=False, log_to_file=False
        )

        # calculate offline graph
        self.ltpl_obj.graph_init()

        # set start pose based on first point in provided reference-line
        self.selected_action = "straight"

        self.position = None
        self.opponent_position = None

        # set start pos
        self.initialized = False

        self.max_range = 4.0
        self.filter_width = 5

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

    def process_lidar(self, ranges, angles):
        ranges = np.nan_to_num(ranges)
        ranges = np.convolve(
            ranges, np.ones(self.filter_width) / self.filter_width, "same"
        )
        # set far scans to max value
        ranges[ranges > self.max_range] = 100
        diff = np.diff(ranges)
        # might break for obstaces on either end of the scan
        rising_edge = (diff < -2 * self.max_range).nonzero()[0]
        falling_edge = (diff > 2 * self.max_range).nonzero()[0] + 1

        n_edges = min(len(rising_edge), len(falling_edge))
        rising_edge = rising_edge[:n_edges]
        falling_edge = falling_edge[:n_edges]

        center_indices = (falling_edge + rising_edge) // 2
        positions = np.array(
            [
                ranges[center_indices] * np.cos(angles[center_indices]),
                ranges[center_indices] * np.sin(angles[center_indices]),
                0,
            ]
        )
        t = np.array(
            [
                self.position[0],
                self.position[1],
                0,
            ]
        )
        r = R.from_euler("z", self.heading, degrees=False).as_matrix()
        map_obs_poses = r @ positions.reshape(-1, 3) + t
        return map_obs_poses

    def scan_callback(self, data: LaserScan):
        if self.position is None:
            return

        # Find angles between -self.angle_max and self.angle_max
        angles = np.arange(data.angle_min, data.angle_max, data.angle_increment)
        ranges = np.asarray(data.ranges)

        # Get angles between -90 and 90 degrees
        mask = np.abs(angles) < np.pi / 2

        angles = angles[mask]
        ranges = ranges[mask]

        self.obstacles = self.process_lidar(ranges, angles)

    def opponent_callback(self, odom_msg: Odometry):
        position = odom_msg.pose.pose.position
        orientation = odom_msg.pose.pose.orientation
        self.opponent_position = np.array([position.x, position.y])
        self.opponent_velocity = odom_msg.twist.twist.linear.x
        self.opponent_heading = (
            R.from_quat(
                [orientation.x, orientation.y, orientation.z, orientation.w]
            ).as_euler("xyz")[2]
            - np.pi / 2
        )

    def select_action(self, trajectory_set):
        for selected_action in [
            "right",
            "left",
            "straight",
            "follow",
        ]:
            if selected_action in trajectory_set.keys():
                return selected_action
        raise RuntimeError("No action found.")

    def get_objects(self):
        if self.opponent_position is None:
            return []
        # opponent = {
        #     "id": 0,  # integer id of the object
        #     "type": "physical",  # type 'physical' (only class implemented so far)
        #     "X": self.opponent_position[0],  # x coordinate
        #     "Y": self.opponent_position[1],  # y coordinate
        #     "theta": self.opponent_heading,  # orientation (north = 0.0)
        #     "v": self.opponent_velocity,  # velocity along theta
        #     "length": 0.1,  # length of the object
        #     "width": 0.1,  # width of the object
        # }
        objects = []
        if self.obstacles is None:
            return []
        for obstacle in self.obstacles:
            opponent = {
                "id": 0,  # integer id of the object
                "type": "physical",  # type 'physical' (only class implemented so far)
                "X": obstacle[0],  # x coordinate
                "Y": obstacle[1],  # y coordinate
                "theta": 0,  # orientation (north = 0.0)
                "v": 0,  # velocity along theta
                "length": 0.1,  # length of the object
                "width": 0.1,  # width of the object
            }
            objects += [opponent]
        return objects

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
