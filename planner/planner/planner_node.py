#!/usr/bin/env python3
import time

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from graph_ltpl.Graph_LTPL import Graph_LTPL
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

# track_param = configparser.ConfigParser()
# if not track_param.read(toppath + "/params/driving_task.ini"):
#     raise ValueError('Specified online parameter config file does not exist or is empty!')


toppath = get_package_share_directory("planner")
track_specifier = "skir"

# define all relevant paths
path_dict = {
    "globtraj_input_path": toppath
    + "/inputs/traj_ltpl_cl/traj_ltpl_cl_"
    + track_specifier
    + ".csv",
    "graph_store_path": toppath + "/output/stored_graph.pckl",
    "ltpl_offline_param_path": toppath + "/params/ltpl_config_offline.ini",
    "ltpl_online_param_path": toppath + "/params/ltpl_config_online.ini",
}


# CAROFFSET = 0.3
# L =2
# KP = 0.3


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

        prefix = "/ego_racecar" if self.opponent else "/opp_racecar"
        self.sub_odom = self.create_subscription(
            Odometry,
            f"{prefix}/odom" if self.sim else "/pf/pose/odom",
            self.odom_callback,
            1,
        )
        self.opponent_topic = self.create_subscription(
            Odometry, f"/ego_racecar/opp_odom", self.opponent_callback, 1
        )

        # intialize graph_ltpl-class
        self.ltpl_obj = Graph_LTPL(
            path_dict=path_dict, visual_mode=True, log_to_file=False
        )

        # calculate offline graph
        self.ltpl_obj.graph_init()

        # set start pose based on first point in provided reference-line
        self.selected_action = "straight"

        # set start pos
        self.initialized = False

    def odom_callback(self, odom_msg: Odometry):
        position = odom_msg.pose.pose.position
        orientation = odom_msg.pose.pose.orientation
        self.position = np.array([position.x, position.y])
        self.velocity = odom_msg.twist.twist.linear.x
        self.heading = (
            Rotation.from_quat(
                [orientation.x, orientation.y, orientation.z, orientation.w]
            ).as_euler("xyz")[2]
            - np.pi / 2
        )

    def opponent_callback(self, odom_msg: Odometry):
        position = odom_msg.pose.pose.position
        orientation = odom_msg.pose.pose.orientation
        self.opponent_position = np.array([position.x, position.y])
        self.opponent_velocity = odom_msg.twist.twist.linear.x
        self.opponent_heading = (
            Rotation.from_quat(
                [orientation.x, orientation.y, orientation.z, orientation.w]
            ).as_euler("xyz")[2]
            - np.pi / 2
        )

    def select_action(self, trajectory_set):
        for selected_action in [
            "straight",
            "left",
            "right",
            "follow",
        ]:
            if selected_action in trajectory_set.keys():
                return selected_action
        raise RuntimeError("No action found.")

    def get_objects(self):
        obj1 = {
            "id": 0,  # integer id of the object
            "type": "physical",  # type 'physical' (only class implemented so far)
            "X": self.opponent_position[0],  # x coordinate
            "Y": self.opponent_position[1],  # y coordinate
            "theta": 0,  # orientation (north = 0.0)
            "v": 0.0,  # velocity along theta
            "length": 0.33,  # length of the object
            "width": 0.31,  # width of the object
        }
        return [obj1]

    def timer_callback(self):
        if (
            not self.initialized
            and self.position is not None
            and self.heading is not None
            and self.velocity is not None
        ):
            self.initialized = self.ltpl_obj.set_startpos(
                pos_est=self.position, heading_est=self.heading, vel_est=self.velocity
            )

        self.ltpl_obj.calc_paths(
            prev_action_id=self.selected_action, object_list=self.get_objects()
        )

        # -- CALCULATE VELOCITY PROFILE AND RETRIEVE TRAJECTORIES ----------------------------------------------------------
        # pos_est:[x, y]
        # vel_est:float
        traj_set = self.ltpl_obj.calc_vel_profile(
            pos_est=self.position, vel_est=self.velocity
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
