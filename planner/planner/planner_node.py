#!/usr/bin/env python3
import time

import numpy as np
import rclpy
from graph_ltpl.Graph_LTPL import Graph_LTPL
from graph_ltpl.imp_global_traj.src.import_globtraj_csv import import_globtraj_csv
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

# track_param = configparser.ConfigParser()
# if not track_param.read(toppath + "/params/driving_task.ini"):
#     raise ValueError('Specified online parameter config file does not exist or is empty!')


toppath = "/home/damow/ese615/f1ten_ws/src/f1ten-tigers/planner"
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

        self.sim = True
        self.timer = self.create_timer(0.05, self.timer_callback)
        self.publisher_array = self.create_publisher(Float32MultiArray, "/path", 1)
        odom_topic = "/ego_racecar/odom" if self.sim else "/pf/pose/odom"
        self.sub_odom = self.create_subscription(
            Odometry, odom_topic, self.odom_callback, 1
        )
        self.sub_odom
        # ----------------------------------------------------------------------------------------------------------------------
        # INITIALIZATION AND OFFLINE PART --------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------------------

        # intialize graph_ltpl-class
        self.ltpl_obj = Graph_LTPL(
            path_dict=path_dict, visual_mode=True, log_to_file=False
        )

        # calculate offline graph
        self.ltpl_obj.graph_init()

        # set start pose based on first point in provided reference-line
        refline = import_globtraj_csv(import_path=path_dict["globtraj_input_path"])[0]
        self.pos_est = refline[0, :]
        heading_est = float(
            np.arctan2(np.diff(refline[0:2, 1]), np.diff(refline[0:2, 0])) - np.pi / 2
        )
        self.vel_est = 0.0
        self.selected_action = "straight"

        # set start pos
        self.ltpl_obj.set_startpos(pos_est=self.pos_est, heading_est=heading_est)
        self.traj_set = {"straight": None}

    def odom_callback(self, odom_msg):
        self.vel_est = odom_msg.twist.twist.linear.x
        position = odom_msg.pose.pose.position
        self.pos_est = np.array([position.x, position.y])

    def choose_action(self):
        for sel_action in [
            "straight",
            "left",
            "right",
            "follow",
        ]:
            if sel_action in self.traj_set.keys():
                return sel_action
        raise RuntimeError("No action found.")

    def get_objects(self):
        obj1 = {
            "id": 0,  # integer id of the object
            "type": "physical",  # type 'physical' (only class implemented so far)
            "X": 2.5,  # x coordinate
            "Y": 1.0,  # y coordinate
            "theta": 0,  # orientation (north = 0.0)
            "v": 0.0,  # velocity along theta
            "length": 0.5,  # length of the object
            "width": 0.5,  # width of the object
        }
        return [obj1]

    def timer_callback(self):
        self.ltpl_obj.calc_paths(
            prev_action_id=self.selected_action, object_list=self.get_objects()
        )

        # -- CALCULATE VELOCITY PROFILE AND RETRIEVE TRAJECTORIES ----------------------------------------------------------
        # pos_est:[x, y]
        # vel_est:float
        self.traj_set = self.ltpl_obj.calc_vel_profile(
            pos_est=self.pos_est, vel_est=self.vel_est
        )[0]

        self.selected_action = self.choose_action()

        # [s, x, y, heading, curvature, vx, ax]
        trajectory = self.traj_set[self.selected_action][0]

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
