#!/usr/bin/env python3
import time

import graph_ltpl
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

# track_param = configparser.ConfigParser()
# if not track_param.read(toppath + "/params/driving_task.ini"):
#     raise ValueError('Specified online parameter config file does not exist or is empty!')


toppath = "f1ten_ws/src/f1ten-tigers/planner"
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
        self.publisher_array = self.create_publisher(Float32MultiArray, "/localpath", 1)
        odom_topic = "/ego_racecar/odom" if self.sim else "/pf/pose/odom"
        self.sub_odom = self.create_subscription(
            Odometry, odom_topic, self.odom_callback, 1
        )
        self.sub_odom
        # ----------------------------------------------------------------------------------------------------------------------
        # INITIALIZATION AND OFFLINE PART --------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------------------

        # intialize graph_ltpl-class
        self.ltpl_obj = graph_ltpl.Graph_LTPL.Graph_LTPL(
            path_dict=path_dict, log_to_file=False
        )

        # calculate offline graph
        self.ltpl_obj.graph_init()

        # set start pose based on first point in provided reference-line
        refline = (
            graph_ltpl.imp_global_traj.src.import_globtraj_csv.import_globtraj_csv(
                import_path=path_dict["globtraj_input_path"]
            )[0]
        )
        self.pos_est = refline[0, :]
        heading_est = (
            np.arctan2(np.diff(refline[0:2, 1]), np.diff(refline[0:2, 0])) - np.pi / 2
        )
        self.vel_est = 0.0

        # set start pos
        self.ltpl_obj.set_startpos(pos_est=self.pos_est, heading_est=heading_est)
        self.traj_set = {"straight": None}
        self.tic = time.time()

        self.vel_est = 0
        self.pos_est = [0, 0]

    def odom_callback(self, odom_msg):
        self.vel_est = odom_msg.twist.twist.linear.x
        position = odom_msg.pose.pose.position
        self.pos_est = [position.x, position.y]

    def timer_callback(self):
        for sel_action in [
            "right",
            "left",
            "straight",
            "follow",
        ]:  # try to force 'right', else try next in list
            if sel_action in self.traj_set.keys():
                break

        self.ltpl_obj.calc_paths(prev_action_id=sel_action, object_list=[])

        self.tic = time.time()

        # -- CALCULATE VELOCITY PROFILE AND RETRIEVE TRAJECTORIES ----------------------------------------------------------
        # pos_est:[x, y]
        # vel_est:float
        self.traj_set = self.ltpl_obj.calc_vel_profile(
            pos_est=self.pos_est, vel_est=self.vel_est
        )[0]
        # print(len(self.traj_set["straight"][0]))
        print(
            "x={:.2f},y={:.2f},v={:.2f}".format(
                self.pos_est[0], self.pos_est[1], self.vel_est
            )
        )

        # [s, x, y, heading, curvature, vx, ax]
        traj_set = self.traj_set["straight"][0]
        print("length of points=", len(traj_set))
        traj_set[:, 3] = traj_set[:, 3] + np.pi / 2
        traj_set = np.array(traj_set)[:, [1, 2, 5, 3]]
        while len(traj_set) != 9:
            traj_set = np.vstack((traj_set, traj_set[-1]))

        # print("shape of traj set=",traj_set.shape)
        print("({:.2f},{:.2f})".format(traj_set[0][0], traj_set[0][1]))
        array_msg = Float32MultiArray()
        array_msg.data = list(traj_set.flatten())
        # array_msg.data[2]= list(traj_set[2])
        self.publisher_array.publish(array_msg)
        # print(traj_set)
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
