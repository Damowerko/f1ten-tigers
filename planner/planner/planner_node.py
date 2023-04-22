#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive

from visualization_msgs.msg import MarkerArray,Marker
from tf2_msgs.msg import TFMessage
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray 
import numpy as np
import json
import time
# import configparser

import graph_ltpl

# ----------------------------------------------------------------------------------------------------------------------
# IMPORT (should not change) -------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# top level path (module directory)
# import os
# import sys
# toppath = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(toppath)

# track_param = configparser.ConfigParser()
# if not track_param.read(toppath + "/params/driving_task.ini"):
#     raise ValueError('Specified online parameter config file does not exist or is empty!')
toppath = "/sim_ws/src/planner"
track_specifier ="skir"

# define all relevant paths
path_dict = {'globtraj_input_path': toppath + "/inputs/traj_ltpl_cl/traj_ltpl_cl_" + track_specifier + ".csv",
             'graph_store_path': toppath + "/output/stored_graph.pckl",
             'ltpl_offline_param_path': toppath + "/params/ltpl_config_offline.ini",
             'ltpl_online_param_path': toppath + "/params/ltpl_config_online.ini"
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
        super().__init__('planner_node')
        # TODO: create ROS subscribers and publishers
        # self.publisher_marker = self.create_publisher(MarkerArray, 'visualization_marker_array', 10)
        # self.scan_sub = self.create_subscription(TFMessage,'tf',self.pose_callback_sim,10)
        # self.drive_pub = self.create_publisher(AckermannDriveStamped, 'drive', 10)
        # self.L = L
        # self.wypts = np.loadtxt(FILENAME,delimiter=",", dtype=str)
        # self.get_logger().info("waypoints size=%d"%self.wypts.shape[0])
        # self.markerMsg = MarkerArray()
        # self.kp = KP
        self.sim =True
        self.timer = self.create_timer(0.05, self.timer_callback)
        self.publisher_array=self.create_publisher(Float32MultiArray,"/localpath",1)
        odom_topic = "/ego_racecar/odom" if self.sim else "/pf/pose/odom"
        self.sub_odom = self.create_subscription(
            Odometry, odom_topic, self.odom_callback, 1
        )
        self.sub_odom
        # ----------------------------------------------------------------------------------------------------------------------
        # INITIALIZATION AND OFFLINE PART --------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------------------

        # intialize graph_ltpl-class
        self.ltpl_obj = graph_ltpl.Graph_LTPL.Graph_LTPL(path_dict=path_dict,
                                                    visual_mode=True,
                                                    log_to_file=False)

        # calculate offline graph
        self.ltpl_obj.graph_init()

        # set start pose based on first point in provided reference-line
        refline = graph_ltpl.imp_global_traj.src.\
            import_globtraj_csv.import_globtraj_csv(import_path=path_dict['globtraj_input_path'])[0]
        self.pos_est = refline[0, :]
        heading_est = np.arctan2(np.diff(refline[0:2, 1]), np.diff(refline[0:2, 0])) - np.pi / 2
        self.vel_est = 0.0

        # set start pos
        self.ltpl_obj.set_startpos(pos_est=self.pos_est,
                            heading_est=heading_est)
        self.traj_set = {'straight': None}
        self.tic = time.time()

        # self.tf_sub =   self.create_subscription(
        #     TFMessage,
        #     'tf',
        #     self.pose_callback_sim,
        #     10)
        # self.tf_sub

        # self.odom_sub = self.create_subscription(
        #     Odometry,
        #     'ego_racecar/odom',     # Self defined topic
        #     self.odom_callback,
        #     10)
        # self.odom_sub

        self.vel_est=0
        self.pos_est=[0,0]

    def odom_callback(self, odom_msg):
        # self.get_logger().info("odom twist lx:%.5f"% odom_msg.twist.twist.linear.x)
        self.vel_est= odom_msg.twist.twist.linear.x
        position = odom_msg.pose.pose.position
        self.pos_est = [position.x,position.y]

    # def pose_callback_sim(self, pose_msg):
    #     for tf in pose_msg.transforms:
    #         if tf.header.frame_id=="map":
    #             self.pos_est = [tf.transform.translation.x,tf.transform.translation.y]


    def timer_callback(self):
        for sel_action in ["right", "left", "straight", "follow"]:  # try to force 'right', else try next in list
            if sel_action in self.traj_set.keys():
                break

        # -- CALCULATE PATHS FOR NEXT TIMESTAMP ----------------------------------------------------------------------------
        self.ltpl_obj.calc_paths(prev_action_id=sel_action,
                            object_list=[])

        # -- GET POSITION AND VELOCITY ESTIMATE OF EGO-VEHICLE -------------------------------------------------------------
        # (here: simulation dummy, replace with actual sensor readings)
        # if self.traj_set[sel_action] is not None:
        #     self.pos_est, self.vel_est = graph_ltpl.testing_tools.src.vdc_dummy.\
        #         vdc_dummy(pos_est=self.pos_est,
        #                 last_s_course=(self.traj_set[sel_action][0][:, 0]),
        #                 last_path=(self.traj_set[sel_action][0][:, 1:3]),
        #                 last_vel_course=(self.traj_set[sel_action][0][:, 5]),
        #                 iter_time=time.time() - self.tic)
        self.tic = time.time()

        # -- CALCULATE VELOCITY PROFILE AND RETRIEVE TRAJECTORIES ----------------------------------------------------------
        # pos_est:[x, y]
        # vel_est:float
        self.traj_set = self.ltpl_obj.calc_vel_profile(pos_est=self.pos_est,
                                            vel_est=self.vel_est)[0]
        # print(len(self.traj_set["straight"][0]))
        print("x={:.2f},y={:.2f},v={:.2f}".format(self.pos_est[0],self.pos_est[1],self.vel_est))

        #[s, x, y, heading, curvature, vx, ax]
        traj_set = self.traj_set["straight"][0]
        print("length of points=",len(traj_set))
        traj_set[:, 3] = traj_set[:, 3] + np.pi / 2
        traj_set = np.array(traj_set)[:,[1,2,5,3]]
        while len(traj_set)!=9:
            traj_set=np.vstack((traj_set,traj_set[-1]))

        # print("shape of traj set=",traj_set.shape)
        print("({:.2f},{:.2f})".format(traj_set[0][0],traj_set[0][1]))
        array_msg = Float32MultiArray()
        array_msg.data=(list(traj_set.flatten()))
        # array_msg.data[2]= list(traj_set[2])
        self.publisher_array.publish(array_msg)
        # print(traj_set)
        # -- LIVE PLOT (if activated) --------------------------------------------------------------------------------------
        self.ltpl_obj.visual()


def main(args=None):
    rclpy.init(args=args)
    print("planner Initialized")
    planner_node = Planner()
    # for i in range(0,1):
    #     planner_node.visualizeWaypoints()
    rclpy.spin(planner_node)

    planner_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
