#!/usr/bin/env python3
import time

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from graph_ltpl.Graph_LTPL import Graph_LTPL
from nav_msgs.msg import Odometry
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
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
            "straight",
            "right",
            "left",
            "follow",
        ]:
            if selected_action in trajectory_set.keys():
                return selected_action
        raise RuntimeError("No action found.")

    def get_objects(self):
        if self.opponent_position is None:
            return []
        opponent = {
            "id": 0,  # integer id of the object
            "type": "physical",  # type 'physical' (only class implemented so far)
            "X": self.opponent_position[0],  # x coordinate
            "Y": self.opponent_position[1],  # y coordinate
            "theta": self.opponent_heading,  # orientation (north = 0.0)
            "v": self.opponent_velocity,  # velocity along theta
            "length": 0.1,  # length of the object
            "width": 0.1,  # width of the object
        }
        return [opponent]

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
