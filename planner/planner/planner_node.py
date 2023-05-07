#!/usr/bin/env python3
import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from ament_index_python.packages import get_package_share_directory
from graph_ltpl.Graph_LTPL import Graph_LTPL
from nav_msgs.msg import Odometry
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray
from tigerstack.mpc.visualize import point_to_sphere_marker
from visualization_msgs.msg import Marker, MarkerArray

toppath = get_package_share_directory("planner")

# define all relevant paths
path_dict = {
    "globtraj_input_path": toppath + "/config/traj_ltpl_cl_levine.csv",
    # "graph_store_path": "/sim_ws/src/f1ten-tigers/planner/output/stored_graph.pckl",
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
        self.brake = bool(self.declare_parameter("brake", False).value)
        self.safety_distance = float(
            self.declare_parameter("safety_distance", 1.0).value  # type: ignore
        )
        self.detection_method = str(
            self.declare_parameter("obstacle_method", "downsample").value
        )

        if self.detection_method == "sim" and not self.sim:
            raise ValueError(
                "Obstacle detection method is set to 'sim' but sim is set to False."
            )

        # self.timer = self.create_timer(0.05, self.timer_callback)
        self.publisher_array = self.create_publisher(Float32MultiArray, "/path", 1)
        self.pub_obs = self.create_publisher(MarkerArray, "~/visualize_obs", 1)
        odom_topic = "/ego_racecar/odom" if self.sim else "/pf/pose/odom"
        self.sub_odom = self.create_subscription(
            Odometry, odom_topic, self.odom_callback, 1
        )

        if self.sim:
            self.opponent_position = np.array([0.0, 0.0])
            self.opponent_topic = self.create_subscription(
                Odometry, "/ego_racecar/opp_odom", self.opponent_callback, 1
            )

        self.sub_scan = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, 1
        )
        self.pub_visualize = self.create_publisher(MarkerArray, "~/visualize", 1)

        # intialize graph_ltpl-class
        self.ltpl_obj = Graph_LTPL(
            path_dict=path_dict, visual_mode=True, log_to_file=False
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

        self.filter_width = 3  # 20
        self.max_range = 4.0
        self.obstacles = []
        self.execute_count = 0
        self.total_runtime = 0.0

        self.transform_position = None
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "drive", 5)

    def odom_callback(self, odom_msg: Odometry):
        position = odom_msg.pose.pose.position
        orientation = odom_msg.pose.pose.orientation
        self.transform_position = orientation
        self.position = np.array([position.x, position.y])
        self.velocity = odom_msg.twist.twist.linear.x
        # self.flag_brake=False if self.velocity<0.1 else True
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

    def emergency_brake(self):
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = 0.0
        self.drive_pub.publish(drive_msg)
        self.get_logger().warn("Emergency brake activated.")

    def scan_to_positions(self, position, ranges, angles):
        ranges = np.convolve(
            ranges, np.ones(self.filter_width) / self.filter_width, "same"
        )
        t = np.array([position[0], position[1]])
        angle_offset = self.heading + np.pi / 2
        scan_positions = (
            t
            + np.array(
                [
                    ranges * np.cos(angles + angle_offset),
                    ranges * np.sin(angles + angle_offset),
                ]
            ).T
        )
        return scan_positions

    def scan_callback(self, data: LaserScan):
        if self.position is None:
            return

        # parse lidar data
        angles = np.arange(data.angle_min, data.angle_max, data.angle_increment)
        ranges = np.asarray(data.ranges)
        ranges = np.nan_to_num(ranges)

        # safety node
        if (
            self.brake
            and (
                (np.abs(angles) < np.pi / 24) & (ranges < self.safety_distance / 2)
            ).any()
        ):
            self.emergency_brake()

        # clear obstacles
        self.obstacles = []
        if self.detection_method == "sim":
            self.obstacles = [
                {
                    "id": 0,  # integer id of the object
                    "type": "physical",  # type 'physical' (only class implemented so far)
                    "X": self.opponent_position[0],  # x coordinate
                    "Y": self.opponent_position[1],  # y coordinate
                    "theta": self.opponent_heading,  # orientation (north = 0.0)
                    "v": self.opponent_velocity,  # velocity along theta
                    "length": 0.3,  # length of the object
                    "width": 0.3,  # width of the object
                }
            ]
        elif self.detection_method == "downsample":
            scan_positions = self.scan_to_positions(self.position, ranges, angles)
            for i in range(0, len(scan_positions), self.filter_width):
                self.obstacles += [
                    {
                        "id": 0,  # integer id of the object
                        "type": "physical",  # type 'physical' (only class implemented so far)
                        "X": scan_positions[i, 0],  # x coordinate
                        "Y": scan_positions[i, 1],  # y coordinate
                        "theta": 0.0,  # orientation (north = 0.0)
                        "v": 0.0,  # velocity along theta
                        "length": 0.3,  # length of the object
                        "width": 0.3,  # width of the object
                    }
                ]
        elif self.detection_method == "threshold":
            scan_positions = self.scan_to_positions(self.position, ranges, angles)
            # pad ranges to find edges at start and end
            occupied = ranges < self.max_range
            occupied = np.pad(occupied, 1, mode="constant", constant_values=False)
            diff = np.diff(1 * occupied)
            rising_edge = (diff > 0).nonzero()[0]
            falling_edge = (diff < 0).nonzero()[0]
            self.obstacles = []

            for i in range(len(rising_edge)):
                start_idx = rising_edge[i]
                end_idx = min(falling_edge[i], len(ranges) - 1)
                center = np.mean(scan_positions[start_idx:end_idx], axis=0)
                self.obstacles += [
                    {
                        "id": 0,  # integer id of the object
                        "type": "physical",  # type 'physical' (only class implemented so far)
                        "X": center[0],  # x coordinate
                        "Y": center[1],  # y coordinate
                        "theta": self.heading,  # orientation (north = 0.0)
                        "v": self.velocity - 1,  # velocity along theta
                        "length": 0.3,  # length of the object
                        "width": 0.3,  # width of the object
                    }
                ]
        elif self.detection_method == "diff":
            diff_threshold = 0.5
            max_object_width = 80  # how many number of lidar points
            cliff_index = (np.abs(np.diff(ranges)) > diff_threshold).nonzero()[0]
            if len(cliff_index) == 0:
                return
            prev_i = cliff_index[0]
            # convert potential objects in lidar to map frame:
            cur_pos = np.array([self.position[0], self.position[1]])
            angle_offset = self.heading + np.pi / 2
            self.obstacles = []
            for index in cliff_index:
                if index == cliff_index[0]:
                    continue
                mid = int((prev_i + index) / 2)
                if np.abs(angles[mid]) > np.pi / 4:
                    continue
                if index - prev_i < max_object_width and ranges[mid] < self.max_range:
                    self.obstacles += [
                        {
                            "id": 0,  # integer id of the object
                            "type": "physical",  # type 'physical' (only class implemented so far)
                            "X": cur_pos[0]
                            + ranges[mid]
                            * np.cos(angles[mid] + angle_offset),  # x coordinate
                            "Y": cur_pos[1]
                            + ranges[mid]
                            * np.sin(angles[mid] + angle_offset),  # y coordinate
                            "theta": self.heading,  # orientation (north = 0.0)
                            "v": self.velocity,  # velocity along theta
                            "length": 0.3,  # length of the object
                            "width": 0.3,  # width of the object
                        }
                    ]
                prev_i = index
        elif self.detection_method == "canny":
            raise NotImplementedError()  # TODO: merge into this branch
        elif self.detection_method == "peak":
            raise NotImplementedError()  # TODO: merge into this branch
        else:
            raise ValueError(f"Unknown detection method {self.detection_method}.")

        self.visualize_obstacles()
        self.timer_callback()

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

    def visualize_obstacles(self):
        # delete all markers in namespace 'planner'
        marker = Marker()
        marker.id = 0
        marker.ns = "planner"
        marker.action = Marker.DELETEALL
        msg = MarkerArray()
        msg.markers = [marker]
        self.pub_obs.publish(msg)

        msg = MarkerArray()
        msg.markers = []
        for id, obstacle in enumerate(self.obstacles):
            position = np.array([obstacle["X"], obstacle["Y"]])
            marker = point_to_sphere_marker(
                position, ns="planner", id=id, color=(0.8, 0.0, 0.8), scale=0.3, msg=msg
            )
            msg.markers.append(marker)
        self.pub_obs.publish(msg)

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
        self.obstacles = []

        # -- CALCULATE VELOCITY PROFILE AND RETRIEVE TRAJECTORIES ----------------------------------------------------------
        traj_set = self.ltpl_obj.calc_vel_profile(
            pos_est=self.position,
            vel_est=self.velocity,
            vel_max=20.0,
            gg_scale=1.0,
            local_gg=(10.0, 6.0),
            ax_max_machines=np.array([[0.0, 10.0], [20.0, 10.0]]),
            safety_d=self.safety_distance,
            incl_emerg_traj=False,
        )[0]
        self.prev_vel = self.velocity

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
    rclpy.spin(planner_node)
    planner_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
