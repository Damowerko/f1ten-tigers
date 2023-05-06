from copy import copy
from typing import List, Optional

import numpy as np
import numpy.typing as npt
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray, PoseStamped, Quaternion
from nav_msgs.msg import Odometry
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from std_msgs.msg import Bool


def quaternion_to_matrix(quaternion: Quaternion):
    return Rotation.from_quat(
        [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
    ).as_matrix()


class PurePursuit(Node):
    def __init__(self):
        super().__init__("pure_pursuit")  # type: ignore

        # parameters
        self.kp: float = self.declare_parameter("kp", 0.1).value  # type: ignore
        self.waypoint_distance: float = self.declare_parameter("waypoint_distance", 0.3).value  # type: ignore
        self.sim = self.declare_parameter("sim", True).value  # type: ignore

        if self.sim:
            self.pose_sub_ = self.create_subscription(
                Odometry, "/ego_racecar/odom", self.sim_pose_callback, 1
            )
        else:
            self.pose_sub_ = self.create_subscription(
                PoseStamped, "/pf/viz/inferred_pose", self.pose_callback, 1
            )
            self.publish_frame = "laser"

        self.colission_sub = self.create_subscription(
            Bool, "rrt/collision", self.collision_callback, 1
        )
        self.colission = False

        self.waypoints: List[npt.NDArray[np.floating]] = []
        self.current_waypoint_index: int = 0
        self.position: Optional[npt.NDArray[np.floating]] = None
        self.rotation_matrix: Optional[npt.NDArray[np.floating]] = None

        self.create_subscription(PoseArray, "/pure_pursuit/waypoints", self.path_callback, 1)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "drive", 1)
        self.current_waypoint_pub = self.create_publisher(
            PoseStamped, "/pure_pursuit/current_waypoint", 1
        )

        # create timer for pure pursuit
        self.create_timer(1 / 30, self.pure_pursuit)

    def sim_pose_callback(self, msg: Odometry) -> None:
        self.position = np.array(
            [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
            ]
        )
        self.rotation_matrix = quaternion_to_matrix(msg.pose.pose.orientation)

    def collision_callback(self, msg : Bool):
        self.colission = msg.data

    def pose_callback(self, msg: PoseStamped) -> None:
        self.position = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        )
        self.rotation_matrix = quaternion_to_matrix(msg.pose.orientation)

    def path_callback(self, msg: PoseArray) -> None:
        self.waypoints = [
            np.array([pose.position.x, pose.position.y, pose.position.z])
            for pose in msg.poses
        ]
        self.current_waypoint_index = 1

    def pure_pursuit(self) -> None:
        if not self.colission:
            return None
        waypoints = copy(self.waypoints)
        current_waypoint_index = copy(self.current_waypoint_index)

        if waypoints is None or self.position is None or self.rotation_matrix is None:
            return
        if current_waypoint_index >= len(waypoints):
            return

        # waypoints should be relative to the car
        waypoints = np.array(waypoints) - self.position
        # rotate waypoints by -heading
        waypoints = waypoints @ np.linalg.inv(self.rotation_matrix).T

        # get the first waypoint that is further than waypoint_distance away
        distances = np.linalg.norm(waypoints[current_waypoint_index:], axis=1)
        current_waypoint_index += int(np.argmax(distances > self.waypoint_distance))

        target = waypoints[current_waypoint_index]

        distance_squared = (target**2).sum()
        curvature = 2 * abs(target[1]) / distance_squared

        target_pose = PoseStamped()
        target_pose.header.frame_id = "ego_racecar/laser" if self.sim else "laser"
        target_pose.pose.position.x = target[0]
        target_pose.pose.position.y = target[1]
        self.current_waypoint_pub.publish(target_pose)

        drive_msg = AckermannDriveStamped()
        steering_angle = curvature * np.sign(target[1]) * self.kp
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = 3.0
        self.drive_pub.publish(drive_msg)


def main():
    rclpy.init()
    node = PurePursuit()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()