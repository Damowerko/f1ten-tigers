#!/usr/bin/env python3
from collections import deque
from typing import List, Optional

import numpy as np
import numpy.typing as npt
import rclpy
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32


class SafetyNode(Node):
    """
    The class that handles emergency braking.
    """

    def __init__(self):
        """
        One publisher should publish to the /drive topic with a AckermannDriveStamped drive message.

        You should also subscribe to the /scan topic to get the LaserScan messages and
        the /ego_racecar/odom topic to get the current speed of the vehicle.

        The subscribers should use the provided odom_callback and scan_callback as callback methods

        NOTE that the x component of the linear velocity in odom is the speed
        """
        super().__init__("safety_node")  # type: ignore

        # car dimensions
        self.car_width = float(self.declare_parameter("car_width", 0.265).value)  # type: ignore
        self.car_front_distance = float(self.declare_parameter("car_front_distance", 0.165).value)  # type: ignore
        self.car_rear_distance = float(self.declare_parameter("car_rear_distance", 0.34).value)  # type: ignore

        self.ttc_filter_length = int(self.declare_parameter("ttc_filter_length", 3).value)  # type: ignore
        self.ttc_threshold = float(self.declare_parameter("ttc_threshold", 0.2).value)  # type: ignore

        self.ttc_deque: deque[float] = deque(maxlen=self.ttc_filter_length)
        self.add_on_set_parameters_callback(self.parameters_callback)

        self.speed: Optional[float] = None

        self.subscriber_scan = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, 10
        )
        self.subscriber_odom = self.create_subscription(
            Odometry, "/ego_racecar/odom", self.odom_callback, 10
        )
        self.publisher_drive = self.create_publisher(
            AckermannDriveStamped, "/drive", 10
        )
        self.publisher_ttc = self.create_publisher(Float32, "~/ttc", 10)
        self.publisher_ttc_filtered = self.create_publisher(
            Float32, "~/ttc_filtered", 10
        )

    def parameters_callback(self, parameters: List[rclpy.Parameter]):
        result = SetParametersResult()
        result.successful = True
        for parameter in parameters:
            if parameter.name == "ttc_filter_length":
                if isinstance(parameter.value, int) and parameter.value > 0:
                    self.ttc_filter_length = parameter.value
                    if self.ttc_deque.maxlen != self.ttc_filter_length:
                        self.ttc_deque = deque(
                            self.ttc_deque, maxlen=self.ttc_filter_length
                        )
                else:
                    result.successful = False
                    result.reason = "ttc_filter_length must be a positive integer"
            elif parameter.name == "ttc_threshold":
                if isinstance(parameter.value, (float, int)) and parameter.value > 0:
                    self.ttc_threshold = float(parameter.value)
                else:
                    result.successful = False
                    result.reason = "ttc_threshold must be a positive number"
        return result

    def odom_callback(self, odom_msg: Odometry):
        self.speed = odom_msg.twist.twist.linear.x

    def distance_to_bounding_box(
        self, angle: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """
        Calculates the distance to the bounding box of the car as a function of angle.
        Currently I assume LIDAR is in the back.
        """
        d_side = self.car_width / 2 / np.abs(np.sin(angle))
        d_front_rear = np.where(np.abs(angle) < np.pi/2, self.car_front_distance, self.car_rear_distance)
        d_front_rear /= np.cos(angle)
        d = np.minimum(d_side, d_front_rear)
        # d[np.abs(angle) > np.pi / 2] = 0
        return d

    def scan_callback(self, scan_msg: LaserScan):
        if self.speed is None:
            return

        angles = np.linspace(
            scan_msg.angle_min, scan_msg.angle_max, len(scan_msg.ranges)
        )
        velocity_along_scan = self.speed * np.cos(angles)

        # sanitize ranges array
        ranges = np.asarray(scan_msg.ranges).astype(np.float32)
        ranges[np.isnan(ranges) | np.isinf(ranges)] = scan_msg.range_max

        # calculate TTC
        d = self.distance_to_bounding_box(angles)
        ttc = (ranges - d) / np.maximum(velocity_along_scan, 1e-8)
        ttc = np.clip(ttc, 0, 5 * self.ttc_threshold)
        ttc = np.min(ttc)

        self.ttc_deque.append(ttc)
        ttc_filtered = np.mean(list(self.ttc_deque))

        # publish TTC
        self.publisher_ttc.publish(Float32(data=ttc))
        self.publisher_ttc_filtered.publish(Float32(data=ttc_filtered))

        if ttc_filtered < self.ttc_threshold:
            self.publisher_drive.publish(
                AckermannDriveStamped(drive=AckermannDrive(speed=0.0))
            )


def main(args=None):
    rclpy.init(args=args)
    safety_node = SafetyNode()
    rclpy.spin(safety_node)

    # Destroy the node explicitly
    # (optional - otherwise it will bne automatically
    # when the garbage collector destroys the node object)
    safety_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
