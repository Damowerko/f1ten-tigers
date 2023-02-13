from copy import deepcopy
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class PID:
    def __init__(self, Kp=0.0, Ki=0.0, Kd=0.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.previous_error = 0.0
        self.integral = 0.0

    def update(self, error, dt) -> float:
        proportional = error
        self.integral = (error + self.previous_error) / 2 * dt
        derivative = (error - self.previous_error) / dt
        return self.Kp * proportional + self.Ki * self.integral + self.Kd * derivative


class ReactiveFollowGap(Node):
    """
    Implement Wall Following on the car
    This is just a template, you are free to implement your own node!
    """

    def __init__(self):
        super().__init__("reactive_node")  # type: ignore
        # Topics & Subs, Pubs
        lidarscan_topic = "/scan"
        drive_topic = "/drive"

        self.max_range = 3.0
        self.filter_width = 5
        self.bubble_radius = 0.3
        self.disparity_threshold = 0.1
        self.turn_threshold = 0.2

        self.stop_distance = 0.2
        self.speed_min = 1.5
        self.speed_max = 6.0
        self.speed_min_distance = 1.0
        self.speed_max_distance = 8.0
        self.speed_angle = np.pi / 30

        self.pid = PID(0.5, 1e-5, 1e-5)
        self.last_time = self.get_clock().now().nanoseconds / 1e9

        self.create_subscription(LaserScan, lidarscan_topic, self.lidar_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.closest_pub = self.create_publisher(PointStamped, "~/closest_point", 10)
        self.extended_pub = self.create_publisher(LaserScan, "~/extended", 10)

    def preprocess_lidar(self, ranges):
        """Preprocess the LiDAR scan array. Expert implementation includes:
        1.Setting each value to the mean over some window
        2.Rejecting high values (eg. > 3m)
        """
        ranges = np.clip(np.nan_to_num(ranges), 0.0, self.max_range)
        ranges = np.convolve(
            ranges, np.ones(self.filter_width) / self.filter_width, "same"
        )
        return ranges

    def find_distance(
        self, angles: npt.NDArray[np.float64], ranges: npt.NDArray[np.float64], idx: int
    ) -> npt.NDArray[np.float64]:
        """
        Find the distance from the point at `idx` to all other points in the scan.
        We do this using the law of cosines: $$c^2 = a^2 + b^2 - 2ab*cos(theta)$$

        Args:
            angles: array of angles
            ranges: array of ranges
            idx: index of point to find distance to
        """
        angle_difference = angles - angles[idx]
        distances = np.sqrt(
            ranges**2
            + ranges[idx] ** 2
            - 2 * ranges * ranges[idx] * np.cos(angle_difference)
        )
        return distances

    def find_widest_range(
        self, free_space_ranges: npt.NDArray[np.bool8]
    ) -> Tuple[int, int]:
        """
        Find the largest contigous sequence of true values in free_space_ranges.

        Args:
            free_space_ranges: array of booleans indicating whether a point is free or not
        Returns:
            start, end: start and end indicies of max-gap range. end_i is exclusive.
        """
        if np.all(~free_space_ranges):
            return 0, 0
        free_space_ranges = np.pad(free_space_ranges.astype(int), 1, constant_values=0)
        diffs = np.diff(free_space_ranges)
        assert len(diffs.shape) == 1
        rising_idx = (diffs > 0).nonzero()[0]
        falling_idx = (diffs < 0).nonzero()[0]
        lengths = falling_idx - rising_idx
        max_gap_idx = np.argmax(lengths)
        start = rising_idx[max_gap_idx]
        end = falling_idx[max_gap_idx]
        return start, end

    def find_best_point(self, ranges) -> Optional[int]:
        """Start_i & end_i are start and end indicies of max-gap range, respectively
        Return index of best point in ranges
            Naive: Choose the furthest point within ranges and go there
        """
        if len(ranges) == 0:
            return None

        max_distance = np.max(ranges)
        close_to_max = np.abs(ranges - max_distance) <= 1e-3
        # Don't want just any point on the edge of the gap, we want the middle
        start, end = self.find_widest_range(close_to_max)
        return int((start + end) / 2)

    def compute_speed(self, distance) -> float:
        if distance < self.stop_distance:
            return 0.0
        elif distance < self.speed_min_distance:
            return self.speed_min
        elif distance > self.speed_max_distance:
            return self.speed_max
        else:
            return self.speed_min + (distance - self.speed_min_distance) / (
                self.speed_max_distance - self.speed_min_distance
            ) * (self.speed_max - self.speed_min)

    def range_angle_to_point(self, angle, range):
        point = PointStamped()
        point.header.frame_id = "ego_racecar/base_link"
        point.point.x = range * np.cos(angle)
        point.point.y = range * np.sin(angle)
        return point

    def extend_disparities(self, ranges, angles):
        """
        Extend disparities to the max range of the LiDAR.
        """
        angle_increment = angles[1] - angles[0]
        diff = np.diff(ranges)
        left_disparities = (diff > self.disparity_threshold).nonzero()[0]
        right_disparities = (diff < -self.disparity_threshold).nonzero()[0] + 1
        for idx in left_disparities:
            extend_angle = self.bubble_radius / max(0.2, ranges[idx])
            extend_len = int(extend_angle / angle_increment)
            target_slice = slice(idx, min(idx + extend_len, len(ranges)))
            ranges[target_slice] = np.minimum(ranges[idx], ranges[target_slice])
        for idx in right_disparities:
            extend_angle = self.bubble_radius / max(0.2, ranges[idx])
            extend_len = int(extend_angle / angle_increment)
            target_slice = slice(max(idx - extend_len, 0), idx)
            ranges[target_slice] = np.minimum(ranges[idx], ranges[target_slice])
            ranges[max(idx - extend_len, 0) : idx] = ranges[idx]
        return ranges, angles

    def lidar_callback(self, data: LaserScan):
        """Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped Message"""

        # Find angles between -self.angle_max and self.angle_max
        angles = np.arange(data.angle_min, data.angle_max, data.angle_increment)
        ranges = np.asarray(data.ranges)

        # run on raw lidar data
        obstacle_distance = np.percentile(ranges[np.abs(angles) < self.speed_angle], 5)

        # Get angles between -90 and 90 degrees
        mask = np.abs(angles) < np.pi / 2

        distance_behind_right = np.percentile(ranges[(~mask) & (angles < 0)], 5)
        distance_behind_left = np.percentile(ranges[(~mask) & (angles > 0)], 5)

        angles = angles[mask]
        ranges = ranges[mask]

        ranges = self.preprocess_lidar(ranges)
        ranges, angles = self.extend_disparities(ranges, angles)
        best_point_idx = self.find_best_point(ranges)

        drive_msg = AckermannDriveStamped()
        if best_point_idx is not None:
            current_time = self.get_clock().now().nanoseconds / 1e9
            dt = current_time - self.last_time
            self.last_time = current_time
            error = angles[best_point_idx]
            steering_angle = self.pid.update(error, dt)
            if distance_behind_left < self.turn_threshold and steering_angle > 0:
                steering_angle = 0.0
            if distance_behind_right < self.turn_threshold and steering_angle < 0:
                steering_angle = 0.0
            drive_msg.drive.steering_angle = steering_angle
            drive_msg.drive.speed = self.compute_speed(obstacle_distance)

        self.drive_pub.publish(drive_msg)

        extended_scan = deepcopy(data)
        extended_scan.ranges = list(ranges)
        extended_scan.angle_min = angles[0]
        extended_scan.angle_max = angles[-1]
        self.extended_pub.publish(extended_scan)


def main(args=None):
    rclpy.init(args=args)
    print("WallFollow Initialized")
    reactive_node = ReactiveFollowGap()
    rclpy.spin(reactive_node)
    reactive_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
