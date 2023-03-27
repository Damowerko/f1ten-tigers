from functools import partial

import rclpy
from sensor_msgs.msg import Imu


def imu_callback(publisher, msg: Imu):
    msg.header.frame_id = "odom"
    publisher.publish(msg)


def main():
    rclpy.init()
    node = rclpy.create_node("odom_frame")  # type: ignore
    imu_pub = node.create_publisher(Imu, "/sensors/imu/raw_framed", 10)
    imu_sub = node.create_subscription(
        Imu, "/sensors/imu/raw", partial(imu_callback, imu_pub), 10
    )
    rclpy.spin(node)
