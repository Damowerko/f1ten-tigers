"""
This file contains the class definition for tree nodes and RRT
Before you start, please read: https://arxiv.org/pdf/1105.1186.pdf
"""
import array
import math
import os
import random
import time
from typing import Dict, List, Set, Tuple

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from geometry_msgs.msg import (
    Point,
    PointStamped,
    Pose,
    PoseArray,
    PoseStamped,
    Quaternion,
)
from std_msgs.msg import Bool
from lab6_pkg.occupancy_grid import LaserOccupancy
from nav_msgs.msg import OccupancyGrid, Odometry
from numpy import linalg as LA
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import LaserScan
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker, MarkerArray

# TODO: import as you need


# class def for tree nodes
# It's up to you if you want to use this
class TreeNode(object):
    def __init__(self, pos: Tuple[float, float], is_root=False, parent=None, cost=None):
        self.pos = np.array(pos)
        self.parent = parent
        self.cost = cost  # only used in RRT*
        self.is_root = is_root
        self.leaves = []

    def add_leaf(self, leaf):
        self.leaves.append(leaf)

    def get_leaves(self):
        leaves = []
        for x in self.leaves:
            leaves.append(x)
            child = x.get_leaves()
            if child:
                leaves.append(child)
        if leaves:
            return leaves


# class def for RRT
class RRT(Node):
    def __init__(self):
        super().__init__("rrt_node")
        # topics, not saved as attributes
        # TODO: grab topics from param file, you'll need to change the yaml file
        odom_pos_topic = "/ego_racecar/odom"
        pos_topic = "/pf/viz/inferred_pose"
        scan_topic = "/scan"
        drive_topic = "/drive"
        occp_grid_topic = "/rrt/occupancy"

        # you could add your own parameters to the rrt_params.yaml file,
        # and get them here as class attributes as shown above.

        wypts_file = get_package_share_directory("tigerstack") + self.declare_parameter("waypoints_file", "/maps/Spielberg.csv").value
        self.waypoints3d = np.loadtxt(wypts_file, delimiter=";", dtype=float)
        self.waypoints3d = self.waypoints3d[:, [1,2,3]]
        #self.waypoints3d = self.waypoints3d[::4]
        self.waypoints2d = self.waypoints3d[:, [0,1]]

        self.publish_frame = "ego_racecar/laser"

        is_simulation = self.declare_parameter("sim", True).value  # type: ignore

        if is_simulation:
            self.pose_sub_ = self.create_subscription(
                Odometry, odom_pos_topic, self.sim_pos_callback, 1
            )
        else:
            self.pose_sub_ = self.create_subscription(
                PoseStamped, pos_topic, self.pose_callback, 1
            )
            self.publish_frame = "laser"
            
        

        self.scan_sub_ = self.create_subscription(
            LaserScan, scan_topic, self.scan_callback, 1
        )

        # publishers
        self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 1)
        self.publisher_marker = self.create_publisher(MarkerArray, "rrt/waypoints", 10)
        self.publisher_goal_marker = self.create_publisher(Marker, "rrt/goalMarker", 10)
        self.ocp_grid_pub = self.create_publisher(OccupancyGrid, occp_grid_topic, 1)
        self.tree_pub = self.create_publisher(MarkerArray, "rrt/tree", 1)
        self.tree_points_pub = self.create_publisher(Marker, "rrt/tree_points", 1)
        self.path_pub = self.create_publisher(MarkerArray, "rrt/path", 1)
        self.path_pure_pursuit_pub = self.create_publisher(
            PoseArray, "pure_pursuit/waypoints", 1
        )
        self.collision_pub = self.create_publisher(Bool, "rrt/collision", 1
        )

        # rrt params
        self.goal_radius = 0.5
        self.lookahead_radius = 1.5
        self.max_step = 0.4
        self.neighbor_radius = 0.75
        self.search_limit = 1000

        # occp map params
        self.occp_gridsize = 6
        self.occp_resolution = 0.15
        self.dilation = 0.2
        self.occupancy = LaserOccupancy(self.occp_gridsize, self.occp_resolution)

        # transformations
        # self.laser_offset
        self.laser_tf_info = None
        self.car_pos = None

        # control flags
        self.waypoints_Mapped = False

        # pure persuit
        self.kp = 0.1

        self.visualizeWaypoints()

    def makeWaypointMarker(
        self,
        wypt,
        timestamp,
        id=0.0,
        ns="ref",
        r=0.0,
        g=0.0,
        b=1.0,
        type=Marker.ARROW,
        x_scale=0.5,
    ):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = timestamp
        marker.ns = ns
        marker.id = id
        marker.type = type
        marker.action = Marker.ADD
        marker.pose.position.x = wypt[0]
        marker.pose.position.y = wypt[1]
        marker.pose.position.z = 0.0
        rot = Rotation.from_euler("xyz", [0, 0, wypt[2]])
        quant = rot.as_quat()
        marker.pose.orientation.x = quant[0]
        marker.pose.orientation.y = quant[1]
        marker.pose.orientation.z = quant[2]
        marker.pose.orientation.w = quant[3]
        marker.scale.x = x_scale
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        return marker

    def curr_marker(self, goal_wypt):
        if self.car_pos:
            wypt = self.transformFrame(self.car_pos, goal_wypt, tf_mode=False)
            # TODO laser transform
            # wypt = self.transformFrame(self.laser_tf_info, wypt)
            marker = self.makeWaypointMarker(
                wypt,
                self.get_clock().now().to_msg(),
                100,
                r=1.0,
                type=Marker.SPHERE,
                x_scale=0.1,
            )
            marker.header.frame_id = self.publish_frame

            self.publisher_goal_marker.publish(marker)
            return wypt

    def visualizeWaypoints(self):
        msg = MarkerArray()
        id = 0
        for wypt in self.waypoints3d:
            timestamp = self.get_clock().now().to_msg()
            marker = self.makeWaypointMarker(wypt, timestamp, id)
            id += 1
            msg.markers.append(marker)
        self.publisher_marker.publish(msg)

    def selectNextGoal(self):
        car = np.array([self.car_pos.position.x, self.car_pos.position.y])
        waypoints = self.waypoints2d
        distances = np.linalg.norm(waypoints - car, axis=1)
        closest_idx = np.argmin(distances)
        goal_idx = closest_idx + np.argmax(
            np.roll(distances, -closest_idx) > self.lookahead_radius
        )
        goal_idx = goal_idx % (len(waypoints)-1)
        return self.waypoints3d[goal_idx]

    def pose_callback_sim(self, pose_msg: TFMessage):
        waypoint = None
        for tf in pose_msg.transforms:
            if (
                tf.header.frame_id == "ego_racecar/base_link"
                and tf.child_frame_id == "ego_racecar/laser"
            ):
                self.laser_tf_info = tf.transform
        if waypoint is None:
            return

    def transformFrame(self, target_frame, waypoint, tf_mode=True, reverse=True):
        translation, rotation_mat = None, None

        if tf_mode:
            translation = np.array(
                [
                    target_frame.translation.x,
                    target_frame.translation.y,
                    target_frame.translation.z,
                ]
            )
        else:
            translation = np.array(
                [
                    target_frame.position.x,
                    target_frame.position.y,
                    target_frame.position.z,
                ]
            )
        if tf_mode:
            rotation_mat = Rotation.from_quat(
                np.array(
                    [
                        target_frame.rotation.x,
                        target_frame.rotation.y,
                        target_frame.rotation.z,
                        target_frame.rotation.w,
                    ]
                )
            ).as_matrix()
        else:
            rotation_mat = Rotation.from_quat(
                np.array(
                    [
                        target_frame.orientation.x,
                        target_frame.orientation.y,
                        target_frame.orientation.z,
                        target_frame.orientation.w,
                    ]
                )
            ).as_matrix()
        if reverse:
            return (np.linalg.inv(rotation_mat) @ (waypoint - translation)).flatten()
        else:
            return (rotation_mat @ waypoint + translation).flatten()
        
    def publish_collision(self, collision : bool):
        msg = Bool()
        msg.data = collision
        self.collision_pub.publish(msg)

    def publish_occupancy(self):
        msg = OccupancyGrid()
        msg.info.resolution = self.occp_resolution
        msg.info.width = int(self.occp_gridsize / self.occp_resolution)
        msg.info.height = msg.info.width

        msg.data = array.array(
            "b", (np.array(self.occupancy.grid, dtype=int) * 100).flatten().tolist()
        )

        origin = Pose()
        origin._position.x = -self.occp_gridsize / 2
        origin._position.y = -self.occp_gridsize / 2
        msg.info.origin = origin

        msg.header.frame_id = self.publish_frame
        self.ocp_grid_pub.publish(msg)

    def scan_callback(self, scan_msg: LaserScan):
        """
        LaserScan callback, you should update your occupancy grid here

        Args:
            scan_msg (LaserScan): incoming message from subscribed topic
        Returns:

        """
        angles = (
            -np.arange(scan_msg.angle_min, scan_msg.angle_max, scan_msg.angle_increment)
            + math.pi / 2
        )
        ranges = np.array(scan_msg.ranges)


        self.occupancy.from_scan(angles, ranges)
        self.occupancy.dilate(self.dilation)
        self.publish_occupancy()

    def sim_pos_callback(self, pose_msg: Odometry):
        self.pose_callback(pose_msg.pose)

    def pose_callback(self, pose_msg: Odometry):
        """
        The pose callback when subscribed to particle filter's inferred pose
        Here is where the main RRT loop happens

        Args:
            pose_msg (PoseStamped): incoming message from subscribed topic
        Returns:

        """
        self.start = pose_msg.pose
        self.car_pos = pose_msg.pose

        num_failed_searches = 0
        search_count = 0
        PathFound = False
        self.path = None

        if not self.waypoints_Mapped:
            self.visualizeWaypoints()
            self.waypoints_Mapped = True

        
        if self.occupancy.filled:
            goal_wypt = self.selectNextGoal()
            goal_wypt = self.curr_marker(goal_wypt)
            

            if self.collision_free():
                self.publish_collision(False)
                return None
            
            self.publish_collision(True)
            self.gen_samples()
            start = time.time()
            goal_point = np.array([goal_wypt[0], goal_wypt[1]])
            tree = np.array([TreeNode((0, 0), is_root=True, cost=0)])
            poses = np.array(np.array([0, 0]))
            while not PathFound and num_failed_searches < 100 and search_count < self.search_limit:
                search_count += 1
                point = self.get_sample()
                near = self.nearest(poses, point)

                if self.check_collision(tree[near].pos, point):
                    num_failed_searches = 0
                    steer_pos, cost = self.steer(tree[near].pos, point)
                    neighbors, distances = self.calculate_neightbors(
                        steer_pos, poses, tree
                    )

                    new = TreeNode(steer_pos, cost=cost + tree[near].cost)
                    new.parent = tree[near]
                    if len(neighbors):
                        self.find_cheapest(tree, neighbors, distances, new)
                        self.rewire_neighbors(tree, neighbors, distances, new)
                    tree = np.append(tree, new)

                    poses = np.vstack((poses, new.pos))
                    if self.is_goal(steer_pos, goal_point):
                        self.path = self.find_path(new)
                        PathFound = True
                else:
                    num_failed_searches += 1

            if self.path:
                self.draw_tree(tree)
                self.draw_tree_points(tree)
                self.draw_path(self.path)
                # self.pure_persuit(self.path)
                self.publish_pure_pursuit(self.path)
                self.get_logger().info(
                    "runtime = {}".format(time.time() - start),
                    throttle_duration_sec=1.0,
                )
            # else:
            #     print("rrt failed to find solution")
            return None
        
    def collision_free(self):
        # car = np.array([self.car_pos.position.x, self.car_pos.position.y])
        # distances = np.linalg.norm(self.waypoints2d - car, axis=1)
        # closest_idx = np.argmin(distances)
        # if (distances[closest_idx] <1):
        #     wypts = np.arange(closest_idx-2, closest_idx+5) % len(self.waypoints2d)
        #     free = True
        #     for point in np.nditer(wypts):
        #         wypt = self.transformFrame(self.car_pos, self.waypoints3d[point], tf_mode=False)
        #         free= free and self.check_collision((0,0), (wypt[0], wypt[1]))
        #     return free
        return False

    def calculate_neightbors(self, steer_pos, poses, tree: List[TreeNode]):
        delt = poses - steer_pos
        if delt.ndim == 1:
            delt = np.array([delt])
        distances = np.linalg.norm(delt, axis=1)
        neightbors_idx = np.argwhere(distances < self.neighbor_radius).flatten()
        free_neightbors = []
        neightbor_dists = []

        for i in range(len(neightbors_idx)):
            if self.check_collision(tree[neightbors_idx[i]].pos, steer_pos):
                free_neightbors.append(neightbors_idx[i])
                neightbor_dists.append(distances[neightbors_idx[i]])

        return np.array(free_neightbors), np.array(neightbor_dists)

    def find_cheapest(
        self, tree: List[TreeNode], neighbors, distances, new_node: TreeNode
    ):
        neighbor_nodes = tree[neighbors]
        for i in range(len(neighbor_nodes)):
            if neighbor_nodes[i].cost + distances[i] < new_node.cost:
                new_node.cost = neighbor_nodes[i].cost + distances[i]
                new_node.parent = neighbor_nodes[i]

    def rewire_neighbors(
        self, tree: List[TreeNode], neighbors, distances, new_node: TreeNode
    ):
        neighbor_nodes = tree[neighbors]
        for i in range(len(neighbor_nodes)):
            if new_node.cost + distances[i] < neighbor_nodes[i].cost:
                neighbor_nodes[i].parent = new_node
                neighbor_nodes[i].cost = new_node.cost + distances[i]

    def pure_persuit(self, path: List[TreeNode]):
        target = path[1].pos

        distance_sq = np.square(target[0]) + np.square(target[1])
        curvature = 2 * abs(target[1]) / distance_sq

        # TODO: calculate curvature/steering angle

        # TODO: publish drive message, don't forget to limit the steering angle.
        drive_msg = AckermannDriveStamped()
        steering_angle = min(curvature * abs(target[1]) / target[1] * self.kp, 0.5)
        drive_msg.drive.steering_angle = steering_angle
        # self.get_logger().info("steering angle=%.3f"%steering_angle)
        # self.get_logger().info("\n")
        drive_msg.drive.speed = 1.0
        self.drive_pub.publish(drive_msg)

    def publish_pure_pursuit(self, path: List[TreeNode]):
        msg = PoseArray()
        for ele in path:
            vec = np.array((ele.pos[0], ele.pos[1], 0))
            world_pos = self.transformFrame(
                self.car_pos, vec, tf_mode=False, reverse=False
            )
            pos = Pose()
            pos.position.x = world_pos[0]
            pos.position.y = world_pos[1]
            msg.poses.append(pos)

        msg.header.frame_id = "/map"
        self.path_pure_pursuit_pub.publish(msg)

    def make_line_marker(
        self, n1: TreeNode, n2: TreeNode, size=0.01, a=1.0, r=0.0, g=1.0, b=0.0
    ):
        marker = Marker()
        marker.header.frame_id = self.publish_frame
        marker.ns = "ref"
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        p = Point()
        p.x = float(n1.pos[0])
        p.y = float(n1.pos[1])
        p.z = 0.0
        marker.points.append(p)
        p2 = Point()
        p2.x = float(n2.pos[0])
        p2.y = float(n2.pos[1])
        p2.z = 0.0
        marker.points.append(p2)
        marker.scale.x = size
        marker.scale.y = size
        marker.scale.z = size
        marker.color.a = a
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        return marker

    def make_tree_point_marker(self):
        marker = Marker()
        marker.header.frame_id = self.publish_frame
        marker.ns = "ref"
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        return marker

    def delete_Marker_Array(self, publisher):
        marker_array_msg = MarkerArray()
        marker = Marker()
        marker.id = 0
        marker.ns = "ref"
        marker.action = Marker.DELETEALL
        marker_array_msg.markers.append(marker)
        publisher.publish(marker_array_msg)

    def delete_tree_points(self):
        marker = Marker()
        marker.id = 0
        marker.ns = "ref"
        marker.action = Marker.DELETEALL
        self.tree_points_pub.publish(marker)

    def draw_tree(self, tree: List[TreeNode]):
        self.delete_Marker_Array(self.tree_pub)
        msg = MarkerArray()
        id = 0
        for ele in tree:
            if not ele.is_root:
                posMarker = self.make_line_marker(ele, ele.parent, a=0.75)
                posMarker.id = id
                id += 1
                msg.markers.append(posMarker)

        self.tree_pub.publish(msg)

    def draw_path(self, path: List[TreeNode]):
        self.delete_Marker_Array(self.path_pub)
        prev = None

        msg = MarkerArray()
        id = 0

        for ele in path:
            if prev == None:
                prev = ele
            else:
                posMarker = self.make_line_marker(prev, ele, size=0.05, g=0.0, r=1.0)
                posMarker.id = id
                id += 1
                msg.markers.append(posMarker)
                prev = ele

        self.path_pub.publish(msg)

    def draw_tree_points(self, tree: List[TreeNode]):
        self.delete_tree_points()
        marker = self.make_tree_point_marker()
        for ele in tree:
            p = Point()
            p.x = float(ele.pos[0])
            p.y = float(ele.pos[1])
            p.z = 0.0
            marker.points.append(p)
        self.tree_points_pub.publish(marker)

    def gen_samples(self):
        """
        This method should randomly sample the free space, and returns a viable point

        Args:
        Returns:
            (x, y) (float float): a tuple representing the sampled point

        """
        self.samples = np.random.random_sample((2, self.search_limit))
        self.sample_num = 0
        
    
    def get_sample(self):
        range = self.occp_gridsize -self.occp_resolution
        x = (self.samples[0][self.sample_num]) * range/2
        y = (self.samples[1][self.sample_num] - 0.5) * range
        self.sample_num+=1
        return (x, y)


    def nearest(self, poses, sampled_point):
        """
        This method should return the nearest node on the tree to the sampled point

        Args:
            tree ([]): the current RRT tree
            sampled_point (tuple of (float, float)): point sampled in free space
        Returns:
            nearest_node (int): index of neareset node on the tree
        """
        delt = poses - np.array(sampled_point)
        if delt.ndim == 1:
            delt = np.array([delt])
        distances = np.linalg.norm(delt, axis=1)
        return np.argmin(distances)

    def steer(self, nearest_node, sampled_point):
        """
        This method should return a point in the viable set such that it is closer
        to the nearest_node than sampled_point is.

        Args:
            nearest_node (tuple of (float, float)): nearest node on the tree to the sampled point
            sampled_point (tuple of (float, float)): sampled point
        Returns:
            new_node (tuple of (float, float)): new node created from steering
        """
        vec = sampled_point - nearest_node
        dist = np.linalg.norm(vec)
        if (dist) > self.max_step:
            unit = vec / np.linalg.norm(vec)
            return nearest_node + unit * self.max_step, self.max_step
        else:
            return sampled_point, dist

    def check_collision(self, nearest_node, new_node):
        """
        This method should return whether the path between nearest and new_node is
        collision free.

        Args:
            nearest (float,float): nearest node on the tree
            new_node (float,float): new node from steering
        Returns:
            collision (bool): whether the path between the two nodes are in collision
                              with the occupancy grid
        """

        return self.occupancy.is_line_free(np.array(nearest_node), np.array(new_node))

    def is_goal(self, node_pos, goal_pos):
        """
        This method should return whether the latest added node is close enough
        to the goal.

        Args:
            latest_added_node (Node): latest added node on the tree

        Returns:
            close_enough (bool): true if node is close enoughg to the goal
        """
        return (np.linalg.norm(node_pos - goal_pos)) < self.goal_radius

    def find_path(self, latest_added_node: TreeNode):
        """
        This method returns a path as a list of Nodes connecting the starting point to
        the goal once the latest added node is close enough to the goal

        Args:
            latest_added_node (Node): latest added node in the tree
        Returns:
            path ([]): valid path as a list of Nodes
        """
        path = []
        curr = latest_added_node
        while curr.parent:
            path.append(curr)
            curr = curr.parent
        path.append(curr)
        return list(reversed(path))


def main(args=None):
    rclpy.init(args=args)
    print("RRT Initialized")
    rrt_node = RRT()
    rclpy.spin(rrt_node)

    rrt_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
