#!/usr/bin/env python3
import math
import time

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from graph_ltpl.Graph_LTPL import Graph_LTPL
from nav_msgs.msg import Odometry
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray
from tigerstack.mpc.visualize import points_to_arrow_markers
from visualization_msgs.msg import Marker, MarkerArray
from tigerstack.mpc import visualize
from tf2_msgs.msg import TFMessage
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
# track_param = configparser.ConfigParser()
# if not track_param.read(toppath + "/params/driving_task.ini"):
#     raise ValueError('Specified online parameter config file does not exist or is empty!')
def waypoint2Marker(obstacles,timestamp,id=0.0,ns="ref",r=0.0,g=0.0,b=1.0):

    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = timestamp
    marker.ns = ns
    marker.id = id
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.pose.position.x = obstacles[0]#.astype(float)
    marker.pose.position.y = obstacles[1]#.astype(float)
    marker.pose.position.z =0.0
    # q = Quaternion()
    # q= yaw2quaternion(wypt[2].astype(float))
    # marker.pose.orientation.x = q.x#.astype(float)#wypt[2].astype(float)
    # marker.pose.orientation.y =q.y#.astype(float)#wypt[2].astype(float)
    # marker.pose.orientation.z = q.z#.astype(float)#wypt[2].astype(float)
    # marker.pose.orientation.w = q.w#.astype(float) #0.0
    marker.scale.x = 0.2
    marker.scale.y = 0.2
    marker.scale.z = 0.1
    marker.color.a = 0.5 #// Don't forget to set the alpha!
    marker.color.r = r
    marker.color.g = g
    marker.color.b = b
    # msg.markers.append(marker)
    # self.publisher_marker.publish(marker)     
    return marker
class Quaternion():
    def __init__(self):
        x=0
        y=0
        z=0
        w=0
def quaternion2yaw(q):
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    yaw = np.arctan2(siny_cosp, cosy_cosp).item()
    return yaw

toppath = get_package_share_directory("planner")

# define all relevant paths
path_dict = {
    "globtraj_input_path": toppath + "/config/traj_ltpl_cl_levine.csv",
    "graph_store_path": "/sim_ws/src/f1ten-tigers/planner/output/stored_graph.pckl",
    # "graph_store_path": toppath + "/stored_graph.pckl",
    "ltpl_offline_param_path": toppath + "/config/ltpl_config_offline.ini",
    "ltpl_online_param_path": toppath + "/config/ltpl_config_online.ini",
}

RUNNING_TOTAL = 0#5000

class Planner(Node):
    """
    Implement Pure Pursuit on the car
    This is just a template, you are free to implement your own node!
    """

    def __init__(self):
        super().__init__("planner_node")  # type: ignore

        self.sim = bool(self.declare_parameter("sim", True).value)

        # self.timer = self.create_timer(0.05, self.timer_callback)
        self.publisher_array = self.create_publisher(Float32MultiArray, "/path", 1)
        self.pub_obs = self.create_publisher(MarkerArray, "~/visualize_obs", 1)
        odom_topic = "/ego_racecar/odom" if self.sim else "/pf/pose/odom"
        self.sub_odom = self.create_subscription(
            Odometry, odom_topic, self.odom_callback, 1
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

        self.filter_width = 3#20
        self.max_range = 4.0
        self.obstacles = []
        self.execute_count = 0
        self.total_runtime = 0.0

        self.transform_position=None
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "drive", 5)
        # self.scan_sub = self.create_subscription(TFMessage,'tf',self.pose_callback_sim,5)
        self.prev_vel=0.0
        self.flag_brake =True

        self.convolve_pub = self.create_publisher(LaserScan,"convolve",5)

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

    def target_transform(self,location,transform,rotation):
        wypt_x = location[0].astype(np.float64)
        wypt_y = location[1].astype(np.float64)
        base_x = transform[0]
        base_y = transform[1]

        q_base= Quaternion()
        q_base.x = rotation.x
        q_base.y = rotation.y
        q_base.z = rotation.z
        q_base.w = rotation.w
        # distance = np.sqrt(np.square(wypt_x-base_x)+np.square(wypt_y-base_y))
        yaw = quaternion2yaw(q_base)
        # self.get_logger().info("wypt pos:%.2f,%.2f,%.3f"%(wypt_x,wypt_y,wypt_thta/np.pi*180))
        x_old = wypt_x-base_x
        y_old = wypt_y-base_y
        rotate_angle = -yaw
        x_new = x_old*np.cos(rotate_angle)-y_old*np.sin(rotate_angle)
        y_new = x_old*np.sin(rotate_angle)+y_old*np.cos(rotate_angle)
        return [x_new,y_new]



    def scan_callback(self, data: LaserScan):

        if self.position is None:
            return
        

        # Find angles between self.angle_min and self.angle_max
        angles = np.arange(data.angle_min, data.angle_max, data.angle_increment)
        ranges = np.asarray(data.ranges)
        ranges = np.nan_to_num(ranges)
        # safety node
        if self.flag_brake and ((np.abs(angles) < np.pi/24) & (ranges < 1.0)).any():
            drive_msg = AckermannDriveStamped()
            drive_msg.drive.speed =0.0
            self.drive_pub.publish(drive_msg)
            print("too close to obstacles")
            # print("too close to obstacles")
            # print("too close to obstacles")
            return

        # obstacle method 1
        sensing_range = 0.5
        object_width = 80    # how many number of lidar points
        pp_ranges = np.copy(ranges)
        pp_cliff_index = (np.abs(np.diff(pp_ranges))>sensing_range).nonzero()[0]  # index
        # print(pp_cliff_index)
        # msg = data
        # msg.ranges = list(np.zeros(len(ranges)))
        if len(pp_cliff_index)==0:
            return
        prev_i=pp_cliff_index[0]
        prev_i = pp_cliff_index[0]
        # convert potential objects in lidar to map frame:
        cur_pos = np.array([self.position[0], self.position[1]])
        angle_offset = self.heading + np.pi / 2
        self.obstacles = []
        for index in pp_cliff_index:
            if index==pp_cliff_index[0]:
                continue
            mid =int((prev_i+index)/2)
            if np.abs(angles[mid])>np.pi/4:
                continue
            if index-prev_i<object_width and ranges[mid]<self.max_range:
                # msg.ranges[mid] = ranges[mid]

                self.obstacles += [
                {
                    "id": 0,  # integer id of the object
                    "type": "physical",  # type 'physical' (only class implemented so far)
                    "X": cur_pos[0]+ranges[mid] * np.cos(angles[mid] + angle_offset),  # x coordinate
                    "Y": cur_pos[1]+ranges[mid] * np.sin(angles[mid] + angle_offset),  # y coordinate
                    "theta": self.heading,  # orientation (north = 0.0)
                    "v":self.velocity,  # velocity along theta
                    "length": 0.3,  # length of the object
                    "width": 0.3,  # width of the object
                }
            ]
            prev_i = index

        # self.convolve_pub.publish(msg)
        self.timer_callback()
        return 0


        ranges = np.clip(ranges, 0, data.range_max)

        ranges = np.convolve(
            ranges, np.ones(self.filter_width) / self.filter_width, "same"
        )



        # mask = (np.abs(angles) < np.pi/3) & (ranges < self.max_range)
        # angles = angles[mask]
        # ranges = ranges[mask]
        t = np.array([self.position[0], self.position[1]])
        angle_offset = self.heading + np.pi / 2
        laser_positions = (
            t
            + np.array(
                [
                    ranges * np.cos(angles + angle_offset),
                    ranges * np.sin(angles + angle_offset),
                ]
            ).T
        )

        # self.obstacles = []
        # for i in range(0, len(laser_positions), self.filter_width):
        #     self.obstacles += [
        #         {
        #             "id": 0,  # integer id of the object
        #             "type": "physical",  # type 'physical' (only class implemented so far)
        #             "X": laser_positions[i, 0],  # x coordinate
        #             "Y": laser_positions[i, 1],  # y coordinate
        #             "theta": 0.0,  # orientation (north = 0.0)
        #             "v": 0.0,  # velocity along theta
        #             "length": 0.3,  # length of the object
        #             "width": 0.3,  # width of the object
        #         }
        #     ]

        occupied = ranges < self.max_range
        # pad ranges to find edges at start and end
        ranges = np.pad(occupied, 1, mode="constant", constant_values=False)

        diff = np.diff(1 * ranges)
        rising_edge = (diff > 0).nonzero()[0]
        falling_edge = (diff < 0).nonzero()[0]
        self.obstacles = []

        for i in range(len(rising_edge)):
            start_idx = rising_edge[i]
            end_idx = min(falling_edge[i], len(ranges) - 1)

            center = np.mean(laser_positions[start_idx:end_idx], axis=0)
            [_x,_]=self.target_transform(center,t,self.transform_position)
            if _x<0.0:
                continue

            self.obstacles += [
                {
                    "id": 0,  # integer id of the object
                    "type": "physical",  # type 'physical' (only class implemented so far)
                    "X": center[0],  # x coordinate
                    "Y": center[1],  # y coordinate
                    "theta": self.heading,  # orientation (north = 0.0)
                    "v":self.velocity-1,  # velocity along theta
                    "length": 0.3,  # length of the object
                    "width": 0.3,  # width of the object
                }
            ]
        if len(self.obstacles)==3:
            del(self.obstacles[2])
            del(self.obstacles[0])
        else:
            self.obstacles=[]
        self.timer_callback()


    # def pose_callback_sim(self, pose_msg):
    #     # self.visualizeWaypoints()

    #     # TODO: find the current waypoint to track using methods mentioned in lecture
    #     waypoint = None
    #     base_link_tf_info=None
    #     for tf in pose_msg.transforms:
    #         if tf.header.frame_id=="map" and "opp_racecar" in tf.child_frame_id:
    #             base_link_tf_info=tf.transform
    #             self.obstacles = [
    #             {
    #                 "id": 0,  # integer id of the object
    #                 "type": "physical",  # type 'physical' (only class implemented so far)
    #                 "X": base_link_tf_info.translation.x,  # x coordinate
    #                 "Y": base_link_tf_info.translation.y,  # y coordinate
    #                 "theta": 0.0,  # orientation (north = 0.0)
    #                 "v": 0.0,  # velocity along theta
    #                 "length": 0.3,  # length of the object
    #                 "width": 0.3,  # width of the object
    #             }
    #         ]



    def select_action(self, trajectory_set):
        for selected_action in [
            "right",
            "straight",
            "left",
            "follow",
        ]:
            if selected_action in trajectory_set.keys():
                return selected_action
        raise RuntimeError("\n\nNo action found.\n")

    def get_objects(self):
        # return []
        obs_list = []
        for obs in self.obstacles:
            obs_list.append([obs["X"],obs["Y"]])
        marker_array_msg = MarkerArray()
        marker = Marker()
        marker.id = 0
        marker.ns ="planner"
        marker.action = Marker.DELETEALL
        marker_array_msg.markers.append(marker)
        self.pub_obs.publish(marker_array_msg)
        msg = MarkerArray()
        id = 0
        for obs in  obs_list:
            # self.get_logger().info("visualizeWaypoints")
            timestamp= self.get_clock().now().to_msg()
            marker = waypoint2Marker(obs,timestamp,id,"planner",0.8,0.0,0.8)
            id+=1
            msg.markers.append(marker)
        self.pub_obs.publish(msg)  
        return self.obstacles

    def timer_callback(self):

        t_start = time.time()
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
        objects_num = len(self.get_objects())
        self.obstacles=[]

        # -- CALCULATE VELOCITY PROFILE AND RETRIEVE TRAJECTORIES ----------------------------------------------------------
        # pos_est:[x, y]
        # vel_est:float

        # if np.abs(self.prev_vel-self.velocity<0.2) and self.prev_vel<0.2:
        #     self.velocity = 1.0
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
        self.prev_vel = self.velocity

        self.selected_action = self.select_action(traj_set)

        # [s, x, y, heading, curvature, vx, ax]
        trajectory = traj_set[self.selected_action][0]
        # print("speed={},\n".format(trajectory[:,5]))
        print("length of traj_set=",len(traj_set))
        # trajectory[:,5]+=2
        array_msg = Float32MultiArray()
        array_msg.data = list(trajectory.flatten())
        self.publisher_array.publish(array_msg)

        # -- LIVE PLOT (if activated) --------------------------------------------------------------------------------------
        self.ltpl_obj.visual()
        t_end =  time.time()
        self.total_runtime+=(t_end-t_start)
        if RUNNING_TOTAL!=0:
            if self.execute_count<RUNNING_TOTAL:
                self.execute_count+=1
            else:
                print("average running frequency = ",self.execute_count/(self.total_runtime))
                time.sleep(100)
        print("frequency={:.2f},\tnumber of objects:{}, speed[0]={:.2f}".format(1/(t_end-t_start),objects_num,trajectory[0,5] ))


def main(args=None):
    rclpy.init(args=args)
    planner_node = Planner()
    print("Planner Initialized")
    rclpy.spin(planner_node)
    planner_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
