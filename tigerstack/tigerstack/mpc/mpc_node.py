#!/usr/bin/env python3
import math
import time
from dataclasses import dataclass, field

import cvxpy
import numpy as np
import numpy.typing as npt
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import Odometry
from rclpy.node import Node
from scipy.linalg import block_diag
from scipy.sparse import block_diag, csc_matrix
from scipy.spatial.transform import Rotation
from std_msgs.msg import Float32MultiArray
from tigerstack.mpc import visualize
from tigerstack.mpc.utils import nearest_point
from tigerstack.mpc.waypoints import trajectory_from_waypoints
from visualization_msgs.msg import MarkerArray


@dataclass
class mpc_config:
    NXK: int = 4  # length of kinematic state vector: z = [x, y, v, yaw]
    NU: int = 2  # length of input vector: u = = [steering speed, acceleration]
    TK: int = 10  # finite time horizon length kinematic

    # ---------------------------------------------------
    # TODO: you may need to tune the following matrices
    Rk: npt.NDArray = field(
        default_factory=lambda: np.diag([0.01, 5.0])
    )  # input cost matrix, penalty for inputs - [accel, steering_speed]
    Rdk: npt.NDArray = field(
        default_factory=lambda: np.diag([0.05, 50.0])
    )  # input difference cost matrix, penalty for change of inputs - [accel, steering_speed]
    Qk: npt.NDArray = field(
        default_factory=lambda: np.diag([5.0, 5.0, 10.0, 3.0])
    )  # state error cost matrix, for the the next (T) prediction time steps [x, y, v, yaw]
    Qfk: npt.NDArray = field(
        default_factory=lambda: np.diag([10.0, 20.0, 10.0, 3.0])
    )  # final state error matrix, penalty  for the final state constraints: [x, y, v, yaw]
    # ---------------------------------------------------

    N_IND_SEARCH: int = 20  # Search index number
    DTK: float = 0.05  # time step [s] kinematic
    dlk: float = 0.05  # dist step [m] kinematic
    LENGTH: float = 0.58  # Length of the vehicle [m]
    WIDTH: float = 0.31  # Width of the vehicle [m]
    WB: float = 0.33  # Wheelbase [m]
    MIN_STEER: float = -0.4189  # maximum steering angle [rad]
    MAX_STEER: float = 0.4189  # maximum steering angle [rad]
    MAX_DSTEER: float = np.deg2rad(180.0)  # maximum steering speed [rad/s]
    MAX_SPEED: float = 10.0  # maximum speed [m/s]
    MIN_SPEED: float = 0.0  # minimum backward speed [m/s]
    MAX_ACCEL: float = 10.0  # maximum acceleration [m/ss]


@dataclass
class State:
    x: float = 0.0
    y: float = 0.0
    delta: float = 0.0
    v: float = 0.0
    yaw: float = 0.0
    yawrate: float = 0.0
    beta: float = 0.0


class MPC(Node):
    """
    Implement Kinematic MPC on the car
    This is just a template, you are free to implement your own node!
    """

    def __init__(self):
        super().__init__("mpc_node")  # type: ignore

        # declare parameters
        self.sim = bool(self.declare_parameter("sim", True).value)
        self.speed_factor = float(self.declare_parameter("speed_factor", 1.0).value)  # type: ignore

        # p ublishers and subscribers
        self.pub_drive = self.create_publisher(AckermannDriveStamped, "/drive", 1)
        self.pub_visualize = self.create_publisher(MarkerArray, "~/visualize", 1)

        odom_topic = "/ego_racecar/odom" if self.sim else "/pf/pose/odom"
        self.sub_odom = self.create_subscription(
            Odometry, odom_topic, self.odom_callback, 1
        )

        self.path_sub = self.create_subscription(
            Float32MultiArray, f"/path", self.path_callback, 1
        )
        self.last_path_time = time.time()

        self.config = mpc_config()
        self.odelta_v = None
        self.odelta = None
        self.oa = None
        self.init_flag = 0

        # variables tracked by odometry
        self.velocity = 0.0
        self.yaw_rate = 0.0
        self.steering_angle = 0.0
        self.chassis_slip_angle = 0.0

        # load waypoints assuming constant speed
        waypoints_filename = (
            get_package_share_directory("tigerstack") + "/maps/skir_6ay.csv"
        )
        self.static_waypoints = np.loadtxt(
            waypoints_filename, delimiter=";", dtype=float
        )
        self.use_static_waypoints = True
        self.update_trajectory(self.static_waypoints)

        # initialize MPC problem
        self.mpc_prob_init()

    def path_callback(self, msg: Float32MultiArray):
        self.last_path_time = time.time()
        self.use_static_waypoints = False
        waypoints = np.array(msg.data).reshape(-1, 7).astype(np.float64)
        self.update_trajectory(waypoints)

    def update_trajectory(self, waypoints):
        self.trajectory = trajectory_from_waypoints(waypoints)
        self.lap_length = waypoints[-1, 0]

    def odom_callback(self, odom_msg: Odometry):
        if not self.use_static_waypoints and time.time() - self.last_path_time > 1.0:
            self.get_logger().warn("Fallback to static waypoints!")
            self.use_static_waypoints = True
            self.update_trajectory(self.static_waypoints)

        position = odom_msg.pose.pose.position
        orientation = odom_msg.pose.pose.orientation
        velocity = odom_msg.twist.twist.linear

        self.velocity = velocity.x
        self.chassis_slip_angle = 0.0  # np.arctan2(velocity.y, velocity.x)
        self.yaw_rate = 0.0  # odom_msg.twist.twist.angular.z
        heading = Rotation.from_quat(
            [orientation.x, orientation.y, orientation.z, orientation.w]
        ).as_euler("ZYX")[0]

        vehicle_state = State(
            x=position.x,
            y=position.y,
            delta=self.steering_angle,
            v=self.velocity,
            yaw=heading,
            yawrate=self.yaw_rate,
            beta=self.chassis_slip_angle,
        )

        # Calculate the next reference trajectory for the next T steps
        #     with current vehicle pose.
        #     ref_x, ref_y, ref_yaw, ref_v are columns of self.waypoints
        ref_path = self.calc_ref_trajectory(vehicle_state)
        x0 = [vehicle_state.x, vehicle_state.y, vehicle_state.v, vehicle_state.yaw]

        (
            self.oa,
            self.odelta_v,
            ox,
            oy,
            oyaw,
            ov,
            state_predict,
        ) = self.linear_mpc_control(ref_path, x0, self.oa, self.odelta_v)

        if ox is None or oy is None:
            return

        markers = MarkerArray()
        markers.markers = [
            # visualize the waypoints
            visualize.points_to_line_marker(
                self.trajectory[:, :2],
                color=(0, 1, 0, 0.5),
                frame_id="map",
                id=0,
                ns="mpc",
            ),
            # visualize heading
            # visualize.points_to_arrow_markers(
            #     self.optimal_trajectory[:, :2],
            #     self.optimal_trajectory[:, 3],
            # ),
            # visualize the reference trajectory
            visualize.points_to_line_marker(
                np.stack(ref_path[:2], axis=1),
                color=(0, 1, 0),
                frame_id="map",
                id=1,
                ns="mpc",
            ),
            # visualize mpc trajectory ox, oy
            visualize.points_to_line_marker(
                np.stack((ox, oy), axis=1),
                color=(1, 0, 0),
                frame_id="map",
                id=2,
                ns="mpc",
            ),
            # show x0
            visualize.point_to_sphere_marker(np.asarray(x0[:2]), color=(0, 0, 1), id=3),
        ]
        self.pub_visualize.publish(markers)

        steer_output = self.odelta_v[0]
        speed_output = vehicle_state.v + self.oa[0] * self.config.DTK

        self.steering_angle = steer_output

        msg = AckermannDriveStamped()
        msg.drive.steering_angle = steer_output
        msg.drive.speed = speed_output * self.speed_factor
        self.pub_drive.publish(msg)

    def mpc_prob_init(self):
        """
        Create MPC quadratic optimization problem using cvxpy, solver: OSQP
        Will be solved every iteration for control.
        More MPC problem information here: https://osqp.org/docs/examples/mpc.html
        More QP example in CVXPY here: https://www.cvxpy.org/examples/basic/quadratic_program.html
        """
        # Initialize and create vectors for the optimization problem
        # Vehicle State Vector
        self.xk = cvxpy.Variable((self.config.NXK, self.config.TK + 1))
        # Control Input vector
        self.uk = cvxpy.Variable((self.config.NU, self.config.TK))
        objective = 0.0  # Objective value of the optimization problem
        constraints = []  # Create constraints array

        # Initialize reference vectors
        self.x0k = cvxpy.Parameter((self.config.NXK,))
        self.x0k.value = np.zeros((self.config.NXK,))

        # Initialize reference trajectory parameter
        self.ref_traj_k = cvxpy.Parameter((self.config.NXK, self.config.TK + 1))
        self.ref_traj_k.value = np.zeros((self.config.NXK, self.config.TK + 1))

        # reference trajectory speed maximum
        self.ref_traj_v_max = cvxpy.Parameter((self.config.TK + 1,))
        self.ref_traj_v_max.value = np.zeros((self.config.TK + 1,))

        # Initializes block diagonal form of R = [R, R, ..., R] (NU*T, NU*T)
        R_block = block_diag(tuple([self.config.Rk] * self.config.TK))

        # Initializes block diagonal form of Rd = [Rd, ..., Rd] (NU*(T-1), NU*(T-1))
        Rd_block = block_diag(tuple([self.config.Rdk] * (self.config.TK - 1)))

        # Initializes block diagonal form of Q = [Q, Q, ..., Qf] (NX*T, NX*T)
        Q_block = [self.config.Qk] * (self.config.TK)
        Q_block.append(self.config.Qfk)
        Q_block = block_diag(tuple(Q_block))

        # Formulate and create the finite-horizon optimal control problem (objective function)
        # The FTOCP has the horizon of T timesteps

        # --------------------------------------------------------
        # Objective part 1: Influence of the control inputs: Inputs u multiplied by the penalty R
        objective = cvxpy.quad_form(cvxpy.vec(self.uk), R_block)

        # Objective part 2: Deviation of the vehicle from the reference trajectory weighted by Q, including final Timestep T weighted by Qf
        objective += cvxpy.quad_form(cvxpy.vec(self.xk - self.ref_traj_k), Q_block)

        # Objective part 3: Difference from one control input to the next control input weighted by Rd
        objective += cvxpy.quad_form(
            cvxpy.vec(self.uk[:, 1:] - self.uk[:, :-1]), Rd_block
        )
        # --------------------------------------------------------

        # Constraints 1: Calculate the future vehicle behavior/states based on the vehicle dynamics model matrices
        # Evaluate vehicle Dynamics for next T timesteps
        A_block = []
        B_block = []
        C_block = []
        # init path to zeros
        path_predict = np.zeros((self.config.NXK, self.config.TK + 1))
        for t in range(self.config.TK):
            A, B, C = self.get_model_matrix(path_predict[2, t], path_predict[3, t], 0.0)
            A_block.append(A)
            B_block.append(B)
            C_block.extend(C)

        A_block = block_diag(tuple(A_block))
        B_block = block_diag(tuple(B_block))
        C_block = np.array(C_block)

        # [AA] Sparse matrix to CVX parameter for proper stuffing
        # Reference: https://github.com/cvxpy/cvxpy/issues/1159#issuecomment-718925710
        m, n = A_block.shape
        self.Annz_k = cvxpy.Parameter(A_block.nnz)
        data = np.ones(self.Annz_k.size)
        rows = A_block.row * n + A_block.col
        cols = np.arange(self.Annz_k.size)
        Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Annz_k.size))

        # Setting sparse matrix data
        self.Annz_k.value = A_block.data

        # Now we use this sparse version instead of the old A_ block matrix
        self.Ak_ = cvxpy.reshape(Indexer @ self.Annz_k, (m, n), order="C")

        # Same as A
        m, n = B_block.shape
        self.Bnnz_k = cvxpy.Parameter(B_block.nnz)
        data = np.ones(self.Bnnz_k.size)
        rows = B_block.row * n + B_block.col
        cols = np.arange(self.Bnnz_k.size)
        Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Bnnz_k.size))
        self.Bk_ = cvxpy.reshape(Indexer @ self.Bnnz_k, (m, n), order="C")
        self.Bnnz_k.value = B_block.data

        # No need for sparse matrices for C as most values are parameters
        self.Ck_ = cvxpy.Parameter(C_block.shape)
        self.Ck_.value = C_block

        # -------------------------------------------------------------

        # get the needed dimensions from the state and input vectors
        velocity = self.xk[2, :]
        acceleration = self.uk[0, :]
        steering_angle = self.uk[1, :]

        # Constraint part 1:
        #     Add dynamics constraints to the optimization problem
        #     This constraint should be based on a few variables:
        #     self.xk, self.Ak_, self.Bk_, self.uk, and self.Ck_
        constraints.append(
            cvxpy.vec(self.xk[:, 1:])
            == self.Ak_ @ cvxpy.vec(self.xk[:, :-1])
            + self.Bk_ @ cvxpy.vec(self.uk)
            + self.Ck_
        )

        # Constraint part 2:
        #     Add constraints on steering, change in steering angle
        #     cannot exceed steering angle speed limit. Should be based on:
        #     self.uk, self.config.MAX_DSTEER, self.config.DTK
        steering_angle_change = steering_angle[1:] - steering_angle[:-1]
        constraints.append(
            cvxpy.norm_inf(steering_angle_change)
            <= self.config.MAX_DSTEER * self.config.DTK  # type: ignore
        )

        # Constraint part 3:
        #     Add constraints on upper and lower bounds of states and inputs
        #     and initial state constraint, should be based on:
        #     self.xk, self.x0k, self.config.MAX_SPEED, self.config.MIN_SPEED,
        #     self.uk, self.config.MAX_ACCEL, self.config.MAX_STEER
        constraints.append(cvxpy.vec(self.xk[:, 0]) == self.x0k)
        constraints.append(cvxpy.max(steering_angle) <= self.config.MAX_STEER)  # type: ignore
        constraints.append(cvxpy.min(steering_angle) >= self.config.MIN_STEER)  # type: ignore
        constraints.append(cvxpy.norm_inf(acceleration) <= self.config.MAX_ACCEL)  # type: ignore

        constraints.append(cvxpy.max(velocity) <= self.config.MAX_SPEED)  # type: ignore
        constraints.append(cvxpy.min(velocity) >= self.config.MIN_SPEED)  # type: ignore
        # constraints.append(velocity <= self.ref_traj_v_max)  # type: ignore

        # -------------------------------------------------------------

        # Create the optimization problem in CVXPY and setup the workspace
        # Optimization goal: minimize the objective function
        self.MPC_prob = cvxpy.Problem(cvxpy.Minimize(objective), constraints)

    def calc_ref_trajectory(self, state):
        _, _, _, nearest_idx = nearest_point(
            np.array([state.x, state.y]), self.trajectory[:, :2]
        )
        lap_step = self.lap_length / len(self.trajectory)
        # position along lap
        s = np.zeros(self.config.TK + 1)
        speed = np.zeros(self.config.TK + 1)
        s[0] = nearest_idx * lap_step
        speed[0] = state.v
        for i in range(self.config.TK):
            idx = int(s[i] / lap_step)
            speed[i + 1] = self.trajectory[idx, 2]
            dv_max = self.config.MAX_ACCEL * self.config.DTK
            speed[i + 1] = min(speed[i] + dv_max, speed[i + 1])
            speed[i + 1] = max(speed[i] - dv_max, speed[i + 1])
            # advance by portion of lap at optimal speed
            avg_speed = (speed[i] + speed[i + 1]) / 2.0
            s[i + 1] = (s[i] + self.config.DTK * avg_speed) % self.lap_length

        # convert s to index
        optimal_idx = (s / lap_step).astype(int)
        # position & speed from optimal trajectory
        position = self.trajectory[optimal_idx, :2]
        speed = self.trajectory[optimal_idx[0].astype(int), 2]

        # heading
        heading = self.trajectory[optimal_idx, 3]
        heading[0] = state.yaw
        heading = np.unwrap(heading)

        ref_traj = np.zeros((self.config.NXK, self.config.TK + 1))
        ref_traj[:2] = position.T
        ref_traj[2] = speed
        ref_traj[3] = heading
        return ref_traj

    def predict_motion(self, x0, oa, od, xref):
        path_predict = xref * 0.0
        for i, _ in enumerate(x0):
            path_predict[i, 0] = x0[i]

        state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
        for ai, di, i in zip(oa, od, range(1, self.config.TK + 1)):
            state = self.update_state(state, ai, di)
            path_predict[0, i] = state.x
            path_predict[1, i] = state.y
            path_predict[2, i] = state.v
            path_predict[3, i] = state.yaw

        return path_predict

    def update_state(self, state, a, delta):
        # input check
        if delta >= self.config.MAX_STEER:
            delta = self.config.MAX_STEER
        elif delta <= -self.config.MAX_STEER:
            delta = -self.config.MAX_STEER

        state.x = state.x + state.v * math.cos(state.yaw) * self.config.DTK
        state.y = state.y + state.v * math.sin(state.yaw) * self.config.DTK
        state.yaw = (
            state.yaw + (state.v / self.config.WB) * math.tan(delta) * self.config.DTK
        )
        state.v = state.v + a * self.config.DTK

        if state.v > self.config.MAX_SPEED:
            state.v = self.config.MAX_SPEED
        elif state.v < self.config.MIN_SPEED:
            state.v = self.config.MIN_SPEED

        return state

    def get_model_matrix(self, v, phi, delta):
        """
        Calc linear and discrete time dynamic model-> Explicit discrete time-invariant
        Linear System: Xdot = Ax +Bu + C
        State vector: x=[x, y, v, yaw]
        :param v: speed
        :param phi: heading angle of the vehicle
        :param delta: steering angle: delta_bar
        :return: A, B, C
        """

        # State (or system) matrix A, 4x4
        A = np.zeros((self.config.NXK, self.config.NXK))
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[3, 3] = 1.0
        A[0, 2] = self.config.DTK * math.cos(phi)
        A[0, 3] = -self.config.DTK * v * math.sin(phi)
        A[1, 2] = self.config.DTK * math.sin(phi)
        A[1, 3] = self.config.DTK * v * math.cos(phi)
        A[3, 2] = self.config.DTK * math.tan(delta) / self.config.WB

        # Input Matrix B; 4x2
        B = np.zeros((self.config.NXK, self.config.NU))
        B[2, 0] = self.config.DTK
        B[3, 1] = self.config.DTK * v / (self.config.WB * math.cos(delta) ** 2)

        C = np.zeros(self.config.NXK)
        C[0] = self.config.DTK * v * math.sin(phi) * phi
        C[1] = -self.config.DTK * v * math.cos(phi) * phi
        C[3] = -self.config.DTK * v * delta / (self.config.WB * math.cos(delta) ** 2)

        return A, B, C

    def mpc_prob_solve(self, ref_traj, path_predict, x0):
        self.x0k.value = x0
        self.ref_traj_v_max.value = ref_traj[2]

        A_block = []
        B_block = []
        C_block = []
        for t in range(self.config.TK):
            A, B, C = self.get_model_matrix(path_predict[2, t], path_predict[3, t], 0.0)
            A_block.append(A)
            B_block.append(B)
            C_block.extend(C)

        A_block = block_diag(tuple(A_block))
        B_block = block_diag(tuple(B_block))
        C_block = np.array(C_block)

        self.Annz_k.value = A_block.data
        self.Bnnz_k.value = B_block.data
        self.Ck_.value = C_block

        self.ref_traj_k.value = ref_traj

        # Solve the optimization problem in CVXPY
        # Solver selections: cvxpy.OSQP; cvxpy.GUROBI
        self.MPC_prob.solve(solver=cvxpy.OSQP, verbose=False, warm_start=True)

        if (
            self.MPC_prob.status == cvxpy.OPTIMAL
            or self.MPC_prob.status == cvxpy.OPTIMAL_INACCURATE
        ):
            ox = np.array(self.xk.value[0, :]).flatten()
            oy = np.array(self.xk.value[1, :]).flatten()
            ov = np.array(self.xk.value[2, :]).flatten()
            oyaw = np.array(self.xk.value[3, :]).flatten()
            oa = np.array(self.uk.value[0, :]).flatten()
            odelta = np.array(self.uk.value[1, :]).flatten()

        else:
            self.get_logger().error("Cannot solve mpc.")
            oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None

        return oa, odelta, ox, oy, oyaw, ov

    def linear_mpc_control(self, ref_path, x0, oa, od):
        """
        MPC contorl with updating operational point iteraitvely
        :param ref_path: reference trajectory in T steps
        :param x0: initial state vector
        :param oa: acceleration of T steps of last time
        :param od: delta of T steps of last time
        """

        if oa is None or od is None:
            oa = [0.0] * self.config.TK
            od = [0.0] * self.config.TK

        # Call the Motion Prediction function: Predict the vehicle motion for x-steps
        path_predict = self.predict_motion(x0, oa, od, ref_path)

        # Run the MPC optimization: Create and solve the optimization problem
        mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v = self.mpc_prob_solve(
            ref_path, path_predict, x0
        )

        return mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v, path_predict


def main(args=None):
    rclpy.init(args=args)
    print("MPC Initialized")
    mpc_node = MPC()
    rclpy.spin(mpc_node)

    mpc_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
