from typing import Tuple

import numpy as np
import numpy.typing as npt
from scipy.interpolate import splev, splprep


def load_waypoints(path: str) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    """
    Load waypoints from a file.

    Args:
        path: Path to the file.

    Returns:
        An array of shape (n, m) containing the waypoints.
    """
    waypoints = np.loadtxt(path, delimiter=";", dtype=float)
    waypoints[:, 3] = waypoints[:, 3] + np.pi / 2
    return waypoints, waypoints[:, [1, 2, 5, 3]]


def fit_spline(waypoints: npt.NDArray[np.floating]):
    """
    Args:
        waypoints: An array of shape (n, 2) containing the x,y coordinates of the waypoints.
    """
    assert waypoints.shape[1] == 2
    tck, _ = splprep(waypoints.T, s=0, per=True)
    return tck


def spline_length(tck, n_samples=1000) -> float:
    """
    Args:
        tck: The output of `fit_spline`.
        n_samples: Number of samples to use when computing the length.
    """
    u = np.linspace(0, 1, n_samples)
    dx = np.stack(splev(u, tck, der=1), axis=1) / n_samples  # type: ignore
    return np.sum(np.linalg.norm(dx, axis=1))


def spline_nearest(
    tck, position: npt.NDArray[np.floating], n_samples=1000
) -> npt.NDArray[np.floating]:
    """
    Args:
        tck: The output of `fit_spline`.
        position: (x, y) The position to find the nearest point on the spline to.
    Returns:
        The parameter u on the spline that corresponds to the nearest point.
    """
    u = np.linspace(0, 1, n_samples)
    points = np.stack(splev(u, tck, der=0), axis=1)  # type: ignore
    distances = np.linalg.norm(points - position, axis=1)
    return u[np.argmin(distances)]


def spline_positions(tck, u) -> npt.NDArray[np.floating]:
    """
    Increase the number of waypoints by interpolating between them.
    Args:
        tck: The output of `fit_spline`.
        n_samples: Number of samples to use when interpolating.

    Returns:
        An array of shape (n_points, 2) containing the interpolated waypoints.
        The columns are: x, y.
    """
    positions = np.stack(splev(u, tck, der=0), axis=1)  # type: ignore
    return positions


def spline_heading(tck, u) -> npt.NDArray[np.floating]:
    """
    Compute the heading along a spline using the tangent vector.
    Args:
        tck: The output of `fit_spline`.
        u: The parameter on the spline to compute the heading at.
    """

    derivative = np.stack(splev(u, tck, der=1), axis=1)  # type: ignore
    # re-compute heading for each waypoint
    heading = np.arctan2(
        derivative[:, 1],
        derivative[:, 0],
    )
    return heading


def spline_curvature(tck, u) -> npt.NDArray[np.floating]:
    """
    Compute the curvature (2nd) derivative along a spline at points u.
    Args:
        tck: The output of `fit_spline`.
        u: The parameter on the spline to compute the heading at.
    Returns:
        (n, 2) array of the 2nd derivative at each point u.
    """
    dx = np.stack(splev(u, tck, der=1), axis=1)  # type: ignore
    ddx = np.stack(splev(u, tck, der=2), axis=1)  # type: ignore
    return np.cross(dx, ddx) / np.linalg.norm(dx, axis=1) ** 3


def optimal_trajectory(
    tck,
    max_speed: float,
    max_acceleration: float,
    max_braking: float,
    max_lateral_acceleration: float,
    n_samples: int = 1000,
):
    """
    Args:
        waypoints: An array of shape (n, 2) containing the x,y coordinates of the waypoints.
    """
    u = np.linspace(0, 1, n_samples)
    positions = spline_positions(tck, u)
    heading = spline_heading(tck, u)
    curvature = spline_curvature(tck, u)

    distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    speed = np.sqrt(max_lateral_acceleration / np.abs(curvature))
    speed = np.minimum(speed, max_speed)

    # solve for speed using the forward/reverse method
    # 1. forward pass: assume braking is instantaneous
    # 2. backward pass: assume acceleration is instantaneous
    max_iterations = 100
    for iteration in range(max_iterations):
        speed[0] = speed[-1] = np.min([speed[0], speed[-1]])
        speed_forward = np.copy(speed)
        for i in range(len(speed) - 1):
            dt = distances[i] / speed[i]
            dv_max = max_acceleration * dt
            speed_forward[i + 1] = np.minimum(
                speed_forward[i + 1], speed_forward[i] + dv_max
            )
        speed[0] = speed[-1] = np.min([speed[0], speed[-1]])
        speed_backward = np.copy(speed)
        for i in range(len(speed) - 1, 0, -1):
            dt = distances[i - 1] / speed[i - 1]
            dv_max = max_braking * dt
            speed_backward[i - 1] = np.minimum(
                speed_backward[i - 1], speed_backward[i] + dv_max
            )
        speed = np.minimum(speed_forward, speed_backward)

        # if starting and ending speed are the same, we're done
        if np.allclose(speed[0], speed[-1]):
            break
        elif iteration == max_iterations - 1:
            raise RuntimeError("Failed to converge on a solution.")
    return np.hstack([positions, speed[:, None], heading[:, None]])


def trajectory_stats(trajectory: npt.NDArray[np.floating]):
    """
    Args:
        trajectory: An array of shape (n, 4) containing the x,y,speed,heading of the trajectory.
    """
    position = trajectory[:, :2]
    speed = trajectory[:, 2]
    distances = np.linalg.norm(np.diff(position, axis=0), axis=1)
    length = np.sum(distances)
    time = np.sum(distances / speed[:-1])
    return length, time


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    waypoints = load_waypoints(
        "/home/damow/ese615/lab_ws/src/lab7/mpc/maps/levine_2nd.csv"
    )
    tck = fit_spline(waypoints[:, :2])
    trajectory = optimal_trajectory(
        tck,
        max_speed=10.0,
        max_acceleration=6.0,
        max_braking=3.0,
        max_lateral_acceleration=1.0,
        n_samples=1000,
    )
    x, y, speed, heading = trajectory.T
    length = spline_length(fit_spline(waypoints[:, :2]))
    time = np.cumsum(
        np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=1) / speed[:-1]
    )

    # plot the trajectory
    plt.figure()
    plt.plot(time, speed[:-1], label="speed")
    plt.legend()
    plt.savefig("speed.png")
    # plt.show()
