from typing import Sequence

import numpy as np
import numpy.typing as npt
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker


def sequence_to_color(color: Sequence) -> ColorRGBA:
    """
    Convert a sequence of 3 or 4 values to a ColorRGBA message.

    Args:
        color: a sequence of 3 or 4 values between 0 and 1.
    """
    return ColorRGBA(
        r=float(color[0]),
        g=float(color[1]),
        b=float(color[2]),
        a=1.0 if len(color) == 3 else float(color[3]),
    )


def make_marker(
    color=(1.0, 0.0, 0.0), frame_id="map", id=0, ns="default", lifetime=None, scale=0.05
):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.id = id
    marker.ns = ns
    marker.action = Marker.ADD
    marker.color = sequence_to_color(color)
    marker.scale.x = scale
    marker.scale.y = scale
    marker.scale.z = scale
    if lifetime is not None:
        marker.lifetime.sec = int(lifetime)
        marker.lifetime.nanosec = int((lifetime - int(lifetime)) * 1e9)
    return marker


def points_to_line_marker(points: npt.NDArray[np.floating], **kwargs) -> Marker:
    """
    Create a marker for a list of points.

    Args:
        points: a (n, 2) numpy array with the x,y positions.
        **kwargs: keyword arguments to pass to make_marker.
    """

    marker = make_marker(**kwargs)
    marker.points = [Point(x=x, y=y, z=0.0) for x, y in points]
    marker.type = Marker.LINE_STRIP
    return marker


def points_to_arrow_markers(
    points: npt.NDArray[np.floating], headings: npt.NDArray[np.floating], **kwargs
) -> Marker:
    marker = make_marker(**kwargs)
    marker.points = []
    for point, heading in zip(points, headings):
        marker.points.append(Point(x=point[0], y=point[1], z=0.0))
        marker.points.append(
            Point(
                x=float(point[0] + np.cos(heading) * 1),
                y=float(point[1] + np.sin(heading) * 1),
                z=0.0,
            )
        )
    marker.type = Marker.LINE_STRIP
    return marker


def point_to_sphere_marker(point: npt.NDArray[np.floating], **kwargs) -> Marker:
    """
    Create a marker for a point.

    Args:
        point: a (2,) or (3,) numpy array with the x,y,z positions.
        **kwargs: keyword arguments to pass to make_marker.
    """

    marker = make_marker(**kwargs)
    marker.pose.position.x = point[0]
    marker.pose.position.y = point[1]
    marker.pose.position.z = 0.0 if len(point) < 3 else point[2]
    marker.type = Marker.SPHERE
    return marker
