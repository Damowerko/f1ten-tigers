from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from skimage.draw import line
from skimage.morphology import binary_dilation, disk


class LaserOccupancy:
    def __init__(
        self, size: float, resolution: float, origin: Optional[npt.ArrayLike] = None
    ) -> None:
        """
        Binary occupancy grid. Occupied cells are True, free cells are False.

        The positions are in (x,y) coordinates in meters. Cell coordinates are (ix, iy).
        The origin parameter is the position of the (0,0) cell coordinate.

        Args:
            size: (width, height) Size of the grid in meters.
            resolution: Resolution of the grid in meters per pixel.
            origin:  Position of the (0,0) cell coordinate. (-size/2, -size/2) by default.
        """
        self.size = size
        self.filled = False
        self.resolution = resolution
        self.origin: npt.NDArray[np.float_] = (
            np.array([-size / 2, -size / 2], dtype=float)
            if origin is None
            else np.array(origin, dtype=float)
        )
        self.grid: npt.NDArray[np.bool_] = np.zeros(
            (int(size / resolution), int(size / resolution)), dtype=bool
        )

    def get_positions(self, positions: npt.NDArray[np.float_]) -> npt.NDArray[np.bool_]:
        pixels = self.positions_to_pixels(positions)
        return self.grid[pixels[:, 0], pixels[:, 1]].copy()

    def is_line_free(
        self, start: npt.NDArray[np.float_], end: npt.NDArray[np.float_]
    ) -> bool:
        """
        Check if a line between two positions is free.

        Args:
            start: (x,y) Start position in meters.
            end: (x,y) End position in meters.

        Returns:
            True if the line is free, False otherwise.
        """
        pixels = self.line_between_positions(np.flip(start), np.flip(end))
        return not self.grid[pixels[:, 0], pixels[:, 1]].any()

    def line_between_positions(
        self, start: npt.NDArray[np.float_], end: npt.NDArray[np.float_]
    ):
        """
        Find the pixel coordinates along a line betwen the start and end positions.
        Args:
            start: (x,y) Start position in meters.
            end: (x,y) End position in meters.
        Returns:
            (n, 2) Array of pixel coordinates (ix, iy).
        """
        start_pixel = self.positions_to_pixels(start[None, :])[0]
        end_pixel = self.positions_to_pixels(end[None, :])[0]
        row_idx, col_idx = line(*start_pixel, *end_pixel)
        pixels = np.stack([row_idx, col_idx], axis=1)
        return pixels

    def positions_to_pixels(
        self, positions: npt.NDArray[np.float_]
    ) -> npt.NDArray[np.int_]:
        """
        Convert an array of positions in meters to an index in the occupancy grid.

        Args:
            position: (n, 2) Position (x,y) in meters.

        Returns:
            (n, 2) Index (i,j) in the grid.
        """
        return np.round((positions - self.origin) / self.resolution).astype(int)

    def from_scan(
        self,
        angles: npt.NDArray[np.float_],
        ranges: npt.NDArray[np.float_],
    ) -> None:
        """
        Update from a laser scan.

        Args:
            angles: Array of angles in radians. Angle 0 is in the x direction. Angles are measured counter-clockwise.
            ranges: Array of ranges in meters.
        """
        self.grid.fill(False)
        directions = np.array([np.cos(angles), np.sin(angles)]).T
        positions = directions * ranges[:, None]
        pixels = self.positions_to_pixels(positions)
        # ignore pixels outside the grid
        pixels = pixels[
            (pixels[:, 0] < int(self.size / self.resolution))
            & (pixels[:, 1] < int(self.size / self.resolution))
            & (pixels[:, 0] >= 0)
            & (pixels[:, 1] >= 0)
        ]
        self.grid[pixels[:, 0], pixels[:, 1]] = True
        self.filled = True

    def dilate(self, radius: float) -> None:
        """
        Dilate the grid.

        Args:
            radius: Radius of the dilation in meters.
        """
        self.grid = binary_dilation(self.grid, disk(int(radius / self.resolution)))


if __name__ == "__main__":
    # test on a 1/4 circle
    size = 8
    resolution = 0.1
    n_particles = 100

    grid = LaserOccupancy(size, resolution)
    angles = np.linspace(0, np.pi / 2, n_particles)
    ranges = np.ones(n_particles) * 2

    grid.from_scan(angles, ranges)
    grid.dilate(0.2)

    plt.imshow(
        grid.grid.T, extent=[-size / 2, size / 2, -size / 2, size / 2], origin="lower"
    )
    plt.show()
