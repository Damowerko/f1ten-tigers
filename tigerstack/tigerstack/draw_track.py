import csv
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.backend_bases import MouseButton


def draw_track(map_path: Path):
    output_path = map_path.with_suffix(".csv")
    track_img = plt.imread(str(map_path))
    config = yaml.load(map_path.with_suffix(".yaml").read_text(), yaml.CLoader)

    origin = config["origin"]
    resolution = config["resolution"]
    width = track_img.shape[1] * resolution
    height = track_img.shape[0] * resolution
    extent = [origin[0], origin[0] + width, origin[1], origin[1] + height]

    plt.figure()
    plt.imshow(track_img, cmap="gray", extent=extent)

    clicks = []

    def on_click(event):
        if event.button is MouseButton.LEFT:
            clicks.append((event.xdata, event.ydata))
            plt.plot(event.xdata, event.ydata, "ro")
            plt.draw()

    plt.connect("button_press_event", on_click)
    plt.show()

    # save to csv with columns x_m,y_m,w_tr_right_m,w_tr_left_m
    with open(output_path, "w") as f:
        f.write("# x_m,y_m,w_tr_right_m,w_tr_left_m\n")
        writer = csv.writer(f)
        positions = np.array(clicks)
        for i in range(len(positions) // 2):
            left = positions[2 * i]
            right = positions[2 * i + 1]

            center = (left + right) / 2
            width_left = np.linalg.norm(left - center)
            width_right = np.linalg.norm(right - center)
            writer.writerow([*center, width_right, width_left])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--map_path",
        type=Path,
        default=Path("lab_ws/src/f1ten-tigers/tigerstack/maps/outside.pgm"),
    )
    args = parser.parse_args()

    draw_track(args.map_path)
