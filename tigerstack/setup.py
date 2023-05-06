import os
from glob import glob

from setuptools import setup

package_name = "tigerstack"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "maps"), glob("maps/*")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="damow",
    maintainer_email="Damowerko@users.noreply.github.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "gap_follow = tigerstack.gap_follow:main",
            "safety_node = tigerstack.safety_node:main",
            "imu_framer = tigerstack.imu_framer:main",
            "mpc_node = tigerstack.mpc.mpc_node:main",
            "mpc_rrt_node = tigerstack.mpc_rrt.mpc_rrt_node:main",
            "mpc_oppo = tigerstack.mpc_rrt.mpc_oppo:main",
            "rrt_node = tigerstack.mpc_rrt.rrt_node:main",
            "pure_pursuit = tigerstack.mpc_rrt.pure_pursuit:main",
        ],
    },
)
