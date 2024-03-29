import os
from glob import glob

from setuptools import setup

package_name = "planner"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="zzangupenn, Hongrui Zheng",
    maintainer_email="zzang@seas.upenn.edu, billyzheng.bz@gmail.com",
    description="f1tenth pure_pursuit",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "planner_node = planner.planner_node:main",
        ],
    },
)
