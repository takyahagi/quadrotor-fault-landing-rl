from setuptools import find_packages, setup

package_name = "fdrone_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/fault_land_demo.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="fdrone",
    maintainer_email="noreply@example.com",
    description="Fault-landing policy inference node and a Genesis-backed sim bridge node",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "policy_node = fdrone_bridge.policy_node:main",
            "sim_bridge_node = fdrone_bridge.sim_bridge_node:main",
        ],
    },
)
