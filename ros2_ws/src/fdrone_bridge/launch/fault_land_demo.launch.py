"""1 シナリオ (高度・故障 rotor・故障時刻・移動速度・向きを指定) を ROS 2 経由で動かすデモ。既定は cf2x の最終モデル `d4g_s1` (yawpen・移動中の故障・着地目標オフセット・100 Hz) 相当の条件。

実行例 (docker/docker-compose.ros2.yml のコンテナ内、colcon build 後):
  ros2 launch fdrone_bridge fault_land_demo.launch.py \
      policy_path:=/workspace/logs/d4g_s1/policy_1499.pt alt:=10.0 fault_rotor:=0 fault_time:=2.0 \
      speed:=2.0 heading_deg:=0.0
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    policy_path = LaunchConfiguration("policy_path")
    alt = LaunchConfiguration("alt")
    fault_rotor = LaunchConfiguration("fault_rotor")
    fault_time = LaunchConfiguration("fault_time")
    speed = LaunchConfiguration("speed")
    heading_deg = LaunchConfiguration("heading_deg")
    decimation = LaunchConfiguration("decimation")
    seed = LaunchConfiguration("seed")

    return LaunchDescription([
        DeclareLaunchArgument("policy_path", description="scripts/export_policy.py が書き出した .pt"),
        DeclareLaunchArgument("alt", default_value="10.0"),
        DeclareLaunchArgument("fault_rotor", default_value="0"),
        DeclareLaunchArgument("fault_time", default_value="2.0"),
        DeclareLaunchArgument("speed", default_value="2.0", description="C4: 故障前の水平移動速度 [m/s] (0 で静止)"),
        DeclareLaunchArgument("heading_deg", default_value="0.0"),
        DeclareLaunchArgument("decimation", default_value="4", description="方策の周期 (4=100Hz)"),
        DeclareLaunchArgument("seed", default_value="0", description="着地目標オフセットの無作為なずれの再現性用"),
        Node(
            package="fdrone_bridge",
            executable="sim_bridge_node",
            name="fdrone_sim_bridge_node",
            parameters=[{
                "alt": alt, "fault_rotor": fault_rotor, "fault_time": fault_time,
                "speed": speed, "heading_deg": heading_deg, "decimation": decimation, "seed": seed,
            }],
            output="screen",
        ),
        Node(
            package="fdrone_bridge",
            executable="policy_node",
            name="fdrone_policy_node",
            parameters=[{"policy_path": policy_path}],
            output="screen",
        ),
    ])
