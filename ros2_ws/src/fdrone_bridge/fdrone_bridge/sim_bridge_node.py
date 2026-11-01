"""ROS 2 ノード: Genesis の 1 env を「ロボット側」として動かす (全体構成の図の
[Genesis 1 env] に相当。実機の Crazyswarm2 スタックの代わりに、デモとしてこのノードが
センサ値 (/fdrone/odom 等) を出し、方策ノードの行動 (/fdrone/action) を受けて物理を進める)。

タイミングは「状態を出す → 行動を受け取る → 1 回 env.step() する」の往復 (ping-pong)。
`env.step()` 自体が decimation 物理ステップ (既定 4 = 100 Hz) を内部で進めるので、
ROS のメッセージの往復がそれより遅くても、1 回の env.step() が表す時間は変わらない
(ノード間の往復レイテンシは学習時の制御周期をそのまま再現するものではない、デモ用の簡略化)。

genesis に依存する (CPU)。方策ノード (policy_node.py) は依存しない。
"""
import math

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from std_msgs.msg import Int32, Float32MultiArray
import torch

import genesis as gs
from fdrone.configs import get_cfgs
from fdrone.envs.fault_land_env import FaultLandEnv


class SimBridgeNode(Node):
    def __init__(self):
        super().__init__("fdrone_sim_bridge_node")
        self.declare_parameter("alt", 10.0)
        self.declare_parameter("fault_rotor", 0)
        self.declare_parameter("fault_time", 2.0)
        self.declare_parameter("speed", 2.0)
        self.declare_parameter("heading_deg", 0.0)
        self.declare_parameter("decimation", 4)
        self.declare_parameter("seed", 0)
        alt = float(self.get_parameter("alt").value)
        fault_rotor = int(self.get_parameter("fault_rotor").value)
        fault_time = float(self.get_parameter("fault_time").value)
        speed = float(self.get_parameter("speed").value)
        heading_deg = float(self.get_parameter("heading_deg").value)
        decimation = int(self.get_parameter("decimation").value)
        seed = int(self.get_parameter("seed").value)

        gs.init(backend=gs.cpu, logging_level="warning", seed=seed)
        torch.manual_seed(seed)  # landing_offset の無作為なずれの再現性
        # `d4g_*` (cf2x の最終モデル、yawpen・移動中の故障・着地目標オフセット・100 Hz) と
        # 同じ条件。実際に動かすチェックポイントとずれないよう、学習済みの exp と揃える前提
        env_cfg, obs_cfg, reward_cfg = get_cfgs(stage=2, variant="yawpen")
        env_cfg["fault"]["prob"] = 0.0  # 決定的に force_fault で注入する
        env_cfg["init_alt_range"] = (alt, alt)
        env_cfg["decimation"] = decimation
        env_cfg["translate"]["enabled"] = speed > 0.0
        env_cfg["landing_offset"]["enabled"] = True
        env_cfg.setdefault("domain_rand", {})["enabled"] = False
        self.env = FaultLandEnv(num_envs=1, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg)
        self.env.reset()
        if speed > 0.0:
            # eval_utils.setup() と同じやり方 (故障前の水平移動、C4)
            heading = math.radians(heading_deg)
            self.env.v_cmd[0, 0] = speed * math.cos(heading)
            self.env.v_cmd[0, 1] = speed * math.sin(heading)
            vel6 = torch.zeros((1, 6))
            vel6[0, :2] = self.env.v_cmd[0, :2]
            self.env.ds.drone.set_dofs_velocity(vel6, envs_idx=torch.tensor([0]))
        self.env.fault.force_fault(env_idx=0, rotor_idx=fault_rotor, step=int(fault_time / self.env.dt))
        self.get_logger().info(
            f"scenario: alt={alt} fault_rotor={fault_rotor} fault_time={fault_time} "
            f"speed={speed} heading_deg={heading_deg} decimation={decimation} (policy {1 / self.env.policy_dt:.0f} Hz)"
        )

        self.odom_pub = self.create_publisher(Odometry, "/fdrone/odom", 10)
        self.target_pub = self.create_publisher(Point, "/fdrone/target", 10)
        self.fault_pub = self.create_publisher(Int32, "/fdrone/fault", 10)
        self.motor_rpm_pub = self.create_publisher(Float32MultiArray, "/fdrone/motor_rpm", 10)
        self.create_subscription(Float32MultiArray, "/fdrone/action", self._on_action, 10)

        self.done = False
        self._step_count = 0
        self._publish_state()

    def _publish_state(self):
        ds = self.env.ds
        q = ds.base_quat[0]
        odom = Odometry()
        pos = ds.base_pos[0]
        odom.pose.pose.position.x = float(pos[0])
        odom.pose.pose.position.y = float(pos[1])
        odom.pose.pose.position.z = float(pos[2])
        # 本プロジェクト・Genesis は w,x,y,z。geometry_msgs/Quaternion は x,y,z,w (ROS の慣習)
        odom.pose.pose.orientation.w = float(q[0])
        odom.pose.pose.orientation.x = float(q[1])
        odom.pose.pose.orientation.y = float(q[2])
        odom.pose.pose.orientation.z = float(q[3])
        lv, av = ds.base_lin_vel[0], ds.base_ang_vel[0]
        odom.twist.twist.linear.x = float(lv[0])
        odom.twist.twist.linear.y = float(lv[1])
        odom.twist.twist.linear.z = float(lv[2])
        odom.twist.twist.angular.x = float(av[0])
        odom.twist.twist.angular.y = float(av[1])
        odom.twist.twist.angular.z = float(av[2])
        self.odom_pub.publish(odom)

        t = self.env.target[0]
        self.target_pub.publish(Point(x=float(t[0]), y=float(t[1]), z=float(t[2])))

        faulted = bool(self.env.faulted[0])
        fault_rotor = int(self.env.fault.fault_rotor[0]) if faulted else -1
        self.fault_pub.publish(Int32(data=fault_rotor))

        rpm_msg = Float32MultiArray()
        rpm_msg.data = [float(x) for x in self.env.motor.rpm[0].tolist()]
        self.motor_rpm_pub.publish(rpm_msg)

    def _on_action(self, msg: Float32MultiArray):
        if self.done:
            return
        action = torch.tensor([list(msg.data)])
        _, _, dones, _ = self.env.step(action)
        self._step_count += 1
        if bool(dones[0]):
            if bool(self.env.success[0]):
                outcome = "success"
            elif bool(self.env.crashed[0]):
                outcome = "out_of_bounds" if bool(self.env.out_of_bounds[0]) else "crash"
            elif bool(self.env.landed_fail[0]):
                outcome = "land_fail"
            else:
                outcome = "timeout"
            self.get_logger().info(f"episode done after {self._step_count} steps ({self._step_count * self.env.policy_dt:.2f}s): {outcome}")
            self.done = True
            return
        self._publish_state()


def main():
    rclpy.init()
    node = SimBridgeNode()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=1.0)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
