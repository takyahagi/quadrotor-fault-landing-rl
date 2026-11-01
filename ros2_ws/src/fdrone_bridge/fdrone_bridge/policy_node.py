"""ROS 2 ノード: 観測の組み立て + 方策推論。

実機・Genesis のどちらにも依存しない (`fdrone.deploy.*` のみ使う。rclpy 以外は
`import genesis` しない)。`scripts/export_policy.py` が書き出した .pt が
`descent_cfg`・`motor_max_rpm`・`obs_scales` を学習時の値のまま 1 ファイルに
まとめて持っているので、ここでそれらを別途指定し直す必要がない (ずれ・重複を避ける)。
"""
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from std_msgs.msg import Int32, Float32MultiArray
import torch

from fdrone.deploy.policy import StandalonePolicy
from fdrone.deploy.observation import ObservationBuilder


class PolicyNode(Node):
    def __init__(self):
        super().__init__("fdrone_policy_node")
        self.declare_parameter("policy_path", "")
        path = self.get_parameter("policy_path").get_parameter_value().string_value
        if not path:
            raise RuntimeError("policy_path パラメータが未設定 (export_policy.py の出力 .pt を指定する)")

        data = torch.load(path, map_location="cpu", weights_only=False)
        self.policy = StandalonePolicy(path)
        self.builder = ObservationBuilder(data["descent_cfg"], data["obs_scales"])
        self.motor_max_rpm = data["motor_max_rpm"]
        self.get_logger().info(f"loaded {path} (source={data.get('source')})")

        self._target = torch.zeros(3)
        self._fault_rotor = -1
        # ホバリング相当の初期値 (frac=0.5 相当)。/fdrone/motor_rpm の最初の値が来るまでの代用
        self._motor_rpm = torch.full((4,), self.motor_max_rpm * 0.5**0.5)

        self.action_pub = self.create_publisher(Float32MultiArray, "/fdrone/action", 10)
        self.create_subscription(Odometry, "/fdrone/odom", self._on_odom, 10)
        self.create_subscription(Point, "/fdrone/target", self._on_target, 10)
        self.create_subscription(Int32, "/fdrone/fault", self._on_fault, 10)
        self.create_subscription(Float32MultiArray, "/fdrone/motor_rpm", self._on_motor_rpm, 10)

    def _on_target(self, msg: Point):
        self._target = torch.tensor([msg.x, msg.y, msg.z])

    def _on_fault(self, msg: Int32):
        self._fault_rotor = int(msg.data)

    def _on_motor_rpm(self, msg: Float32MultiArray):
        self._motor_rpm = torch.tensor(list(msg.data))

    def _on_odom(self, msg: Odometry):
        # geometry_msgs/Quaternion は x,y,z,w (ROS の慣習)。本プロジェクト・Genesis は w,x,y,z
        q = msg.pose.pose.orientation
        quat_wxyz = torch.tensor([q.w, q.x, q.y, q.z])
        p = msg.pose.pose.position
        pos_world = torch.tensor([p.x, p.y, p.z])
        lv, av = msg.twist.twist.linear, msg.twist.twist.angular
        lin_vel_body = torch.tensor([lv.x, lv.y, lv.z])
        ang_vel_body = torch.tensor([av.x, av.y, av.z])

        faulted = self._fault_rotor >= 0
        obs = self.builder.build(
            quat_wxyz=quat_wxyz,
            ang_vel_body=ang_vel_body,
            lin_vel_body=lin_vel_body,
            pos_world=pos_world,
            target_world=self._target,
            faulted=faulted,
            fault_rotor=self._fault_rotor,
            motor_rpm=self._motor_rpm,
            motor_max_rpm=self.motor_max_rpm,
        )
        action = self.policy.act(obs)
        self.builder.set_last_actions(action)

        out = Float32MultiArray()
        out.data = [float(x) for x in action.tolist()]
        self.action_pub.publish(out)


def main():
    rclpy.init()
    node = PolicyNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
