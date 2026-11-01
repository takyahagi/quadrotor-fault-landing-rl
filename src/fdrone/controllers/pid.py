"""カスケード PID (位置 -> 速度 -> 姿勢 -> RPM ミキサ)。

Genesis の `examples/drone/quadcopter_controller.py` (Apache-2.0) を元に、
num_envs 個を並列に処理できるよう torch テンソルに書き換えた。
故障時にヨー角を 0 に保とうとして破綻するのは意図的な挙動
(RL との比較ベースラインとして使う。)。
"""
import torch


class _AxisPID:
    def __init__(self, kp, ki, kd, num_envs, device):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral = torch.zeros(num_envs, device=device)
        self.prev_error = torch.zeros(num_envs, device=device)

    def reset_idx(self, envs_idx):
        self.integral[envs_idx] = 0.0
        self.prev_error[envs_idx] = 0.0

    def update(self, error, dt):
        self.integral = self.integral + error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative


# (kp, ki, kd) 各軸。Genesis サンプルのデフォルト値を踏襲。
DEFAULT_PID_PARAMS = {
    "pos": (2.0, 0.0, 0.0),
    "vel": (2.0, 0.0, 0.0),
    "att": (2.0, 0.0, 2.0),
}


class DronePIDController:
    def __init__(self, num_envs: int, dt: float, base_rpm: float, device, pid_params=None):
        p = pid_params or DEFAULT_PID_PARAMS
        self.dt = dt
        self.base_rpm = base_rpm
        self.device = device

        self.pid_pos = [_AxisPID(*p["pos"], num_envs, device) for _ in range(3)]
        self.pid_vel = [_AxisPID(*p["vel"], num_envs, device) for _ in range(3)]
        self.pid_att = [_AxisPID(*p["att"], num_envs, device) for _ in range(3)]  # roll, pitch, yaw

    def reset_idx(self, envs_idx):
        for pid in (*self.pid_pos, *self.pid_vel, *self.pid_att):
            pid.reset_idx(envs_idx)

    def update(self, pos, vel, euler_deg, target_pos) -> torch.Tensor:
        """1 ステップ分の指令 RPM (num_envs, 4) を計算する。

        pos, vel: world frame (num_envs, 3)
        euler_deg: roll, pitch, yaw [deg] (num_envs, 3)
        target_pos: world frame (num_envs, 3)
        """
        err_pos = target_pos - pos
        vel_des = torch.stack(
            [self.pid_pos[i].update(err_pos[:, i], self.dt) for i in range(3)], dim=-1
        )

        err_vel = vel_des - vel
        thrust_del = self.pid_vel[2].update(err_vel[:, 2], self.dt)
        x_vel_del = self.pid_vel[0].update(err_vel[:, 0], self.dt)
        y_vel_del = self.pid_vel[1].update(err_vel[:, 1], self.dt)

        err_att = -euler_deg  # 目標姿勢は常に水平 (roll=pitch=yaw=0)
        roll_del = self.pid_att[0].update(err_att[:, 0], self.dt)
        pitch_del = self.pid_att[1].update(err_att[:, 1], self.dt)
        yaw_del = self.pid_att[2].update(err_att[:, 2], self.dt)

        thrust, roll, pitch, yaw = thrust_del, roll_del, pitch_del, yaw_del
        m1 = self.base_rpm + (thrust - roll - pitch - yaw - x_vel_del + y_vel_del)
        m2 = self.base_rpm + (thrust - roll + pitch + yaw + x_vel_del + y_vel_del)
        m3 = self.base_rpm + (thrust + roll + pitch - yaw + x_vel_del - y_vel_del)
        m4 = self.base_rpm + (thrust + roll - pitch + yaw - x_vel_del - y_vel_del)
        return torch.stack([m1, m2, m3, m4], dim=-1)
