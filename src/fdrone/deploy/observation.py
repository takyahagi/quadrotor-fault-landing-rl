"""`fault_land_env.py` の観測ベクトル (27 次元) を、Genesis に依存せず組み立てる。

`FaultLandEnv._compute_state` / `_get_obs` と**式を揃えてある**。あちらを変えたら
ここも変える (2026-10-18 時点でコードの重複を許容している。ずれると ROS 2 ノードが
学習時と違う観測を方策に渡すことになるので危険)。

クォータニオンは w,x,y,z (Genesis と同じ)。回転は `quat_rotate` で行い、Genesis の
`transform_by_quat` と一致することを確認済み (2026-10-18、最大誤差 3.6e-7)。
"""
import torch


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """v を クォータニオン q (w,x,y,z) で回転する。q, v は (..., 4) / (..., 3)。"""
    w, x, y, z = q.unbind(-1)
    qvec = torch.stack([x, y, z], dim=-1)
    uv = torch.cross(qvec, v, dim=-1)
    uuv = torch.cross(qvec, uv, dim=-1)
    return v + 2 * (w.unsqueeze(-1) * uv + uuv)


def quat_inv(q: torch.Tensor) -> torch.Tensor:
    """単位クォータニオンの逆 (共役)。"""
    w, x, y, z = q.unbind(-1)
    return torch.stack([w, -x, -y, -z], dim=-1)


class ObservationBuilder:
    """1 機体分の観測を組み立てる (バッチなし)。呼び出し側がセンサから最新の状態を更新する。"""

    def __init__(self, descent_cfg: dict, obs_scales: dict, device: str = "cpu"):
        self.descent_cfg = descent_cfg
        self.obs_scales = obs_scales
        self.device = device
        self.last_actions = torch.zeros(4, device=device)

    def build(
        self,
        quat_wxyz: torch.Tensor,  # (4,) 機体の姿勢 (world->body の逆、= body の world での向き)
        ang_vel_body: torch.Tensor,  # (3,) 機体座標系
        lin_vel_body: torch.Tensor,  # (3,) 機体座標系
        pos_world: torch.Tensor,  # (3,)
        target_world: torch.Tensor,  # (3,)
        faulted: bool,
        fault_rotor: int,  # -1 なら故障なし
        motor_rpm: torch.Tensor,  # (4,) 実測 (ESC テレメトリ等)。無ければ前回の指令値で代用
        motor_max_rpm: float,
    ) -> torch.Tensor:
        inv_q = quat_inv(quat_wxyz)
        gravity_world = torch.tensor([0.0, 0.0, -1.0], device=self.device)
        gravity_body = quat_rotate(inv_q, gravity_world)

        err_world = target_world - pos_world
        if faulted:
            err_world = err_world * torch.tensor([1.0, 1.0, 0.0], device=self.device)
        err_body = quat_rotate(inv_q, err_world)

        alt = pos_world[2]
        vz = lin_vel_body_to_world_z(quat_wxyz, lin_vel_body)  # vz は世界座標系の値 (学習時と同じ)
        vc = self.descent_cfg
        if faulted:
            v_ref = -min(vc["v_max"], (vc["v_land"] ** 2 + 2 * vc["a_plan"] * max(alt.item(), 0.0)) ** 0.5)
        else:
            v_ref = 0.0

        fault_onehot = torch.zeros(4, device=self.device)
        if faulted and fault_rotor >= 0:
            fault_onehot[fault_rotor] = 1.0

        s = self.obs_scales
        obs = torch.cat([
            gravity_body,
            torch.clip(ang_vel_body * s["ang_vel"], -5, 5),
            torch.clip(lin_vel_body * s["lin_vel"], -5, 5),
            torch.clip(err_body * s["pos_err"], -1, 1),
            torch.tensor([alt * s["alt"]], device=self.device),
            torch.tensor([max(min(vz * s["lin_vel"], 5), -5)], device=self.device),
            torch.tensor([v_ref * s["lin_vel"]], device=self.device),
            self.last_actions,
            motor_rpm / motor_max_rpm,
            fault_onehot,
        ])
        return obs

    def set_last_actions(self, actions: torch.Tensor):
        self.last_actions = actions.clone()


def lin_vel_body_to_world_z(quat_wxyz: torch.Tensor, lin_vel_body: torch.Tensor) -> torch.Tensor:
    """vz は学習時 (`drone.get_vel()`、世界座標系) と同じ量にする。機体座標系の速度を world に戻して z だけ使う。"""
    world_vel = quat_rotate(quat_wxyz, lin_vel_body)
    return world_vel[2]
