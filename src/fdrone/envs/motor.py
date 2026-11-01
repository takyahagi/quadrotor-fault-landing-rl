"""モータモデル: 1 次遅れ + 推力重量比に基づく RPM 上限。

Genesis の `drone.set_propellers_rpm()` は指令した RPM を即座に反映する
(モータの応答遅れが無い)。実機の Crazyflie はこの前提では飛べない余裕
(推力重量比) しか無いため、ここで遅れと上限を自前で追加する。

"""
import math
import torch


def _rand_uniform(lo, hi, n, device):
    return (hi - lo) * torch.rand(n, device=device) + lo


class MotorModel:
    """4 枚のプロペラそれぞれに 1 次遅れ `tau` と RPM 上限を課す。

    「目標 RPM (方策や PID の出力)」を受け取り、「実際にプロペラへ渡す RPM」
    を返す。rotor が故障した場合は FaultInjector がこの出力をさらに書き換える
    (motor -> fault の順。モータが壊れているのではなく、プロペラが物理的に
    無くなる想定なので、モータの応答遅れを経由させない)。

    `tw_range` / `tau_range` を渡すと、`reset_idx` のたびに推力重量比・時定数を
    env ごとにランダムに引き直す (B2 ドメインランダム化)。
    `None` のときは引数の固定値のまま (B1 の固定条件評価と同じ挙動)。
    """

    def __init__(
        self,
        num_envs: int,
        n_rotors: int,
        hover_rpm: float,
        thrust_to_weight: float,
        tau: float,
        dt: float,
        device,
        tw_range=None,
        tau_range=None,
    ):
        self.hover_rpm = hover_rpm
        self.dt = dt
        self.device = device
        self.tw_range = tw_range
        self.tau_range = tau_range
        # (num_envs, 1) にして rpm (num_envs, n_rotors) へブロードキャストする
        self.max_rpm = torch.full((num_envs, 1), hover_rpm * math.sqrt(thrust_to_weight), device=device)
        # 離散化した 1 次遅れの更新係数: rpm += alpha * (target - rpm)
        self.alpha = torch.full((num_envs, 1), 1.0 - math.exp(-dt / tau), device=device)
        self.rpm = torch.full((num_envs, n_rotors), hover_rpm, device=device)

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return
        self.rpm[envs_idx] = self.hover_rpm
        n = len(envs_idx)
        if self.tw_range is not None:
            tw = _rand_uniform(*self.tw_range, n, self.device)
            self.max_rpm[envs_idx, 0] = self.hover_rpm * torch.sqrt(tw)
        if self.tau_range is not None:
            tau = _rand_uniform(*self.tau_range, n, self.device)
            self.alpha[envs_idx, 0] = 1.0 - torch.exp(-self.dt / tau)

    def step(self, target_rpm: torch.Tensor) -> torch.Tensor:
        target_rpm = torch.minimum(torch.clamp(target_rpm, min=0.0), self.max_rpm)
        self.rpm = self.rpm + self.alpha * (target_rpm - self.rpm)
        return self.rpm
