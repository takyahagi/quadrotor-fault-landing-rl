"""故障注入: エピソード中の無作為な時刻に、無作為な 1 枚のプロペラを完全停止させる。

「完全停止のみ」(部分的な効率低下は対象外) は 2026-10-15 の決定事項。学習時はランダムな rotor・時刻で注入し、
検証スクリプトでは `force_fault()` で決定的に注入する。
"""
import torch


class FaultInjector:
    def __init__(
        self,
        num_envs: int,
        n_rotors: int,
        fault_prob: float,
        time_range_s,
        dt: float,
        device,
        rotors=None,
    ):
        self.num_envs = num_envs
        self.n_rotors = n_rotors
        self.fault_prob = fault_prob
        self.time_range_s = time_range_s
        self.dt = dt
        self.device = device
        # 故障させるロータを限る (既定 None = 全ロータから無作為)。racer で鏡像の片側の故障を学ばない局所解に
        # 対し、欠けた側だけで追加学習する検証用
        self.rotors = torch.tensor(list(rotors) if rotors else list(range(n_rotors)), device=device)

        self.has_fault = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.fault_rotor = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.fault_step = torch.zeros(num_envs, dtype=torch.long, device=device)
        # triggered: 故障が「もう発生した」env (fault_step を過ぎた後、エピソード終了まで True のまま)
        self.triggered = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset_idx(self, envs_idx):
        n = len(envs_idx)
        if n == 0:
            return
        has = torch.rand(n, device=self.device) < self.fault_prob
        self.has_fault[envs_idx] = has
        self.fault_rotor[envs_idx] = self.rotors[torch.randint(0, len(self.rotors), (n,), device=self.device)]
        t_min, t_max = self.time_range_s
        step_min, step_max = int(t_min / self.dt), max(int(t_max / self.dt), int(t_min / self.dt))
        self.fault_step[envs_idx] = torch.randint(step_min, step_max + 1, (n,), device=self.device)
        self.triggered[envs_idx] = False

    def force_fault(self, env_idx: int, rotor_idx: int, step: int):
        """検証・デモ用: 特定の env に決定的に故障を仕込む。"""
        self.has_fault[env_idx] = True
        self.fault_rotor[env_idx] = rotor_idx
        self.fault_step[env_idx] = step
        self.triggered[env_idx] = False

    def apply(self, rpm_cmd: torch.Tensor, episode_step: torch.Tensor):
        """rpm_cmd (num_envs, n_rotors) に故障を適用して返す。

        episode_step: (num_envs,) 現在のエピソード内ステップ数。
        戻り値: (適用後の rpm, 現在故障が有効な env のマスク)
        """
        due = self.has_fault & (episode_step >= self.fault_step)
        self.triggered = self.triggered | due
        out = rpm_cmd.clone()
        rows = self.triggered.nonzero(as_tuple=False).squeeze(-1)
        if len(rows) > 0:
            out[rows, self.fault_rotor[rows]] = 0.0
        return out, self.triggered

    def absent_mask(self) -> torch.Tensor:
        """プロペラが無い rotor のマスク (num_envs, n_rotors)。DroneScene.set_propellers_rpm に渡す。"""
        mask = torch.zeros((self.num_envs, self.n_rotors), dtype=torch.bool, device=self.device)
        rows = self.triggered.nonzero(as_tuple=False).squeeze(-1)
        if len(rows) > 0:
            mask[rows, self.fault_rotor[rows]] = True
        return mask
