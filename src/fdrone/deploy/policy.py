"""学習の仕組み (rsl-rl・Genesis) に依存しない、推論だけの Actor。

`scripts/export_policy.py` が書き出した .pt を読む。依存は torch のみ (numpy も不要)。
ROS 2 ノード (`ros2_ws/`) など、実機に積む側はこちらを使う。

2 通りの分布に対応する (`data["distribution"]`、無ければ古い形式として "gaussian" 扱い):
- "gaussian" (旧): rsl_rl.models.mlp_model.MLPModel.forward / distribution.GaussianDistribution
  .deterministic_output を確認して確定 (2026-10-18): action = clip(MLP(obs), -1, 1)。
  std_type="scalar" の std が学習中に発散し、det (この式) と学習時
  (サンプリング) で別の制御器になる問題があった
- "beta" (2026-10-19 以降): distribution.BetaDistribution.deterministic_output を確認して
  確定。MLP は [alpha_raw, beta_raw] (2*num_actions 次元) を出し、
  action = (softplus(alpha_raw)+1)/((softplus(alpha_raw)+1)+(softplus(beta_raw)+1)) を
  action_range にスケールする。クリップによるずれは無いが、Beta では学習自体がうまく
  いかなかった
"""
import torch
from torch import nn


_ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}


def _build_mlp(num_obs: int, out_dim: int, hidden_dims: list[int], activation: str) -> nn.Sequential:
    act_cls = _ACTIVATIONS[activation]
    dims = [num_obs, *hidden_dims, out_dim]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(act_cls())
    return nn.Sequential(*layers)


class StandalonePolicy:
    """export_policy.py の .pt を読み、観測ベクトル (1 サンプル) から行動を計算する。"""

    def __init__(self, path: str, device: str = "cpu"):
        data = torch.load(path, map_location=device, weights_only=False)
        self.num_obs = data["num_obs"]
        self.num_actions = data["num_actions"]
        self.obs_scales = data["obs_scales"]
        self.source = data.get("source", {})
        self.distribution = data.get("distribution", "gaussian")
        self.action_range = data.get("action_range", (-1.0, 1.0))
        mlp_out_dim = data.get("mlp_out_dim", self.num_actions)
        self.mlp = _build_mlp(self.num_obs, mlp_out_dim, data["hidden_dims"], data["activation"])
        self.mlp.load_state_dict(data["mlp_state_dict"])
        self.mlp.eval()
        self.device = device

    @torch.no_grad()
    def act(self, obs) -> torch.Tensor:
        """obs: (num_obs,) または (1, num_obs)。戻り値: (num_actions,) の行動 (actuator への変換は呼び出し側)。"""
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        raw = self.mlp(x)
        if self.distribution == "beta":
            raw = raw.view(-1, 2, self.num_actions)
            alpha = torch.nn.functional.softplus(raw[:, 0]) + 1.0
            beta = torch.nn.functional.softplus(raw[:, 1]) + 1.0
            lo, hi = self.action_range
            action = (alpha / (alpha + beta)) * (hi - lo) + lo
        else:
            action = raw.clamp(-1.0, 1.0)
        return action.squeeze(0)
