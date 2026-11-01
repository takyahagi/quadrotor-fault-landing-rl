"""学習済みポリシーの Actor だけを、rsl-rl / Genesis に依存しない形で書き出す (C3 用)。

実機 (や ROS 2 ノード) に積む推論コードは、学習の仕組み (rsl-rl の Runner・観測正規化・
分布) を知る必要が無い。推論の式は分布の種類 (Gaussian/Beta) で変わるので、どちらかを
メタデータ ("distribution") として一緒に保存する。式の詳細は `src/fdrone/deploy/policy.py`
の docstring。obs_normalizer は Identity なので無視できる。

出力した .pt は `src/fdrone/deploy/policy.py` (plain torch のみ、rsl_rl 不要) で読む。

実行例 (CPU、GPU 不要):
  python scripts/export_policy.py -e stage2_yawpen_td_s1 --ckpt 1499
"""
import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from fdrone.envs.fault_land_env import FaultLandEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", required=True)
    parser.add_argument("--ckpt", type=int, required=True)
    parser.add_argument("-o", "--out", default=None, help="既定: logs/<exp>/policy_<ckpt>.pt")
    args = parser.parse_args()

    gs.init(backend=gs.cpu, logging_level="warning")
    log_dir = f"logs/{args.exp_name}"
    with open(f"{log_dir}/cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, train_cfg = pickle.load(f)

    # ダミーの env (1 env) を 1 つ作って Runner を構築するためだけに使う。学習や物理は回さない。
    env = FaultLandEnv(num_envs=1, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg)
    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cpu")
    runner.load(f"{log_dir}/model_{args.ckpt}.pt", map_location="cpu")

    actor = runner.alg.actor  # MLPModel
    hidden_dims = list(train_cfg["actor"]["hidden_dims"])
    activation = train_cfg["actor"]["activation"]
    num_obs = actor.obs_dim
    num_actions = env.num_actions
    # Gaussian: 最後が Linear(num_actions)。Beta は output_dim が [2, num_actions] のタプルなので
    # rsl_rl の MLP が末尾に nn.Unflatten を足す (パラメータは持たない)。最後の Linear を探す
    mlp_out_dim = next(m for m in reversed(list(actor.mlp)) if isinstance(m, torch.nn.Linear)).out_features
    dist_cfg = train_cfg["actor"]["distribution_cfg"]
    distribution = "beta" if dist_cfg["class_name"] == "BetaDistribution" else "gaussian"
    action_range = tuple(dist_cfg.get("action_range", (-1.0, 1.0)))

    out = Path(args.out) if args.out else Path(log_dir) / f"policy_{args.ckpt}.pt"
    torch.save(
        {
            "mlp_state_dict": actor.mlp.state_dict(),
            "hidden_dims": hidden_dims,
            "activation": activation,
            "num_obs": num_obs,
            "num_actions": num_actions,
            "mlp_out_dim": mlp_out_dim,
            "distribution": distribution,
            "action_range": action_range,
            "obs_scales": obs_cfg["obs_scales"],
            # ROS 2 ノード (ros2_ws/) 側は genesis を入れずに ObservationBuilder (deploy/observation.py)
            # を動かすので、観測の組み立てに要る量もここに含めて 1 ファイルで完結させる
            # (ずれると学習時と違う観測になるので危険。export 時点の env から直接取る)
            "descent_cfg": env_cfg["descent"],
            "motor_max_rpm": float(env.motor.max_rpm[0, 0].item()),
            "source": {"exp_name": args.exp_name, "ckpt": args.ckpt},
        },
        out,
    )
    print(f"[saved] {out} (num_obs={num_obs}, num_actions={num_actions}, hidden_dims={hidden_dims})")


if __name__ == "__main__":
    main()
