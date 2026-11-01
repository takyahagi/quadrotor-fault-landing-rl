"""PID ベースラインを、RL ポリシーと同じ固定シナリオ格子・同じ判定基準で評価する (フェーズ A1。
2026-10-20、C4 (移動中の故障・着地目標オフセット) にも対応)。

PID (`src/fdrone/controllers/pid.py`) は常に機体を水平・ヨー角 0 に保とうとするだけで、
故障後にヨー回転を許容する概念を持たない。そのまま破綻するのが期待される挙動。学習済みポリシーとの比較基準として使う。

PID は世界座標系の pos/vel を想定しているが、DroneScene.base_lin_vel は機体座標系
(2026-10-18 に気づいた不整合。P1 の検証ではヨー角 ≈0 のため問題にならなかった)。
ここでは world-frame の `drone.get_vel()` を渡す。

実行例 (C4: 2 m/s で移動中の故障、着地目標オフセットあり、`d4g_s1` と同じ条件):
  python scripts/eval_pid_grid.py --speed 2.0 --heading-deg 0 --landing-offset --decimation 4 --gpu
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

import genesis as gs

from fdrone.envs.fault_land_env import FaultLandEnv
from fdrone.controllers.pid import DronePIDController
from fdrone.configs import get_cfgs
from fdrone.eval_utils import build_scenarios, run, summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=2, help="env_cfg (dt・T/W・空気抵抗など) を揃える Stage")
    parser.add_argument("--speed", type=float, default=0.0, help="C4: 故障前の水平移動速度 [m/s] (既定 0 = 静止)")
    parser.add_argument("--heading-deg", type=float, default=0.0, help="C4: 故障前の移動方向 [deg]")
    parser.add_argument("--landing-offset", action="store_true",
                         help="着地目標を故障地点からランダムにずらす (`d4g_*` と同じ条件)")
    parser.add_argument("--decimation", type=int, default=1,
                         help="PID の指令を何物理ステップ保持するか (既定 1 = 400 Hz、PID にとって最良の条件)。"
                              "RL 側 (`d4g_*` は 4 = 100 Hz) と同じ周期で比較したい場合に指定する。"
                              "eval_utils.run() は 1 ポリシーステップ = decimation 物理ステップごとに policy(obs) を呼ぶため、"
                              "decimation>1 では PID の指令もその間保持される (400 Hz 毎回計算ではなくなる)")
    parser.add_argument("-e", "--out-name", default="pid_eval", help="出力先 logs/<out-name>/")
    parser.add_argument("--tag", default="pid")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu, logging_level="warning", seed=0)
    log_dir = f"logs/{args.out_name}"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    env_cfg, obs_cfg, reward_cfg = get_cfgs(args.stage)
    env_cfg["fault"]["prob"] = 0.0
    env_cfg["decimation"] = args.decimation
    env_cfg["translate"]["enabled"] = args.speed > 0.0
    env_cfg["landing_offset"]["enabled"] = args.landing_offset
    print(f"条件 (Stage {args.stage} の env_cfg): T/W={env_cfg['thrust_to_weight']} "
          f"aero_scale={env_cfg.get('aero_scale', 1.0)} motor_tau={env_cfg['motor_tau']} dt={env_cfg['dt']} "
          f"speed={args.speed} heading_deg={args.heading_deg} landing_offset={args.landing_offset} "
          f"decimation={args.decimation}")

    scenarios = build_scenarios()
    n = len(scenarios)
    env = FaultLandEnv(num_envs=n, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg)
    # eval_utils.run() は 1 ポリシーステップ (= policy_dt = dt*decimation) ごとに pid_policy() を
    # 呼ぶ (env.step() が内部で decimation 物理ステップを同じ指令で進める)。PID の積分・微分項は
    # 実際に呼ばれる間隔 (policy_dt) で計算しないと、decimation>1 のときに利得がずれる
    pid = DronePIDController(num_envs=n, dt=env.policy_dt, base_rpm=env.ds.hover_rpm, device=env.device)

    def pid_policy(_obs):
        pos = env.ds.base_pos
        vel = env.ds.drone.get_vel()  # world frame (PID の前提)
        euler = env.ds.base_euler
        rpm_cmd = pid.update(pos, vel, euler, env.target)
        rpm_cmd = torch.minimum(torch.clamp(rpm_cmd, min=0.0), env.motor.max_rpm)
        frac = (rpm_cmd / env.motor.max_rpm) ** 2
        return torch.clamp(frac * 2.0 - 1.0, -1.0, 1.0)

    rows = run(env, pid_policy, scenarios, speed=args.speed, heading_deg=args.heading_deg,
               extra_reset=lambda: pid.reset_idx(torch.arange(n, device=env.device)))
    summarize("PID", rows)

    out = Path(log_dir) / f"eval_grid_{args.tag}.csv"
    with open(out, "w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
