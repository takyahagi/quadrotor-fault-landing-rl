"""固定した評価条件の一式で、チェックポイントごとの成功率と失敗パターンを出す。

強化学習には固定のデータセットも検証データも無いので、代わりに:
  - 評価条件を固定する (高度 × 故障する rotor × 故障時刻、加えて故障なし)。乱数は使わない
  - 方策は探索ノイズなし (平均の行動) で動かす
  - 学習に使っていない条件 (--tw 2.25、--aero-scale 0.5/2.0 など) を「未知の検証データ」にする
  - 100 iteration ごとのチェックポイントを同じ条件で動かし、成功率が伸びているか (学習不足か) を見る
全条件を 1 回で並列に動かす (GPU)。シナリオ生成・判定基準は `src/fdrone/eval_utils.py` にあり、
PID ベースラインの評価 (`scripts/eval_pid_grid.py`) と共通。

出力: logs/<exp>/eval_grid[_<tag>].csv (チェックポイントごとの全エピソード)、標準出力に要約

実行例:
  python scripts/eval_grid.py -e stage2 --ckpts 100,500,1000,1500
  python scripts/eval_grid.py -e stage2 --ckpts 1500 --tw 2.25 --tag tw225
"""
import argparse
import csv
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from fdrone.envs.fault_land_env import FaultLandEnv
from fdrone.eval_utils import ACTION_MODES, build_scenarios, make_action_fn, run, summarize


def _mirror_wrap(env, policy, scenarios, rotors_str):
    from tensordict import TensorDict
    from fdrone.symmetry import mirror_actions, mirror_obs_tensor
    rotors = {int(r) for r in rotors_str.split(",")}
    mask = torch.tensor([s[1] in rotors for s in scenarios], device=env.device).unsqueeze(1)

    def fn(obs):
        o = obs["policy"]
        o2 = torch.where(mask, mirror_obs_tensor(env, o), o)
        a = policy(TensorDict({"policy": o2}, batch_size=obs.batch_size, device=o.device))
        return torch.where(mask, mirror_actions(a), a)
    return fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", required=True)
    parser.add_argument("--ckpts", required=True, help="例: 100,500,1500")
    parser.add_argument("--tw", type=float, default=None, help="推力重量比を上書き (未知条件)")
    parser.add_argument("--aero-scale", type=float, default=None, help="空気抵抗係数の倍率 (未知条件)")
    parser.add_argument("--motor-tau", type=float, default=None, help="モータ時定数を上書き (未知条件)")
    parser.add_argument("--speed", type=float, default=0.0, help="C4: 故障前の水平移動速度 [m/s] (既定 0 = 静止)")
    parser.add_argument("--heading-deg", type=float, default=0.0, help="C4: 故障前の移動方向 [deg]")
    parser.add_argument("--action-mode", choices=ACTION_MODES, default="det",
                        help="det: clip(μ) / stoch: 学習時と同じサンプリング / meanfield: ノイズ込みの平均指令")
    parser.add_argument("--alts", default=None, help="B3: 高度を既定の [2,5,10,15,20] の代わりに指定 (例: 1,1.5,2,2.5,3,3.5,4,4.5,5)")
    parser.add_argument("--drone", choices=["cf2x", "racer", "pelican"], default=None,
                        help="機体を上書き (既定: 学習時の機体)。別の機体で学習した方策をそのまま動かす検証用 "
                             "。観測・行動の次元は機体によらず同じ")
    parser.add_argument("--eval-seeds", default="0",
                        help="stoch の評価の乱数の種 (カンマ区切りで複数)。1 回の stoch 評価は 10〜20 m の 60 条件で "
                             "±12 pt ばらつくので、比較には 5 通り程度を使う")
    parser.add_argument("--speeds", default=None, help="--speed の代わりに複数の速度をカンマ区切りで (例 0,2,4)")
    parser.add_argument("--headings", default=None, help="--heading-deg の代わりに複数の向きをカンマ区切りで (例 0,90,180,270)")
    parser.add_argument("--nominal-only", action="store_true",
                        help="故障なしの条件だけを動かす (通常飛行の評価を増やす)")
    parser.add_argument("--obs-noise", type=float, default=None,
                        help="観測ノイズの std を上書き (ドメインランダム化の観測ノイズの項目だけの感度を見る)")
    parser.add_argument("--mirror-rotors", default=None,
                        help="この番号のロータが故障するシナリオだけ、観測を鏡映して方策に入れ、出た行動を鏡映し直す "
                             "(例 1,3)。racer の鏡像の片側を学んでいない方策が、学んだ側の鏡像で解けるかの検証。"
                             "評価側が故障の側を教えているので、方策の性能としては扱わない")
    parser.add_argument("--lin-drag-scale", type=float, default=None,
                        help="並進抵抗・H-force のヨー減衰の係数の倍率 (未知条件。形状抵抗には掛けない)。racer の係数は cf2x の流用で、"
                             "プロペラの大きさからは 2 桁小さい可能性がある")
    parser.add_argument("--form-drag-k", default=None,
                        help="機体の形状抵抗の係数を上書き (数値、または est = DRONE_PRESETS の機体ごとの見積もり)。"
                             "学習時は全機体 2.6e-8 (cf2x の見積もり)")
    parser.add_argument("--tag", default="")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, logging_level="warning", seed=0)
    log_dir = f"logs/{args.exp_name}"
    with open(f"{log_dir}/cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, train_cfg = pickle.load(f)
    env_cfg["fault"]["prob"] = 0.0
    # B2 で学習した exp は domain_rand.enabled=True のまま保存されている。評価は常に
    # --tw/--aero-scale/--motor-tau で指定した固定条件で動かす (ランダム化を無効にする)
    env_cfg.setdefault("domain_rand", {})["enabled"] = False
    if args.drone is not None:
        env_cfg["drone"] = args.drone
    if args.form_drag_k is not None:
        from fdrone.configs import DRONE_PRESETS
        k = DRONE_PRESETS[env_cfg.get("drone", "cf2x")]["form_drag_k_est"] if args.form_drag_k == "est" else float(args.form_drag_k)
        env_cfg["form_drag_k"] = k
        print(f"形状抵抗の係数 k={k:.3g}")
    if args.lin_drag_scale is not None:
        env_cfg["lin_drag_scale"] = args.lin_drag_scale
        print(f"並進抵抗・ヨー減衰の係数 ×{args.lin_drag_scale}")
    for key, val in (("thrust_to_weight", args.tw), ("aero_scale", args.aero_scale), ("motor_tau", args.motor_tau)):
        if val is not None:
            env_cfg[key] = val
    print(f"条件: drone={env_cfg.get('drone', 'cf2x')} T/W={env_cfg['thrust_to_weight']} aero_scale={env_cfg.get('aero_scale', 1.0)} "
          f"motor_tau={env_cfg['motor_tau']} speed={args.speed} heading_deg={args.heading_deg} "
          f"action_mode={args.action_mode}")

    alts = [float(a) for a in args.alts.split(",")] if args.alts else None
    scenarios = build_scenarios(alts)
    if args.nominal_only:
        scenarios = [s for s in scenarios if s[1] < 0]
    env = FaultLandEnv(num_envs=len(scenarios), env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg)
    if args.obs_noise is not None:
        env.obs_noise_std = args.obs_noise
        print(f"観測ノイズ std={args.obs_noise}")
    speeds = [float(v) for v in args.speeds.split(",")] if args.speeds else [args.speed]
    headings = [float(v) for v in args.headings.split(",")] if args.headings else [args.heading_deg]
    seeds = [int(v) for v in args.eval_seeds.split(",")]
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    out = Path(log_dir) / (f"eval_grid_{args.tag}.csv" if args.tag else "eval_grid.csv")
    with open(out, "w", newline="") as fcsv:
        writer = None
        for ckpt in [int(c) for c in args.ckpts.split(",")]:
            runner.load(os.path.join(log_dir, f"model_{ckpt}.pt"), map_location=str(gs.device))
            actor = runner.alg.actor
            actor.eval()
            for spd in speeds:
                for hd in headings:
                    for es in seeds:
                        torch.manual_seed(es)  # stoch の再現性
                        policy = make_action_fn(actor, args.action_mode)
                        if args.mirror_rotors:
                            policy = _mirror_wrap(env, policy, scenarios, args.mirror_rotors)
                        rows = run(env, policy, scenarios, speed=spd, heading_deg=hd)
                        if args.nominal_only:
                            ok = sum(r["outcome"] == "timeout" for r in rows)
                            print(f"ckpt {ckpt:5d} speed={spd:g} heading={hd:g} seed={es}: 故障なしで飛び続けた "
                                  f"{ok}/{len(rows)} 内訳 {[r['outcome'] for r in rows]}")
                        else:
                            summarize(f"ckpt {ckpt:5d} speed={spd:g} heading={hd:g} seed={es}", rows)
                        for r in rows:
                            r = {"ckpt": ckpt, **r, "eval_seed": es, "speed": spd, "heading": hd}
                            if writer is None:
                                writer = csv.DictWriter(fcsv, fieldnames=list(r.keys()))
                                writer.writeheader()
                            writer.writerow(r)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
