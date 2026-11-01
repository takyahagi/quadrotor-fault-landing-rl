"""高度ごとの「安全な着陸地点まで届く範囲」の表を作る (2026-10-20)。

将来の階層構成 (上位: 安全な着陸地点を選ぶ / 下位: 今の方策) のうち、上位が候補地点を選ぶ際に
「この高度・この距離なら下位の方策が本当に着地できるか」を知っておく必要がある。cf2x の最終モデルは
`landing_offset` (故障地点から無作為な方向・距離 (学習時は最大 8 m) にずらした地点へ降りる) を
学習済みなので、距離を固定して高度×距離の格子で成功率を測る。

`env_cfg["landing_offset"]["enabled"]=False` にして env 内部の無作為オフセットを止め、故障が
発動した瞬間 (env が target を「その場の真下」に設定した直後) に、このスクリプト側で望みの距離・
方向のオフセットを足す (env 本体は変更していない)。

実行例:
  python scripts/eval_reach_range.py -e d4g_s1 --ckpt 1499 --gpu
  python scripts/eval_reach_range.py -e d4g_c2_s1 --ckpt 1499 --stage 4 --gpu
"""
import argparse
import csv
import math
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from fdrone.envs.fault_land_env import FaultLandEnv
from fdrone.eval_utils import ACTION_MODES, classify, make_action_fn, severity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", required=True)
    parser.add_argument("--ckpt", type=int, required=True)
    parser.add_argument("--alts", default="2,5,10,15,20")
    parser.add_argument("--dists", default="0,2,4,6,8,10,12,15",
                         help="着陸目標までの距離 [m]。学習時の範囲は最大 8 m (landing_offset.max_dist)。"
                              "8 m 超は未知の条件 (B1 相当)")
    parser.add_argument("--n-repeat", type=int, default=12, help="セルごとの繰り返し (rotor・故障時刻・方向を無作為化)")
    parser.add_argument("--speed", type=float, default=0.0, help="C4: 故障前の水平移動速度 [m/s] (既定 0 = 静止)")
    parser.add_argument("--heading-deg", type=float, default=0.0)
    parser.add_argument("--action-mode", choices=ACTION_MODES, default="stoch")
    parser.add_argument("--tag", default="")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, logging_level="warning", seed=0)
    log_dir = f"logs/{args.exp_name}"
    with open(f"{log_dir}/cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, train_cfg = pickle.load(f)
    env_cfg["fault"]["prob"] = 0.0
    env_cfg.setdefault("domain_rand", {})["enabled"] = False
    env_cfg["translate"]["enabled"] = args.speed > 0.0
    # env 内部の無作為オフセットは止め、このスクリプトが固定距離を与える
    env_cfg["landing_offset"]["enabled"] = False
    # target/base_pos は _LATCHED に入っていないため、終了した env から先に auto_reset される
    # (既定 True) と、position_at_end の計算が「リセット後の新しいエピソードの値」になってしまう
    # (target≈base_pos で誤って 0 m 付近に見える、2026-10-20、dist_to_target_at_end 追加時に発覚)
    env_cfg["auto_reset"] = False

    alts = [float(a) for a in args.alts.split(",")]
    dists = [float(d) for d in args.dists.split(",")]
    n = len(alts) * len(dists) * args.n_repeat
    cells = [(a, d) for a in alts for d in dists for _ in range(args.n_repeat)]

    env = FaultLandEnv(num_envs=n, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg)
    dev = env.device
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{args.ckpt}.pt"), map_location=str(gs.device))
    actor = runner.alg.actor
    actor.eval()
    torch.manual_seed(0)
    policy = make_action_fn(actor, args.action_mode)

    cell_alt = torch.tensor([c[0] for c in cells], device=dev)
    cell_dist = torch.tensor([c[1] for c in cells], device=dev)
    rotor = torch.randint(0, 4, (n,), device=dev)
    fault_time = 1.0 + 2.0 * torch.rand(n, device=dev)  # 1.0〜3.0 s (build_scenarios の FAULT_TIMES と同じ範囲)
    offset_heading = 2 * math.pi * torch.rand(n, device=dev)

    env.reset()
    env.init_alt[:] = cell_alt
    env.ds.base_init_pos[:] = torch.stack(
        [torch.zeros(n, device=dev), torch.zeros(n, device=dev), cell_alt], dim=1
    )
    all_idx = torch.arange(n, device=dev)
    env.ds.reset_idx(all_idx)
    env.motor.reset_idx(all_idx)
    env.target[:] = env.ds.base_init_pos
    env.faulted[:] = False
    env.last_actions[:] = 0.0
    env.episode_length_buf[:] = 0
    env.v_cmd[:] = 0.0
    if args.speed > 0.0:
        heading = math.radians(args.heading_deg)
        env.v_cmd[:, 0] = args.speed * math.cos(heading)
        env.v_cmd[:, 1] = args.speed * math.sin(heading)
        vel6 = torch.zeros((n, 6), device=dev)
        vel6[:, :2] = env.v_cmd[:, :2]
        env.ds.drone.set_dofs_velocity(vel6, envs_idx=all_idx)
    f = env.fault
    f.has_fault[:] = True
    f.fault_rotor[:] = rotor
    f.fault_step[:] = (fault_time / env.dt).long()
    f.triggered[:] = False
    env._compute_state()

    offset_vec = torch.stack([cell_dist * torch.cos(offset_heading), cell_dist * torch.sin(offset_heading)], dim=1)

    obs = env.get_observations()
    prev_faulted = env.faulted.clone()
    done_once = torch.zeros(n, dtype=torch.bool, device=dev)
    rows = [None] * n
    with torch.no_grad():
        for step in range(env.max_episode_length + 1):
            obs, _, dones, _ = env.step(policy(obs))
            newly = env.faulted & ~prev_faulted
            if newly.any():
                env.target[newly, :2] += offset_vec[newly]
            prev_faulted = env.faulted.clone()
            live = ~done_once
            for i in (dones & live).nonzero(as_tuple=False).reshape(-1).tolist():
                v = env.prev_world_vel[i]
                # 着地時に本当に「ずらした目標」の近くまで来ているか (classify() は着地の柔らかさしか
                # 見ないので、遠くの目標に届いていなくても「成功」になってしまう可能性を確かめる、
                # 2026-10-20に発覚)
                dist_to_target = torch.norm(env.target[i, :2] - env.ds.base_pos[i, :2]).item()
                rows[i] = {
                    "alt": cell_alt[i].item(), "dist": cell_dist[i].item(), "rotor": rotor[i].item(),
                    "fault_time": round(fault_time[i].item(), 2), "outcome": classify(env, i),
                    "t_end": round(step * env.policy_dt, 3),
                    "vz": round(v[2].item(), 3), "vxy": round(torch.norm(v[:2]).item(), 3),
                    "tilt": round(env.prev_tilt[i].item(), 1),
                    "dist_to_target_at_end": round(dist_to_target, 2),
                }
            done_once |= dones
            if bool(done_once.all()):
                break

    out = Path(log_dir) / (f"eval_reach_range_{args.tag}.csv" if args.tag else "eval_reach_range.csv")
    with open(out, "w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[saved] {out}")

    by_cell = defaultdict(list)
    for r in rows:
        by_cell[(r["alt"], r["dist"])].append(r)
    print(f"\n成功率の表 (行=高度 m、列=距離 m、{args.n_repeat} 回平均、action_mode={args.action_mode}):")
    header = "alt\\dist".ljust(10) + "".join(f"{d:>7.1f}" for d in dists)
    print(header)
    for a in alts:
        line = f"{a:<10.1f}"
        for d in dists:
            cell = by_cell[(a, d)]
            sr = sum(r["outcome"] == "success" for r in cell) / len(cell)
            line += f"{sr:>6.0%} "
        print(line)
    print(f"\n重大な失敗 (crash/out_of_bounds) 率の表:")
    print(header)
    for a in alts:
        line = f"{a:<10.1f}"
        for d in dists:
            cell = by_cell[(a, d)]
            sev = sum(severity(r["outcome"]) == "severe" for r in cell) / len(cell)
            line += f"{sev:>6.0%} "
        print(line)
    print(f"\n着地時に目標までどれだけ残っていたか (success のみ、平均 [m]。遠くの目標に"
          f"届かず近くで「安全に着地」しただけの可能性を見る):")
    print(header)
    for a in alts:
        line = f"{a:<10.1f}"
        for d in dists:
            cell = [r for r in by_cell[(a, d)] if r["outcome"] == "success"]
            if not cell:
                line += f"{'(n/a)':>7} "
                continue
            avg = sum(r["dist_to_target_at_end"] for r in cell) / len(cell)
            line += f"{avg:>6.1f} "
        print(line)


if __name__ == "__main__":
    main()
