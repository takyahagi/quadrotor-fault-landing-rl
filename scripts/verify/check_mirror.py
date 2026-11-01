"""鏡映 (src/fdrone/symmetry.py) が物理と等価かを確かめる (2026-10-26)。

2 つの env を同じ高度・同じ故障時刻で動かし、env 0 はロータ i の故障と行動列 A、env 1 は鏡像のロータ P(i) の故障と
鏡映した行動列 P(A) を入れる。鏡映が正しければ、各時刻で mirror(obs0) == obs1 になる。
2 m/s で x 方向 (向き 0°、鏡映で変わらない) に移動中の故障も含める。ランダム化・観測ノイズ・着地目標のずらしは外す。

  python scripts/verify/check_mirror.py --drone racer
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import torch
import genesis as gs

from fdrone.configs import get_cfgs
from fdrone.envs.fault_land_env import FaultLandEnv
from fdrone.eval_utils import setup
from fdrone.symmetry import MIRROR_PERM, mirror_actions, mirror_obs_tensor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--drone", default="racer")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--speed", type=float, default=2.0)
    p.add_argument("--fault-time", type=float, default=0.2)
    p.add_argument("--gpu", action="store_true")
    args = p.parse_args()
    gs.init(backend=gs.gpu if args.gpu else gs.cpu, precision="32", logging_level="warning", seed=0)
    env_cfg, obs_cfg, reward_cfg = get_cfgs(4, None)
    env_cfg["drone"] = args.drone
    env_cfg["decimation"] = 4
    env_cfg["translate"]["enabled"] = True
    env_cfg["domain_rand"]["enabled"] = False
    env_cfg["landing_offset"]["enabled"] = False
    env_cfg["fault"]["prob"] = 0.0
    env_cfg["auto_reset"] = False
    # (env 0 のロータ, env 1 のロータ) の組。-1 = 故障なし
    pairs = [(0, MIRROR_PERM[0]), (1, MIRROR_PERM[1]), (-1, -1)]
    scen = []
    for a, b in pairs:
        scen += [(10.0, a, args.fault_time if a >= 0 else 0.0), (10.0, b, args.fault_time if b >= 0 else 0.0)]
    env = FaultLandEnv(num_envs=len(scen), env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg)
    env.reset()
    setup(env, scen, speed=args.speed, heading_deg=0.0)
    obs = env.get_observations()["policy"]
    g = torch.Generator(device="cpu").manual_seed(0)
    worst = [0.0] * len(pairs)
    worst_early = [0.0] * len(pairs)
    t_fault = int(round(args.fault_time / env.policy_dt))
    with torch.no_grad():
        for t in range(args.steps):
            a = torch.zeros((len(scen), 4), device=env.device)
            for k in range(len(pairs)):
                ak = (torch.rand(4, generator=g) * 2 - 1).to(env.device)
                a[2 * k] = ak
                a[2 * k + 1] = mirror_actions(ak)
            out, _, _, _ = env.step(a)
            obs = out["policy"]
            for k in range(len(pairs)):
                d = (mirror_obs_tensor(env, obs[2 * k]) - obs[2 * k + 1]).abs().max().item()
                worst[k] = max(worst[k], d)
                if t <= t_fault + 20:
                    worst_early[k] = max(worst_early[k], d)
            if t in (0, 10, 19, 21, 25, 30, 40, 60, 80, 100, args.steps - 1):
                print(f"step {t:4d}: " + "  ".join(f"pair{pairs[k]} max|M(o0)-o1|={worst[k]:.2e}" for k in range(len(pairs))))
    # 機体は回転・転倒して軌道が不安定になるので、浮動小数点の丸め誤差 (1e-9 程度) は時間とともに指数的に増える。
    # 鏡映の表が間違っていれば、故障の直後 (ロータの入れ替えが効く最初のステップ) に O(0.1) のずれが出る。
    # 判定は故障後 20 ステップ (0.2 s) までの最大のずれで行う
    ok = all(w < 1e-3 for w in worst_early)
    print(f"[{'OK' if ok else 'NG'}] 故障後 20 ステップまでの最大のずれ " + ", ".join(f"{pairs[k]}: {worst_early[k]:.2e}" for k in range(len(pairs))))


if __name__ == "__main__":
    main()
