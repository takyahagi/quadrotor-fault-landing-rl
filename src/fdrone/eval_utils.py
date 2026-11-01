"""固定シナリオ格子での評価の共通処理。

RL ポリシーと PID ベースラインを同じ条件・同じ判定基準で比べるために、
`scripts/eval_grid.py` と `scripts/eval_pid_grid.py` の両方から使う
(§3.2)。
"""
import math
from collections import Counter, defaultdict

import torch

ACTION_MODES = ("det", "stoch", "meanfield")


def make_action_fn(actor, mode: str, n_quantiles: int = 32):
    """rsl-rl の actor (MLPModel + GaussianDistribution) から、評価用の行動関数を作る。

    det:       clip(μ)。従来の評価・ROS 2 ノードと同じ
    stoch:     N(μ, σ) からサンプル。学習時と同じ
    meanfield: ノイズなしで決定的だが、学習時の「ノイズ込みの平均的なモータ指令」を再現する。
               env は rpm 指令 = max_rpm * sqrt((clip(a)+1)/2) を 1 次遅れのモータに入れるので、
               その期待値 E[sqrt((clip(X)+1)/2)] (X~N(μ,σ)、分位点 n_quantiles 個で近似) と
               同じ rpm 指令になる a を返す。σ が大きい方策 向け
    """
    if mode == "det":
        return lambda obs: actor(obs)
    if mode == "stoch":
        return lambda obs: actor(obs, stochastic_output=True)
    if mode != "meanfield":
        raise ValueError(f"unknown action mode {mode}")
    dist = actor.distribution
    if not hasattr(dist, "std_param"):
        # BetaDistribution はクリップによるずれは無いが、sqrt の非線形によるずれは残る
        #。meanfield は Gaussian 用の近似なので使わず、det と stoch を比べる
        raise ValueError("meanfield は GaussianDistribution (std_param) 専用。Beta 分布では det と stoch を使う")
    std = dist.std_param.detach().clamp(dist.std_range[0], dist.std_range[1])  # (num_actions,)
    q = (torch.arange(n_quantiles, dtype=torch.float32) + 0.5) / n_quantiles
    z = torch.distributions.Normal(0.0, 1.0).icdf(q).to(std.device)  # (K,)

    def fn(obs):
        mu = actor(obs)  # (n, A)
        x = (mu.unsqueeze(-1) + std.unsqueeze(-1) * z).clamp(-1.0, 1.0)  # (n, A, K)
        s = torch.sqrt((x + 1.0) * 0.5).mean(dim=-1)
        return 2.0 * s**2 - 1.0

    return fn

ALTS = [2.0, 5.0, 10.0, 15.0, 20.0]
ROTORS = [0, 1, 2, 3]
FAULT_TIMES = [1.0, 1.5, 2.0, 2.5, 3.0]


def build_scenarios(alts=None):
    """alts を指定すると、既定の高度 (ALTS) の代わりに使う (B3: 細かい高度の掃引用)。"""
    alts = alts if alts is not None else ALTS
    sc = [(a, r, t) for a in alts for r in ROTORS for t in FAULT_TIMES]
    sc += [(a, -1, 0.0) for a in alts]  # 故障なし (rotor=-1)
    return sc


def setup(env, scenarios, speed=0.0, heading_deg=0.0):
    """speed>0 のとき、故障前に水平方向へ `speed` [m/s]・`heading_deg` 度の方向で動かした状態で
    開始する (C4)。既定の speed=0.0 では以前と全く同じ (静止ホバリング) 挙動。

    注意: 以前は `FaultLandEnv.reset_idx()` を経由しないため、`translate` が有効な設定で学習した
    方策でも、ここでは常に速度ゼロで評価してしまっていた (c4_s1 の無効評価)。
    """
    dev = env.device
    n = len(scenarios)
    alts = torch.tensor([s[0] for s in scenarios], device=dev)
    env.init_alt[:] = alts
    env.ds.base_init_pos[:] = torch.stack([torch.zeros(n, device=dev), torch.zeros(n, device=dev), alts], dim=1)
    all_idx = torch.arange(n, device=dev)
    env.ds.reset_idx(all_idx)
    env.motor.reset_idx(all_idx)
    env.target[:] = env.ds.base_init_pos
    env.faulted[:] = False
    env.last_actions[:] = 0.0
    env.episode_length_buf[:] = 0
    env.v_cmd[:] = 0.0
    if speed > 0.0:
        heading = math.radians(heading_deg)
        env.v_cmd[:, 0] = speed * math.cos(heading)
        env.v_cmd[:, 1] = speed * math.sin(heading)
        vel6 = torch.zeros((n, 6), device=dev)
        vel6[:, :2] = env.v_cmd[:, :2]
        env.ds.drone.set_dofs_velocity(vel6, envs_idx=all_idx)
    f = env.fault
    f.has_fault[:] = torch.tensor([s[1] >= 0 for s in scenarios], device=dev)
    f.fault_rotor[:] = torch.tensor([max(s[1], 0) for s in scenarios], device=dev)
    f.fault_step[:] = torch.tensor([int(s[2] / env.dt) for s in scenarios], device=dev)
    f.triggered[:] = False
    env._compute_state()


def classify(env, i):
    if bool(env.success[i]):
        return "success"
    if bool(env.landed_fail[i]):
        v = env.prev_world_vel[i]
        if env.prev_tilt[i] >= env.env_cfg["success"]["max_tilt_deg"]:
            return "land_fail_tilt"
        if -v[2] >= env.env_cfg["success"]["max_vz"]:
            return "land_fail_vz"
        return "land_fail_vxy"
    if bool(env.crashed[i]):
        return "out_of_bounds" if bool(env.out_of_bounds[i]) else "crash"
    if bool(env.ground_unfaulted[i]):
        return "ground_nofault"
    return "timeout"


def run(env, policy, scenarios, extra_reset=None, speed=0.0, heading_deg=0.0):
    """policy(obs) -> actions (num_envs, 4) を、全シナリオが終わるまで動かす。

    extra_reset: setup() の直後に呼ぶ追加のリセット (PID の積分項など)。
    speed/heading_deg: C4 の評価用。setup() にそのまま渡す。
    """
    n = len(scenarios)
    env.reset()
    setup(env, scenarios, speed=speed, heading_deg=heading_deg)
    if extra_reset is not None:
        extra_reset()
    obs = env.get_observations()
    done_once = torch.zeros(n, dtype=torch.bool, device=env.device)
    max_sink = torch.zeros(n, device=env.device)
    max_yaw = torch.zeros(n, device=env.device)
    rows = [None] * n
    with torch.no_grad():
        for step in range(env.max_episode_length + 1):
            obs, _, dones, _ = env.step(policy(obs))
            live = ~done_once
            sink = torch.where(env.faulted, -env.vz, torch.zeros_like(env.vz))
            max_sink = torch.where(live, torch.maximum(max_sink, sink), max_sink)
            max_yaw = torch.where(live, torch.maximum(max_yaw, env.ds.base_ang_vel[:, 2].abs()), max_yaw)
            for i in (dones & live).nonzero(as_tuple=False).reshape(-1).tolist():
                v = env.prev_world_vel[i]
                a, r, t = scenarios[i]
                rows[i] = {
                    "alt": a, "rotor": r, "fault_time": t, "outcome": classify(env, i),
                    "t_end": round(step * env.policy_dt, 3),
                    "vz": round(v[2].item(), 3), "vxy": round(torch.norm(v[:2]).item(), 3),
                    "tilt": round(env.prev_tilt[i].item(), 1),
                    "max_sink": round(max_sink[i].item(), 2), "max_yaw_rate": round(max_yaw[i].item(), 1),
                }
            done_once |= dones
            if bool(done_once.all()):
                break
    return rows


# 失敗の重大度。成功率の差が小さいときは、こちらで判断する (2026-10-18 ユーザー指示)。
# 重大: 転倒・範囲外逸脱 (制御を失っている)。軽微: 接地はできたが閾値を少し超えた
SEVERE_OUTCOMES = {"crash", "out_of_bounds"}
MINOR_FAIL_OUTCOMES = {"land_fail_vxy", "land_fail_vz", "land_fail_tilt"}


def severity(outcome: str) -> str:
    if outcome == "success":
        return "success"
    if outcome in SEVERE_OUTCOMES:
        return "severe"
    if outcome in MINOR_FAIL_OUTCOMES:
        return "minor"
    return "other"  # ground_nofault, timeout など


def summarize(tag, rows):
    fault = [r for r in rows if r["rotor"] >= 0]
    nominal = [r for r in rows if r["rotor"] < 0]
    n = len(fault)
    sr = sum(r["outcome"] == "success" for r in fault) / n
    sev = Counter(severity(r["outcome"]) for r in fault)
    severe_rate = sev.get("severe", 0) / n
    minor_rate = sev.get("minor", 0) / n
    by_alt = defaultdict(list)
    for r in fault:
        by_alt[r["alt"]].append(r["outcome"] == "success")
    alt_str = " ".join(f"{a:g}m:{sum(v) / len(v):.0%}" for a, v in sorted(by_alt.items()))
    kinds = Counter(r["outcome"] for r in fault)
    nom_ok = sum(r["outcome"] == "timeout" for r in nominal) / len(nominal)
    print(
        f"{tag}: 故障時の成功率 {sr:.0%} [{alt_str}] "
        f"重大な失敗(転倒/範囲外) {severe_rate:.0%} 軽微な失敗(閾値超過) {minor_rate:.0%} "
        f"故障なしで飛び続けた {nom_ok:.0%} 内訳 {dict(kinds)}"
    )
