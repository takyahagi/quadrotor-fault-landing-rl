"""PPO 学習 (rsl-rl)。Genesis examples/drone/hover_train.py (Apache-2.0) をベースに、
FaultLandEnv (モータモデル・空気力学・故障注入つき) へ差し替えた。

  Stage 1: 通常ホバリングのみ (故障なし)
  Stage 2: 70% のエピソードで 1 枚故障 → 軟着陸

GPU 前提。実行例:
  python scripts/train.py --stage 2 -e stage2 -b 4096 --max-iterations 1500
"""
import argparse
import os
import pickle
import shutil
import sys
from importlib import metadata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError, ValueError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e
from rsl_rl.runners import OnPolicyRunner

import torch
import genesis as gs

from fdrone.configs import get_cfgs, get_train_cfg
from fdrone.envs.fault_land_env import FaultLandEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--variant", type=str, default=None, help="設定の派生 (例: yawpen)")
    parser.add_argument("--drone", choices=["cf2x", "racer", "pelican"], default="cf2x",
                         help="機体プリセット (既定 cf2x = Crazyflie、27g)。racer = gym-pybullet-drones "
                              "のレーシングドローン (830g)。pelican = AscTec Pelican (RotorS 公開値、1.0kg、"
                              "+ 配置、2026-10-20 ユーザー指示: マイクロドローン以外の規模で試す)")
    parser.add_argument("--tw", type=float, default=None,
                         help="thrust_to_weight を上書き (既定 2.5)。T/W は無次元だが、同じ T/W でも"
                              "アーム長が長い機体は角加速度の余裕が小さくなる (概ね g/arm に比例、"
                              "2026-10-20 の推測)。大きい機体で学習が進まないときに上げてみる")
    parser.add_argument("-e", "--exp-name", type=str, default=None)
    parser.add_argument("-b", "--num-envs", type=int, default=4096)
    parser.add_argument("--max-iterations", type=int, default=301)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--translate", action="store_true",
                         help="C4: 故障前に水平移動させる (speed_range からサンプルした速度・無作為な向き)")
    parser.add_argument("--dr-tw-range", type=float, nargs=2, default=None, help="ランダム化の推力重量比の範囲を上書き (既定 2.25 2.5)")
    parser.add_argument("--dr-tau-range", type=float, nargs=2, default=None, help="ランダム化のモータ時定数の範囲を上書き (既定 0.05 0.1)")
    parser.add_argument("--dr-mass-range", type=float, nargs=2, default=None, help="ランダム化の質量の倍率の範囲を上書き (既定 0.9 1.1)")
    parser.add_argument("--dr-aero-range", type=float, nargs=2, default=None, help="ランダム化の空気抵抗の倍率の範囲を上書き (既定 0.5 2.0)")
    parser.add_argument("--dr-obs-noise", type=float, default=None,
                         help="ランダム化の観測ノイズの std を上書き (既定 0.03)。racer の既にある解を壊さない範囲に狭める "
                              "(時定数 0.1 s で 17%%、質量 ±10%% は実効 T/W を下げる)")
    parser.add_argument("--lin-drag-scale", type=float, default=None,
                         help="並進抵抗・H-force のヨー減衰の係数の倍率 (既定 1)。racer の係数は cf2x の流用")
    parser.add_argument("--form-drag-k", default=None,
                         help="形状抵抗の係数 (数値、または est = DRONE_PRESETS の機体ごとの見積もり、既定 2.6e-8)")
    parser.add_argument("--symmetry", action="store_true",
                         help="機体の左右対称 (y 軸の鏡映) で学習データを増やす (rsl-rl の Symmetry 拡張、"
                              "src/fdrone/symmetry.py)。racer が鏡像の片側の故障だけを学ぶ局所解への対策")
    parser.add_argument("--fault-rotors", type=str, default=None,
                         help="学習で故障させるロータをカンマ区切りで限る (既定: 全ロータ)。鏡像の片側の故障を学ばない局所解の検証用 "
                              "")
    parser.add_argument("--domain-rand", action="store_true",
                         help="B2: T/W・空気抵抗・モータ時定数・質量・観測ノイズを env ごとランダム化する")
    parser.add_argument("--landing-offset", action="store_true",
                         help="将来の安全着陸の準備: 故障地点の真下ではなく無作為にずらした地点を着陸目標にする")
    parser.add_argument("--decimation", type=int, default=1,
                         help="方策の 1 ステップあたりの物理ステップ数 (4 = 方策 100 Hz、8 = 50 Hz。物理は 400 Hz のまま)")
    parser.add_argument("--smooth-scale", type=float, default=None,
                         help="行動の変化の罰則 (smooth) の重みを上書き (既定 -0.01)。停止・全開の切り替えを抑える試み")
    parser.add_argument("--beta", action="store_true",
                         help="行動分布を Beta にする (既定は Gaussian。Beta は学習できなかった)")
    parser.add_argument("--land-fail-mode", choices=["linear", "floor", "asymptote"], default="linear",
                         help="land_fail の終端罰則の式。linear: 既定 (不採用になった "
                              "floor を戻した形)。floor: 衝突速度の罰則に crash+1 の下限を設ける (旧案)。"
                              "asymptote: crash に漸近するが勾配は飽和しない (2026-10-20)")
    parser.add_argument("--landing-pos-penalty", type=float, default=0.0,
                         help="接地時の着地目標までの残距離 [m] に比例する終端罰則の重み (既定 0.0 ="
                              "無効)。descent_xy/descent_xy_far だけでは着地目標に接近する能力が"
                              "全く学習されなかった ことへの対策 (6h 節)。"
                              "単独では効果なし")
    parser.add_argument("--descent-xy-far-scale", type=float, default=5.0,
                         help="descent_xy_far のガウシアンの幅 [m] (既定 5.0)。landing_offset.max_dist "
                              "(既定 8.0) まで勾配を残す狙いで広げる")
    parser.add_argument("--descent-xy-far-weight", type=float, default=None,
                         help="descent_xy_far の重みを上書き (既定 0.2)")
    parser.add_argument("--yaw-rate-cont-weight", type=float, default=0.0,
                         help="ヨー角速度を故障直後から連続的に罰する重み (既定 0.0 = 無効)。"
                              "racer の暴走対策 (2026-10-21)")
    parser.add_argument("--climb-weight", type=float, default=0.0,
                         help="上昇速度 (vz>0) を故障直後から連続的に罰する重み (既定 0.0 = 無効)")
    parser.add_argument("--speed-mag-weight", type=float, default=0.0,
                         help="水平速度の大きさを故障直後から連続的に罰する重み (既定 0.0 = 無効)")
    parser.add_argument("--action-sat-weight", type=float, default=0.0,
                         help="行動が ±1 付近に張り付く (飽和) ことを故障直後に罰する重み (既定 0.0 = 無効)")
    parser.add_argument("--gamma", type=float, default=None,
                         help="PPO の割引率を方策 1 ステップあたりの値で上書き (既定 0.9975**decimation、"
                              "100 Hz で 0.990 = 見通し約 1 s)。racer は回転を育てるのに約 3 s かかるため "
                              "見通しを伸ばす")
    parser.add_argument("--init-alt-min", type=float, default=None, help="初期高度の下限 [m] を上書き (既定 2.0)")
    parser.add_argument("--init-alt-max", type=float, default=None, help="初期高度の上限 [m] を上書き (既定 20.0)")
    parser.add_argument("--episode-s", type=float, default=None, help="エピソードの長さ [s] を上書き (既定 25.0)")
    parser.add_argument("--std-max", type=float, default=None,
                         help="Gaussian の行動の std に上限を付ける (既定 なし)。racer A2c4 は std≈54 で行動の 99.6%% が"
                              "全開/停止に張り付き、ノイズを抜くと故障後は 92〜95%% 成功するが通常飛行ができなかった"
                              "。学習済みの方策から --resume-from と組み合わせて std を下げて仕上げる")
    parser.add_argument("--reset-std", type=float, default=None,
                         help="--resume-from で読み込んだ後、std をこの値に置き直す (--std-max 以下にする)")
    parser.add_argument("--resume-from", type=str, default=None,
                         help="続きから学習するチェックポイント (例 logs/d4g_racer_A2c4_s1/model_1499.pt)。"
                              "方策・価値関数・最適化器の状態と iteration 番号を引き継ぎ、-e の新しいフォルダに保存する。"
                              "環境の設定は今回の引数で作り直すので、元と同じ引数を渡す")
    parser.add_argument("--cpu", action="store_true", help="CPU backend で動かす (開発機での動作確認用、低速)")
    args = parser.parse_args()
    exp_name = args.exp_name or f"stage{args.stage}"

    backend = gs.cpu if args.cpu else gs.gpu
    gs.init(backend=backend, precision="32", logging_level="warning", seed=args.seed, performance_mode=True)

    log_dir = f"logs/{exp_name}"
    env_cfg, obs_cfg, reward_cfg = get_cfgs(args.stage, args.variant)
    env_cfg["drone"] = args.drone
    if args.tw is not None:
        env_cfg["thrust_to_weight"] = args.tw
    if args.translate:
        env_cfg["translate"]["enabled"] = True
    dr = env_cfg["domain_rand"]
    for key, val in (("thrust_to_weight_range", args.dr_tw_range), ("motor_tau_range", args.dr_tau_range),
                     ("mass_scale_range", args.dr_mass_range), ("aero_scale_range", args.dr_aero_range)):
        if val is not None:
            dr[key] = tuple(val)
    if args.dr_obs_noise is not None:
        dr["obs_noise_std"] = args.dr_obs_noise
    if args.lin_drag_scale is not None:
        env_cfg["lin_drag_scale"] = args.lin_drag_scale
    if args.form_drag_k is not None:
        from fdrone.configs import DRONE_PRESETS
        env_cfg["form_drag_k"] = (DRONE_PRESETS[args.drone]["form_drag_k_est"] if args.form_drag_k == "est"
                                  else float(args.form_drag_k))
    if args.fault_rotors:
        env_cfg["fault"]["rotors"] = [int(r) for r in args.fault_rotors.split(",")]
    if args.domain_rand:
        env_cfg["domain_rand"]["enabled"] = True
    if args.landing_offset:
        env_cfg["landing_offset"]["enabled"] = True
    if args.smooth_scale is not None:
        reward_cfg["reward_scales"]["smooth"] = args.smooth_scale
    reward_cfg["terminal"]["land_fail_mode"] = args.land_fail_mode
    reward_cfg["terminal"]["landing_pos_penalty"] = args.landing_pos_penalty
    env_cfg["descent_xy_far_scale"] = args.descent_xy_far_scale
    if args.descent_xy_far_weight is not None:
        reward_cfg["reward_scales"]["descent_xy_far"] = args.descent_xy_far_weight
    reward_cfg["reward_scales"]["descent_yaw_rate_cont"] = args.yaw_rate_cont_weight
    reward_cfg["reward_scales"]["descent_climb"] = args.climb_weight
    reward_cfg["reward_scales"]["descent_speed_mag"] = args.speed_mag_weight
    reward_cfg["reward_scales"]["descent_action_sat"] = args.action_sat_weight
    env_cfg["decimation"] = args.decimation
    lo, hi = env_cfg["init_alt_range"]
    env_cfg["init_alt_range"] = (args.init_alt_min if args.init_alt_min is not None else lo,
                                 args.init_alt_max if args.init_alt_max is not None else hi)
    if args.episode_s is not None:
        env_cfg["episode_length_s"] = args.episode_s
    train_cfg = get_train_cfg(exp_name, beta=args.beta, decimation=args.decimation)
    if args.gamma is not None:
        train_cfg["algorithm"]["gamma"] = args.gamma
    if args.symmetry:
        train_cfg["algorithm"]["symmetry_cfg"] = {
            "data_augmentation_func": "fdrone.symmetry:augment",
            "use_data_augmentation": True, "use_mirror_loss": False, "mirror_loss_coeff": 0.0,
        }
    if args.std_max is not None:
        # cfgs.pkl にも残す (評価スクリプトは cfgs.pkl から actor を作り直すので、上限が評価にも効く)
        train_cfg["actor"]["distribution_cfg"]["std_range"] = (1e-6, args.std_max)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    with open(f"{log_dir}/cfgs.pkl", "wb") as f:
        pickle.dump([env_cfg, obs_cfg, reward_cfg, train_cfg], f)
    # 学習条件の記録: どのコミットのコードで学習したか。GPU マシンは git pull ではなく
    # git archive で同期しているので (.git は古いまま)、同期時に書いたコミットを写す
    synced = Path(__file__).resolve().parent.parent / ".synced_commit"
    with open(f"{log_dir}/run_info.txt", "w") as f:
        f.write(f"commit={synced.read_text().strip() if synced.exists() else 'unknown'}\n")
        f.write(f"stage={args.stage} variant={args.variant} seed={args.seed} num_envs={args.num_envs} "
                f"max_iterations={args.max_iterations} translate={args.translate} "
                f"domain_rand={args.domain_rand} landing_offset={args.landing_offset} "
                f"decimation={args.decimation} beta={args.beta} "
                f"smooth={reward_cfg['reward_scales']['smooth']} land_fail_mode={args.land_fail_mode} "
                f"landing_pos_penalty={args.landing_pos_penalty} "
                f"descent_xy_far_scale={args.descent_xy_far_scale} "
                f"descent_xy_far_weight={reward_cfg['reward_scales']['descent_xy_far']} "
                f"drone={args.drone} thrust_to_weight={env_cfg['thrust_to_weight']} "
                f"yaw_rate_cont_weight={args.yaw_rate_cont_weight} climb_weight={args.climb_weight} "
                f"speed_mag_weight={args.speed_mag_weight} action_sat_weight={args.action_sat_weight} "
                f"gamma={train_cfg['algorithm']['gamma']} init_alt_range={env_cfg['init_alt_range']} "
                f"episode_length_s={env_cfg['episode_length_s']} resume_from={args.resume_from} fault_rotors={args.fault_rotors} symmetry={args.symmetry} "
                f"lin_drag_scale={env_cfg.get('lin_drag_scale', 1.0)} form_drag_k={env_cfg.get('form_drag_k', 2.6e-8)} "
                f"dr={dr if args.domain_rand else None} "
                f"std_max={args.std_max} reset_std={args.reset_std}\n")

    env = FaultLandEnv(num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg)
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    if args.resume_from is not None:
        runner.load(args.resume_from, map_location=str(gs.device))
        print(f"[resume] {args.resume_from} から再開 (iteration {runner.current_learning_iteration})")
    dist = runner.alg.actor.distribution
    if args.reset_std is not None:
        with torch.no_grad():
            dist.std_param.fill_(args.reset_std)
    if hasattr(dist, "std_param"):
        print(f"[std] std_param={[round(v, 2) for v in dist.std_param.detach().cpu().tolist()]} "
              f"std_range={dist.std_range}")
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
