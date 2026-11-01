"""学習済みポリシーを 1 env で動かし、動画と着地時の値を出す。
Genesis examples/drone/hover_eval.py (Apache-2.0) をベースに、FaultLandEnv と
DroneScene の第三者視点カメラ (follow_entity) に差し替えた。

実行例:
  python scripts/eval.py -e stage2 --ckpt 1500 --alt 20 --fault-rotor 0 --fault-time 2.0 --record
  python scripts/eval.py -e stage1 --ckpt 300 --alt 20 --no-fault --record
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import math

import numpy as np
import torch
import imageio.v2 as imageio
from PIL import Image, ImageDraw
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from fdrone.envs.fault_land_env import FaultLandEnv
from fdrone.eval_utils import ACTION_MODES, make_action_fn

VIDEO_FPS = 25


def overlay_text(rgb: np.ndarray, lines: list[str]) -> np.ndarray:
    """高度などの数値をフレームに焼き込む。follow カメラだと機体が常に同じ大きさに
    映るので、文字で読み上げないと「高いところから落ちている」ことが伝わらない
    (2026-10-17、固定カメラで見せる案は機体が小さすぎて不採用にした)。"""
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    from PIL import ImageFont

    font = ImageFont.load_default(size=28)
    for i, line in enumerate(lines):
        y = 16 + i * 34
        draw.text((16, y), line, fill=(255, 255, 0), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    return np.array(img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", type=str, required=True)
    parser.add_argument("--ckpt", type=int, required=True)
    parser.add_argument("--alt", type=float, default=20.0)
    parser.add_argument("--fault-rotor", type=int, default=0)
    parser.add_argument("--fault-time", type=float, default=2.0)
    parser.add_argument("--no-fault", action="store_true")
    parser.add_argument("--speed", type=float, default=0.0, help="C4: 故障前の水平移動速度 [m/s] (既定 0 = 静止)")
    parser.add_argument("--heading-deg", type=float, default=0.0, help="C4: 故障前の移動方向 [deg]")
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--alt-obs-max", type=float, default=None,
                        help="方策に見せる高度の上限 [m] (既定なし)。学習した高度より高い所からの評価で、観測を学習時の範囲に収める")
    parser.add_argument("--motor-tau", type=float, default=None, help="モータ時定数を上書き [s] (未知条件、eval_grid.py と同じ)")
    parser.add_argument("--racer-visual", action="store_true",
                        help="racer の録画で、プロペラが回って見える URDF (見た目だけ差し替え、src/fdrone/vis_urdf.py) を使う")
    parser.add_argument("-r", "--record", action="store_true")
    parser.add_argument("--cam-mode", choices=["follow", "static"], default="follow",
                         help="follow: 機体に追従 (姿勢の detail) / static: 固定カメラ (落下距離が分かる)")
    parser.add_argument("--action-mode", choices=ACTION_MODES, default="det",
                        help="行動の作り方 (eval_grid.py と同じ)。Gaussian の方策は det だと学習時と別の制御器に"
                             "なるので、成功例の録画は stoch で撮る")
    parser.add_argument("--seed", type=int, default=0, help="stoch のサンプリングの乱数の種")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--trace-csv", default=None,
                        help="方策の 1 ステップごとの時系列 (t, 高度, 降下速度, 傾き, ヨー角速度, 故障) を CSV に書く (レポートの図用)")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu, logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    with open(f"{log_dir}/cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, train_cfg = pickle.load(f)
    env_cfg["visualize_camera"] = args.record
    env_cfg["cam_mode"] = args.cam_mode
    if args.cam_mode == "static":
        # 落下距離を見せたいとき用の、少し広めの固定カメラ位置 (高度 0〜20 m の全体が入る)
        env_cfg["cam_static_pos"] = (12.0, 12.0, 8.0)
        env_cfg["cam_static_lookat"] = (0.0, 0.0, 7.0)
        env_cfg["cam_static_fov"] = 80.0
    env_cfg["init_alt_range"] = (args.alt, args.alt)
    env_cfg["fault"]["prob"] = 0.0
    env_cfg.setdefault("domain_rand", {})["enabled"] = False  # 1 例の観察なので固定条件にする
    # crash (tilt>crash_tilt_deg) は姿勢を失った瞬間の判定で、接地よりずっと前に成立する。動画では
    # そこで終わらず、実際に地面へ落ちるまで録画を続けたいので、エピソード終了直後の
    # 自動リセットを止める (2026-10-20)
    env_cfg["auto_reset"] = False
    if args.duration_s is not None:
        env_cfg["episode_length_s"] = args.duration_s
    if args.alt_obs_max is not None:
        env_cfg["alt_obs_max"] = args.alt_obs_max
    if args.motor_tau is not None:
        env_cfg["motor_tau"] = args.motor_tau
    if args.racer_visual:
        from fdrone.vis_urdf import racer_visual_urdf

        assert env_cfg.get("drone") == "racer", "--racer-visual は racer の方策だけ"
        env_cfg["drone_file_override"] = racer_visual_urdf()

    env = FaultLandEnv(num_envs=1, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg)
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{args.ckpt}.pt"), map_location=str(gs.device))
    actor = runner.alg.actor
    actor.eval()
    torch.manual_seed(args.seed)
    policy = make_action_fn(actor, args.action_mode)

    obs = env.reset()
    if args.speed > 0.0:
        # eval_utils.setup() / ros2_ws の sim_bridge_node と同じやり方。cfgs.pkl の
        # translate.enabled が True だと reset() 時点で無作為な速度・向きが入っているので上書きする
        heading = math.radians(args.heading_deg)
        env.v_cmd[0, 0] = args.speed * math.cos(heading)
        env.v_cmd[0, 1] = args.speed * math.sin(heading)
        vel6 = torch.zeros((1, 6), device=env.device)
        vel6[0, :2] = env.v_cmd[0, :2]
        env.ds.drone.set_dofs_velocity(vel6, envs_idx=torch.tensor([0], device=env.device))
    if not args.no_fault:
        env.fault.force_fault(env_idx=0, rotor_idx=args.fault_rotor, step=int(args.fault_time / env.dt))

    # env.step() は 1 回で decimation 物理ステップ分を進める。このループは
    # 方策ステップ単位 (= policy_dt) で回るので、フレーム間隔・経過時間の表示も policy_dt 基準にする
    pdt = env.policy_dt
    stride = max(1, int(round((1.0 / pdt) / VIDEO_FPS)))
    print_stride = max(1, int(round(1.0 / pdt)))
    # crash は姿勢を失った瞬間の判定で、接地よりずっと前に成立する (tilt>crash_tilt_deg)。
    # auto_reset=False にしてあるので、ここで打ち切らず実際に接地するまで録画を続ける
    POST_END_S = 6.0
    # 接地判定が付いた瞬間に切ると、stride (0.04s) の境目とずれて高度 0 m 付近が映らないことが
    # あった (2026-10-20、ユーザー指摘)。接地後も実際に物理を進めながら、着地の様子が分かる
    # 程度に録画を続ける
    GROUND_HOLD_S = 1.0
    frames = []
    max_yaw_rate = 0.0
    ended = False
    end_time = None
    grounded_since = None
    n_steps = env.max_episode_length // env.decimation + 1 + int(POST_END_S / pdt)
    if args.trace_csv:
        os.makedirs(os.path.dirname(args.trace_csv) or ".", exist_ok=True)
    trace = open(args.trace_csv, "w") if args.trace_csv else None
    if trace:
        # wx・wy: 機体座標のロール・ピッチの角速度、gx・gy: 機体座標で見た重力の向き (傾きの向き)、rpm0〜3: モータの回転数
        # (回転が軸を結び付けているか・推力の配分を見る、2026-10-28 追加)
        trace.write("t,alt,vz,tilt,yaw_rate,faulted,vxy,wx,wy,gx,gy,rpm0,rpm1,rpm2,rpm3\n")
    with torch.no_grad():
        for i in range(n_steps):
            obs, _, dones, _ = env.step(policy(obs))
            max_yaw_rate = max(max_yaw_rate, abs(env.ds.base_ang_vel[0, 2].item()))
            if trace:
                trace.write(f"{(i + 1) * pdt:.3f},{env.alt[0].item():.4f},{env.vz[0].item():.4f},{env.tilt[0].item():.2f},"
                            f"{env.ds.base_ang_vel[0, 2].item():.3f},{int(bool(env.faulted[0]))},"
                            f"{torch.norm(env.ds.drone.get_vel()[0, :2]).item():.4f},"
                            f"{env.ds.base_ang_vel[0, 0].item():.3f},{env.ds.base_ang_vel[0, 1].item():.3f},"
                            f"{env.gravity_body[0, 0].item():.4f},{env.gravity_body[0, 1].item():.4f},"
                            + ",".join(f"{v:.0f}" for v in env.motor.rpm[0].tolist()) + "\n")
            if args.record and i % stride == 0:
                rgb = env.ds.capture_frame()
                vxy = torch.norm(env.ds.drone.get_vel()[0, :2]).item()
                dist_to_target = torch.norm(env.target[0, :2] - env.ds.base_pos[0, :2]).item()
                rgb = overlay_text(rgb, [
                    f"t = {i * pdt:5.1f} s",
                    f"alt = {env.alt[0].item():5.1f} m",
                    f"horiz speed = {vxy:4.1f} m/s",
                    f"dist to target = {dist_to_target:4.1f} m",
                    f"yaw rate = {env.ds.base_ang_vel[0, 2].item():6.0f} rad/s",
                ])
                frames.append(rgb)
            if i % print_stride == 0:
                print(
                    f"t={i * pdt:5.1f}s alt={env.alt[0].item():6.2f} vz={env.vz[0].item():6.2f} "
                    f"v_ref={env.v_ref[0].item():5.2f} tilt={env.tilt[0].item():5.1f} "
                    f"yaw_rate={env.ds.base_ang_vel[0, 2].item():7.1f} faulted={bool(env.faulted[0])}"
                )
            if not ended and bool(dones[0]):
                ended = True
                end_time = i * pdt
                v = env.prev_world_vel[0]
                kind = (
                    "SUCCESS" if bool(env.success[0]) else
                    "LAND_FAIL" if bool(env.landed_fail[0]) else
                    "CRASH" if bool(env.crashed[0]) else
                    "GROUND(no fault)" if bool(env.ground_unfaulted[0]) else "TIMEOUT"
                )
                print(
                    f"[end] t={i * pdt:.2f}s {kind}: 接地直前 vz={v[2].item():.2f} m/s "
                    f"vxy={torch.norm(v[:2]).item():.2f} m/s tilt={env.prev_tilt[0].item():.1f} deg "
                    f"/ 最大ヨー角速度 {max_yaw_rate:.1f} rad/s"
                )
                if not args.record:
                    break  # 録画しないときは判定が付いた時点で終わる (従来どおり)
            if ended and grounded_since is None and bool(env.ds.grounded_mask()[0]):
                grounded_since = i * pdt
            if ended and (
                (grounded_since is not None and i * pdt - grounded_since >= GROUND_HOLD_S)
                or i * pdt - end_time >= POST_END_S
            ):
                print(f"[ground] t={i * pdt:.2f}s alt={env.alt[0].item():.2f}m (録画の終端)")
                break

    if args.record and frames:
        tag = "nofault" if args.no_fault else f"r{args.fault_rotor}"
        if args.speed > 0.0:
            tag += f"_s{args.speed:g}h{args.heading_deg:g}"
        out_path = Path(log_dir) / f"eval_{args.ckpt}_alt{args.alt:g}_{tag}.mp4"
        imageio.mimwrite(out_path, frames, fps=VIDEO_FPS, macro_block_size=None)
        print(f"[saved] {out_path} ({len(frames)} frames @ {VIDEO_FPS}fps)")


if __name__ == "__main__":
    main()
