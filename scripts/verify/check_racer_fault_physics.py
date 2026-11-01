"""racer の故障の物理を、学習を挟まずに開ループで確かめる (cf2x と比較、2026-10-21)。

racer は 1500 iteration 学習しても故障時の成功が 0%。報酬を疑う前に、
物理モデル側の食い違いが無いかを見る:

  A. ホバリング中に 0 番ロータを止めた直後 (0.3 s) の、ロール/ピッチ/ヨーの向き。
     期待値: 止めたロータの側に傾く (そのロータの高さが下がる)。ヨーの角加速度は
     -spin[0] の向き (残り 3 枚の反トルクの和 = -spin[0] * KM * rpm^2)
  B. 傾きを生まない純粋なヨートルク (故障時と同じ大きさ) をかけ続けたとき、ヨー角速度が
     頭打ちになるか・何 rad/s で釣り合うか。racer は空気抵抗の係数を cf2x から流用しているので
     (configs.py の DRONE_PRESETS)、減衰が小さすぎて頭打ちにならない可能性がある

CPU, 1 env。実行例:
  python scripts/verify/check_racer_fault_physics.py --drone racer
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import torch
import genesis as gs

from fdrone.configs import DRONE_PRESETS, DT
from fdrone.envs.drone_scene import DroneScene


def make_scene(name, lin_drag_scale=1.0, form_drag_k=2.6e-8):
    p = DRONE_PRESETS[name]
    return DroneScene(num_envs=1, dt=DT, base_init_pos=(0.0, 0.0, 200.0), aero=True, form_drag_k=form_drag_k,
                      drone_file=p["urdf"], drone_arm=p["arm"],
                      drone_lin_drag_coeff=tuple(c * lin_drag_scale for c in p["lin_drag_coeff"]))


def prop_link_z(ds):
    """プロペラのリンクの world 高さ (base_link を除く全リンク)。"""
    pos = ds.drone.get_links_pos()[0]  # (n_links, 3)
    names = [l.name for l in ds.drone.links]
    return names, pos


def test_a(ds):
    n = ds.n_rotors
    h = ds.hover_rpm
    spin = ds.spin.tolist()
    print(f"[A] mass={float(ds.drone.get_mass()):.4f} kg KF={ds.drone.KF:.3e} KM={ds.drone.KM:.3e} "
          f"hover_rpm={h:.0f} spin={spin}")
    names, pos = prop_link_z(ds)
    for nm, p in zip(names, pos.tolist()):
        print(f"    link {nm:12s} body-pos(at t=0) x={p[0]:+.4f} y={p[1]:+.4f} z={p[2] - 200.0:+.4f}")
    rpm = torch.full((1, n), h, device=gs.device)
    for _ in range(int(0.2 / DT)):
        ds.set_propellers_rpm(rpm)
        ds.step()
    absent = torch.zeros((1, n), dtype=torch.bool, device=gs.device)
    absent[0, 0] = True
    rpm_f = torch.full((1, n), h * math.sqrt(n / (n - 1)), device=gs.device)
    rpm_f[0, 0] = 0.0
    print(f"    期待: ヨー角加速度の符号 = {-spin[0]:+.0f} (= -spin[0])、0 番ロータ側が下がる")
    for k in range(int(0.3 / DT) + 1):
        if k % int(0.05 / DT) == 0:
            w = ds.base_ang_vel[0].tolist()
            _, pos = prop_link_z(ds)
            z = (pos[:, 2] - ds.base_pos[0, 2]).tolist()
            print(f"    t+{k * DT:.2f}s p={w[0]:+8.2f} q={w[1]:+8.2f} r={w[2]:+8.2f} rad/s  "
                  f"tilt={float(ds.tilt_deg()[0]):6.1f}deg  link dz=" + " ".join(f"{v:+.3f}" for v in z))
        ds.set_propellers_rpm(rpm_f, absent)
        ds.step()


def test_b(ds, t_total, csv_path=None):
    """故障時と同じ大きさのヨートルクを、傾きを生まない形でかける。

    故障時のヨートルク = KM * rpm_f^2 (rpm_f^2 = hover^2 * 4/3)。対角 2 枚を (1+d)、残りを (1-d) にした
    ときのヨートルクは 4 d KM hover^2 なので、d = 1/3 で揃う。
    """
    h = ds.hover_rpm
    d = 1.0 / 3.0
    spin = ds.spin.tolist()
    rpm = torch.tensor([[h * math.sqrt(1 + d) if s > 0 else h * math.sqrt(1 - d) for s in spin]], device=gs.device)
    tau = ds.drone.KM * sum(s * r ** 2 for s, r in zip(spin, rpm[0].tolist()))
    print(f"[B] 純粋なヨートルク {tau:.3e} N*m (故障時と同じ大きさ) を {t_total:.0f} s かけ続ける")
    f = open(csv_path, "w") if csv_path else None
    if f:
        f.write("t,yaw_rate\n")
    for k in range(int(t_total / DT) + 1):
        if f and k % int(0.05 / DT) == 0:
            f.write(f"{k * DT:.3f},{float(ds.base_ang_vel[0, 2]):.4f}\n")
        if k % int(1.0 / DT) == 0:
            print(f"    t={k * DT:5.1f}s r={float(ds.base_ang_vel[0, 2]):+8.2f} rad/s "
                  f"tilt={float(ds.tilt_deg()[0]):5.1f}deg z={float(ds.base_pos[0, 2]):7.1f}")
        ds.set_propellers_rpm(rpm)
        ds.step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drone", choices=list(DRONE_PRESETS), default="racer")
    parser.add_argument("--test", choices=["a", "b"], default="a")
    parser.add_argument("--csv", default=None, help="test b のヨー角速度を 0.05 s ごとに CSV に書く (レポートの図用)")
    parser.add_argument("--lin-drag-scale", type=float, default=1.0, help="並進抵抗・ヨー減衰の係数の倍率")
    parser.add_argument("--form-drag-k", default="2.6e-8", help="形状抵抗の係数 (数値、または est = 機体ごとの見積もり)")
    parser.add_argument("--t-total", type=float, default=15.0)
    args = parser.parse_args()
    gs.init(backend=gs.cpu, precision="32", logging_level="warning")
    k = DRONE_PRESETS[args.drone]["form_drag_k_est"] if args.form_drag_k == "est" else float(args.form_drag_k)
    print(f"形状抵抗 k={k:.3g}、並進抵抗・ヨー減衰の係数 ×{args.lin_drag_scale}")
    ds = make_scene(args.drone, args.lin_drag_scale, k)
    print(f"=== {args.drone}")
    if args.test == "a":
        test_a(ds)
    else:
        test_b(ds, args.t_total, args.csv)


if __name__ == "__main__":
    main()
