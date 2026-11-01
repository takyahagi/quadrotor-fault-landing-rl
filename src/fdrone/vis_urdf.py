"""録画用に、racer の見た目だけを変えた URDF を作る (2026-10-28)。

Genesis 同梱の racer.urdf (gym-pybullet-drones 由来) は、見た目が base_link の 1 つのメッシュ (Crazyflie の cf2.dae を 3 倍に
拡大、プロペラも一体) で、プロペラのリンクに見た目が無い。そのため Genesis はプロペラのアニメーションを省き、動画では
プロペラが止まって見える。ここでは、cf2x の本体 (body.obj) とプロペラ (propeller0〜3.obj、どちらも Genesis 同梱、Apache-2.0)
を racer の寸法に合わせて拡大し、プロペラのリンクに付けた URDF を一時ファイルに書く。質量・慣性・プロペラの位置など
物理に使う値は元の racer.urdf のまま変えない (見た目の要素だけを足し替える)。
"""
import os
import re
import tempfile

import genesis as gs

# cf2x のプロペラの見た目の中心は (±0.0323, ±0.0323, 0.0132) [m]。racer のプロペラの位置は (±0.085, ±0.0675) [m]
_CF2X_PROP_XY = 0.0323
_CF2X_PROP_Z = 0.0132
_CF2X_PROP_RADIUS = 0.0196  # propeller*.obj の半径 (頂点の範囲から)
_RACER_PROP_RADIUS = 0.0635  # 5 インチのプロペラ (直径 12.7 cm)


def racer_visual_urdf() -> str:
    """見た目を差し替えた racer の URDF を一時ファイルに書き、そのパスを返す。"""
    drones = os.path.join(gs.utils.get_assets_dir(), "urdf", "drones")
    src = open(os.path.join(drones, "racer.urdf"), encoding="utf-8").read()
    # プロペラの位置は、各 prop リンクの inertial の origin (物理で推力をかける位置)
    pos = {int(m.group(1)): (float(m.group(2)), float(m.group(3)))
           for m in re.finditer(r'<link name="prop(\d)_link">\s*<inertial>\s*<origin rpy="[^"]*" xyz="([-\d.e]+) ([-\d.e]+) [-\d.e]+"', src)}
    sx = abs(pos[0][0]) / _CF2X_PROP_XY
    sy = abs(pos[0][1]) / _CF2X_PROP_XY
    sz = (sx + sy) / 2
    body = (f'<origin rpy="0 0 0" xyz="0 0 0"/>\n      <geometry>\n        '
            f'<mesh filename="{os.path.join(drones, "body.obj")}" scale="{sx:.3f} {sy:.3f} {sz:.3f}"/>\n      </geometry>')
    src, n = re.subn(r'<origin rpy="0 0 55" xyz="0 0 0"/>\s*<geometry>\s*<mesh filename="\./cf2\.dae" scale="3 3 3"/>\s*</geometry>', body, src)
    assert n == 1, "racer.urdf の base_link の見た目が想定と違う"
    ps = _RACER_PROP_RADIUS / _CF2X_PROP_RADIUS
    for i, (x, y) in pos.items():
        vis = (f'\n    <visual>\n      <origin rpy="0 0 0" xyz="{x} {y} {_CF2X_PROP_Z * sz:.4f}"/>\n      <geometry>\n        '
               f'<mesh filename="{os.path.join(drones, f"propeller{i}.obj")}" scale="{ps:.3f} {ps:.3f} {ps:.3f}"/>\n'
               f'      </geometry>\n    </visual>')
        src, n = re.subn(rf'(<link name="prop{i}_link">\s*<inertial>.*?</inertial>)', lambda m: m.group(1) + vis, src, flags=re.S)
        assert n == 1
    fd, path = tempfile.mkstemp(prefix="racer_vis_", suffix=".urdf")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(src)
    return path
