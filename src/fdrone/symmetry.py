"""機体の左右対称 (機体 x-z 面での鏡映) を使った観測・行動の変換 (2026-10-26)。

racer は C4 (移動中の故障) の学習で、鏡像の関係にある 2 種類の故障 (ロータ 0・2 と 1・3) の片方だけを学ぶ局所解に
3 seed 中 2 本が入った (6al 節)。機体は y 軸の鏡映 (y → −y) で厳密に対称なので、学習データを鏡映して増やす
(rsl-rl 5.5.1 の `Symmetry` 拡張、Mittal et al. ICRA 2024) と、片側だけを学ぶことを防げるかもしれない。

鏡映 y → −y での各量の変換:
  - 普通のベクトル (重力の向き・速度・位置誤差): y 成分の符号を反転
  - 角速度 (軸性ベクトル): (p, q, r) → (−p, q, −r)
  - ロータごとの量 (行動・モータ回転数・故障の one-hot): ロータを入れ替える
    racer: 0 (+x, +y) ↔ 3 (+x, −y)、1 (−x, +y) ↔ 2 (−x, −y)。cf2x も同じ入れ替え (0 (+, −) ↔ 3 (+, +)、1 ↔ 2)
    回転方向 spin = [−1, 1, −1, 1] は入れ替えで符号が反転する (鏡映で回転の向きが逆になるのと整合)
  - 高度・降下速度・その目標値: そのまま

観測の並びは `FaultLandEnv._get_obs()` と一致させてある。並びを変えたらここも直し、
`scripts/verify/check_mirror.py` で物理の等価性を確かめること。
"""
import torch
from tensordict import TensorDict

MIRROR_PERM = [3, 2, 1, 0]

_VEC = (1.0, -1.0, 1.0)   # 普通のベクトル
_AXIAL = (-1.0, 1.0, -1.0)  # 角速度


def _obs_map(obs_cfg: dict, n_rotors: int = 4):
    """観測 1 本分の (並べ替えの添字, 符号) を返す。mirrored = x[..., idx] * sign"""
    idx, sign = [], []

    def add_vec(s):
        base = len(idx)
        idx.extend([base, base + 1, base + 2])
        sign.extend(s)

    def add_scalar():
        idx.append(len(idx))
        sign.append(1.0)

    def add_rotors():
        base = len(idx)
        idx.extend([base + p for p in MIRROR_PERM])
        sign.extend([1.0] * n_rotors)

    add_vec(_VEC)    # 重力の向き (機体座標)
    add_vec(_AXIAL)  # 角速度
    add_vec(_VEC)    # 速度 (機体座標)
    add_vec(_VEC)    # 位置誤差 (機体座標)
    add_scalar()     # 高度
    add_scalar()     # 降下速度
    add_scalar()     # 降下速度の目標値
    add_rotors()     # 前回の行動
    add_rotors()     # モータ回転数の比
    if obs_cfg.get("fault_flag", True):
        add_rotors()  # 故障ロータの one-hot (stage 2)
    hc = obs_cfg.get("history")
    if hc is not None:
        for _ in range(hc["k"]):
            add_vec(_VEC)
            add_vec(_AXIAL)
            add_vec(_VEC)
            add_rotors()
    return idx, sign


_CACHE = {}


def _maps(env, device):
    key = (id(env), str(device))
    if key not in _CACHE:
        spin = env.ds.spin.detach().cpu()
        assert torch.equal(spin[MIRROR_PERM], -spin), f"機体の回転方向 {spin.tolist()} がこの鏡映と合わない"
        idx, sign = _obs_map(env.obs_cfg, env.ds.n_rotors)
        _CACHE[key] = (torch.tensor(idx, device=device), torch.tensor(sign, device=device))
    return _CACHE[key]


def mirror_obs_tensor(env, x: torch.Tensor) -> torch.Tensor:
    idx, sign = _maps(env, x.device)
    assert x.shape[-1] == len(idx), f"観測の次元 {x.shape[-1]} が鏡映の表 {len(idx)} と合わない"
    return x[..., idx] * sign


def mirror_actions(a: torch.Tensor) -> torch.Tensor:
    return a[..., MIRROR_PERM]


def augment(env, obs=None, actions=None):
    """rsl-rl の Symmetry 拡張から呼ばれる。元のサンプルの後ろに鏡映したサンプルを連結して返す。"""
    obs_aug = None
    if obs is not None:
        o = obs["policy"]
        obs_aug = TensorDict({"policy": torch.cat([o, mirror_obs_tensor(env, o)], dim=0)},
                             batch_size=[2 * o.shape[0]], device=o.device)
    act_aug = None
    if actions is not None:
        act_aug = torch.cat([actions, mirror_actions(actions)], dim=0)
    return obs_aug, act_aug
