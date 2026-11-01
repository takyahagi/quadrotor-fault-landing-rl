"""学習・評価の設定。"""
import copy
from pathlib import Path
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]

DT = 0.0025  # 400 Hz (ArduPilot Copter の SCHED_LOOP_RATE 上限に合わせる)

# 機体プリセット (2026-10-20、ユーザー指示: マイクロドローン以外の機体規模で試す)。
# arm は Genesis が URDF から属性として公開しないので自分で持つ。KF/KM/質量は
# Genesis が URDF から読んで drone.KF 等に公開するので、ここでは持たない
# (DroneScene が起動時に動的に hover_rpm を計算する)。
# lin_drag_coeff は cf2x.urdf/racer.urdf の drag_coeff_xy/z (同じ値)。pelican は
# RotorS (Apache-2.0、Furrer et al. 2016) の rotor_drag_coefficient を単位を合わせず
# そのまま流用した近似 (未検証)
DRONE_PRESETS = {
    # cf2x: Crazyflie 2.X、質量 27 g、アーム長 3.97 cm (既定、これまでの全実験)
    "cf2x": {"urdf": "urdf/drones/cf2x.urdf", "arm": 0.0397,
             "lin_drag_coeff": (9.1785e-7, 9.1785e-7, 10.311e-7),
             # 形状抵抗: アーム 9.2e-9 + モータ 1.7e-8
             "form_drag_k_est": 2.6e-8},
    # racer: gym-pybullet-drones 同梱のレーシングドローン、質量 830 g (cf2x の約 30 倍)、
    # アーム長 10.9 cm、最高速 200 km/h。X 配置なので PID ミキサーは変更不要
    "racer": {"urdf": "urdf/drones/racer.urdf", "arm": 0.109,
              "lin_drag_coeff": (9.1785e-7, 9.1785e-7, 10.311e-7),
              # 形状抵抗の見積もり (寸法は 5 インチ級レーサーの一般的な値の仮定):
              # アーム 4 本 (長さ 0.109 m・厚さ 5 mm・Cd 1.2) 5.1e-7 + モータ 4 個 (直径 28 mm・高さ 20 mm・Cd 1.0) 1.74e-6
              "form_drag_k_est": 2.25e-6},
    # pelican: AscTec Pelican (RotorS, ETH Zurich, Apache-2.0)、質量 1.0 kg、アーム長 0.21 m。
    # + 配置 (前後左右) なので PID ミキサーの式が X 配置用のままでは合わない (要修正)
    # URDF は Genesis に同梱されていないので、repo の assets/urdf/pelican.urdf (RotorS の値から作成、2026-10-27) を使う
    "pelican": {"urdf": str(Path(__file__).resolve().parents[2] / "assets/urdf/pelican.urdf"), "arm": 0.21,
                "lin_drag_coeff": (8.06428e-5, 8.06428e-5, 8.06428e-5), "layout": "plus",
                # 形状抵抗の見積もり (寸法は仮定): アーム 4 本 (長さ 0.21 m・厚さ 10 mm・Cd 1.2) 1.40e-5
                # + モータ 4 個 (直径 28 mm・高さ 25 mm・Cd 1.0) 1.56e-5
                "form_drag_k_est": 2.96e-5},
}

BASE_ENV_CFG = {
    "dt": DT,
    "drone": "cf2x",
    # 方策の 1 ステップあたりの物理ステップ数。1 = 方策も 400 Hz (従来)、4 = 100 Hz、8 = 50 Hz。
    # 行動のノイズがモータの 1 次遅れでならされて探索が効かない問題への対策
    "decimation": 1,
    "episode_length_s": 25.0,
    "init_alt_range": (2.0, 20.0),
    "thrust_to_weight": 2.5,
    "motor_tau": 0.05,
    "aero": True,
    "fault": {"prob": 0.7, "time_range_s": (1.0, 3.0)},
    # v_land: 0.3→0.15 (2026-10-19)。接地判定の高度 (0.05 m) での目標降下速度
    # sqrt(v_land^2 + 2*a_plan*h) が 0.41 m/s で、成功基準 (0.5 m/s 未満) に対して余裕が無く、
    # meanfield 抜きの stoch 評価でも land_fail_vz が失敗の大半を占めていた
    "descent": {"v_max": 1.5, "v_land": 0.15, "a_plan": 0.8},
    "success": {"max_vz": 0.5, "max_vxy": 0.5, "max_tilt_deg": 45.0},
    "crash_tilt_deg": 100.0,
    # 10.0→15.0 (2026-10-19)。landing_offset (最大 8 m) を入れると、着地目標からの許容距離の
    # 大半を offset だけで使ってしまうため広げる
    "max_xy_dist": 15.0,
    "max_climb": 5.0,
    "visualize_camera": False,
    # C4 (ユーザー追加): 故障前に水平移動させるかどうか。既定は無効 (ホバリング)。
    # 有効にすると、故障前の目標が speed_range からサンプルした速度・無作為な向きで動き続け、
    # 故障時にその場で固定される (= 移動中の故障から減速して戻るタスクになる)。
    # speed_range は 2026-10-19 に (1,3)→(0,4) に広げた (meanfield 評価で 3 m/s 付近の劣化を確認)
    "translate": {"enabled": False, "speed_range": (0.0, 4.0)},
    # 将来の安全着陸 (2026-10-19 ユーザーと合意) に向けた準備: 故障した
    # 瞬間の「その場の真下」ではなく、無作為にずらした地点を着陸目標にする。これを下位の制御として、
    # 将来は安全な地点を選ぶ上位の判断と組み合わせる想定
    "landing_offset": {"enabled": False, "max_dist": 8.0},
    # B2: 学習中にエピソードごと
    # (reset_idx のたび) に env ごとランダムに引き直す。範囲は B1 で評価した未知条件を含むように
    # 選んだ (motor_tau は B1 で最も弱かったため基準寄りに絞った範囲ではなく B1 の値まで広げてある)
    "domain_rand": {
        "enabled": False,
        "thrust_to_weight_range": (2.25, 2.5),
        "aero_scale_range": (0.5, 2.0),
        "motor_tau_range": (0.05, 0.1),
        "mass_scale_range": (0.9, 1.1),
        "obs_noise_std": 0.03,
    },
}

OBS_CFG = {
    # pos_err: 1/5→1/10 (2026-10-19)。位置誤差の観測が 5 m で飽和し、着地目標から離れると
    # 引き戻す向きの情報が失われていた。landing_offset (最大 8 m) にも合わせる
    "obs_scales": {"ang_vel": 1 / 10.0, "lin_vel": 1 / 3.0, "pos_err": 1 / 10.0, "alt": 1 / 20.0},
    "fault_flag": True,
}

REWARD_CFG = {
    "reward_scales": {
        "hover_pos": 1.0,
        "hover_upright": 0.2,
        "hover_angvel": -0.01,
        "descent_vz": 1.0,
        "descent_xy": 0.3,
        # 2026-10-19 追加: descent_xy は幅 1 m で、2〜3 m 以上離れると
        # 勾配がほぼ無くなる。幅 5 m の項を足して、遠いところでも目標へ向かう力を残す
        "descent_xy_far": 0.2,
        "descent_tilt": -0.5,
        "descent_rp_rate": -0.002,
        "smooth": -0.01,
        # racer の暴走 を受けたユーザー提案 (2026-10-21)。閾値方式
        # (descent_yaw_excess/descent_tilt) は暴走が大きくなるまで発動しない。結果 (高度・位置の誤差)
        # ではなく先行指標 (角速度・上昇速度・水平速度・行動の飽和) を故障直後から連続的に罰する。
        # 既定 0.0 (既存の実験には影響しない)
        "descent_yaw_rate_cont": 0.0,
        "descent_climb": 0.0,
        "descent_speed_mag": 0.0,
        "descent_action_sat": 0.0,
    },
    "terminal": {
        "success": 10.0,
        "land_fail": -5.0,
        "land_fail_per_speed": 2.0,
        "ground_unfaulted": -10.0,
        "crash": -10.0,
        # 着地目標 (landing_offset) への接近を、毎ステップの整形報酬 (descent_xy/descent_xy_far)
        # だけに任せると descent_vz に埋もれる。既定 0.0 (既存の実験に影響しない)。
        # --landing-pos-penalty で上書きする
        "landing_pos_penalty": 0.0,
    },
}


def get_cfgs(stage: int, variant: str | None = None):
    env_cfg = copy.deepcopy(BASE_ENV_CFG)
    obs_cfg = copy.deepcopy(OBS_CFG)
    reward_cfg = copy.deepcopy(REWARD_CFG)
    if stage == 1:
        # 通常ホバリングのみ (故障なし)
        env_cfg["fault"]["prob"] = 0.0
        env_cfg["episode_length_s"] = 10.0
    elif stage == 2:
        pass
    elif stage == 4:
        # C2: 故障を知らせずに検知させる。観測から故障 rotor の
        # one-hot を外し (fault_flag=False)、代わりに短い履歴 (重力方向・角速度・並進速度・
        # モータ RPM 比を 5 サンプル、0.05 s 間隔 = 計 0.25 s 分) を加える。
        # 他の設定 (故障の確率・報酬) は stage2 と同じ
        obs_cfg["fault_flag"] = False
        obs_cfg["history"] = {"k": 5, "stride_steps": 20}
    else:
        raise ValueError(f"unknown stage {stage}")

    if variant is None:
        pass
    elif variant == "yawpen":
        # 故障後のヨー角速度が 60 rad/s を超えた分を罰する (2026-10-17)。罰則なしの stage2 では
        # 約 230 rad/s の高速回転を選び、400 Hz でも 1 周期に 33° 回る・モデル化していない
        # プロペラのジャイロ効果が効く領域に入ったため
        reward_cfg["reward_scales"]["descent_yaw_excess"] = -0.5
        reward_cfg["yaw_free"] = 60.0
    elif variant == "yawpen_touchdown":
        # yawpen (全体を通じた高速回転の罰則) に加え、接地直前 (高度 3 m 未満) だけ
        # ヨー角速度 10 rad/s 超を強く罰する (案 3、フェーズ C1。2026-10-18、A2 と並行して前倒し)。
        # A3 の仮説: 回転を止めてから接地すれば、水平速度の振動が止まり
        # land_fail_vxy が減るはず。まだ検証していない仮説への対策で、効果は未確認
        reward_cfg["reward_scales"]["descent_yaw_excess"] = -0.5
        reward_cfg["yaw_free"] = 60.0
        reward_cfg["reward_scales"]["touchdown_calm"] = -1.0
        reward_cfg["touchdown_calm_alt"] = 3.0
        reward_cfg["touchdown_calm_yaw_free"] = 10.0
    else:
        raise ValueError(f"unknown variant {variant}")
    return env_cfg, obs_cfg, reward_cfg


def get_train_cfg(exp_name: str, beta: bool = False, decimation: int = 1):
    # 既定は Gaussian。Beta は 400 Hz・モータ 1 次遅れの下では探索が足りず学習できなかった
    #。Gaussian は std が発散して評価時 (det) と別の制御器になるので、
    # 評価は stoch か meanfield で行う (5l〜5q 節)
    if beta:
        distribution_cfg = {"class_name": "BetaDistribution", "action_range": (-1.0, 1.0)}
    else:
        distribution_cfg = {"class_name": "GaussianDistribution", "init_std": 0.5, "std_type": "scalar"}
    return {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.004,
            # 方策の周期に合わせて、秒単位の見通しを揃える (400 Hz で 0.9975 / 0.987。
            # 100 Hz (decimation=4) では 0.990 / 0.949、50 Hz (8) では 0.980 / 0.901)
            "gamma": 0.9975**decimation,
            "lam": 0.987**decimation,
            "learning_rate": 0.0003,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [256, 256],
            "activation": "tanh",
            "distribution_cfg": distribution_cfg,
        },
        "critic": {"class_name": "MLPModel", "hidden_dims": [256, 256], "activation": "tanh"},
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "num_steps_per_env": 400 // decimation,  # = 1 s (1 iteration あたりの物理の量を揃える)
        "save_interval": 100,
        "run_name": exp_name,
        "logger": "tensorboard",
    }
