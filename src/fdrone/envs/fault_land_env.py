"""ホバリング + プロペラ 1 枚故障時の軟着陸の学習環境 (Stage 1〜4 を設定で切替)。

Genesis examples/drone/hover_env.py (Apache-2.0) の構成 (報酬項を `_reward_<名前>` で持ち、
重みに dt を掛けて合計する形、rsl-rl 向けの step/reset の形) を元に、モータ・故障・接地・
降下の報酬などを加えた。Stage 1 は fault.prob=0 (故障なし) で使う。
観測はヨー角に依存しない量 (重力の向き・機体座標系の速度など) で組み、故障後に
機体が回り続けても定義が崩れないようにしてある。
"""
import math

import torch
from tensordict import TensorDict
from genesis.utils.geom import inv_quat, transform_by_quat

from fdrone.envs.drone_scene import DroneScene
from fdrone.envs.motor import MotorModel
from fdrone.envs.fault import FaultInjector
from fdrone.configs import DRONE_PRESETS

G = 9.81


def rand_uniform(lo, hi, n, device):
    return (hi - lo) * torch.rand(n, device=device) + lo


class FaultLandEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, show_viewer=False):
        self.num_envs = num_envs
        self.num_actions = 4
        self.env_cfg = env_cfg
        self.cfg = env_cfg  # rsl-rl の Logger が env.cfg をログ用メタデータとして読む
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.dt = env_cfg["dt"]  # 物理の周期 (400 Hz)
        # 方策の 1 ステップあたりの物理ステップ数 (2026-10-19)。1 なら方策も 400 Hz
        self.decimation = env_cfg.get("decimation", 1)
        self.policy_dt = self.dt * self.decimation
        self._substep_idx = 0
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)  # 物理ステップ数

        # B2: domain_rand.enabled のときだけ、各 env のパラメータを
        # reset_idx のたびにランダム化する。評価 (eval_grid.py 等) は常に enabled=False で
        # 固定条件のまま動かすので、既存の挙動は変わらない
        dr = env_cfg.get("domain_rand", {})
        dr_on = dr.get("enabled", False)

        alt_lo, alt_hi = env_cfg["init_alt_range"]
        # 機体プリセット (2026-10-20
        # 規模で試す、ユーザー指示): 既定 cf2x (これまでの全実験と同じ)
        preset = DRONE_PRESETS[env_cfg.get("drone", "cf2x")]
        self.ds = DroneScene(
            num_envs=num_envs,
            dt=self.dt,
            base_init_pos=(0.0, 0.0, alt_hi),
            show_viewer=show_viewer,
            visualize_camera=env_cfg.get("visualize_camera", False),
            cam_mode=env_cfg.get("cam_mode", "follow"),
            cam_static_pos=env_cfg.get("cam_static_pos", (10.0, 10.0, 7.0)),
            cam_static_lookat=env_cfg.get("cam_static_lookat", (0.0, 0.0, 6.0)),
            cam_static_fov=env_cfg.get("cam_static_fov", 75.0),
            aero=env_cfg.get("aero", True),
            aero_scale=env_cfg.get("aero_scale", 1.0),
            aero_scale_range=dr.get("aero_scale_range") if dr_on else None,
            mass_scale_range=dr.get("mass_scale_range") if dr_on else None,
            # drone_file_override: 録画で見た目だけを変えた URDF を使う (src/fdrone/vis_urdf.py)。物理の値は同じ
            drone_file=env_cfg.get("drone_file_override") or preset["urdf"],
            drone_arm=preset["arm"],
            # lin_drag_scale: 並進抵抗・H-force のヨー減衰の係数の倍率 (既定 1)。racer の係数は cf2x の流用で
            # 過小評価の可能性があるため、感度を見る
            drone_lin_drag_coeff=tuple(c * env_cfg.get("lin_drag_scale", 1.0) for c in preset["lin_drag_coeff"]),
            # form_drag_k: 機体の形状抵抗の係数 (既定 2.6e-8 = cf2x の形からの推定値を全機体に使う、従来どおり)。
            # 機体ごとの見積もりは DRONE_PRESETS の form_drag_k_est
            form_drag_k=env_cfg.get("form_drag_k", 2.6e-8),
        )
        self.device = self.ds.device
        n_rot = self.ds.n_rotors

        self.motor = MotorModel(
            num_envs=num_envs,
            n_rotors=n_rot,
            hover_rpm=self.ds.hover_rpm,
            thrust_to_weight=env_cfg["thrust_to_weight"],
            tau=env_cfg["motor_tau"],
            dt=self.dt,
            device=self.device,
            tw_range=dr.get("thrust_to_weight_range") if dr_on else None,
            tau_range=dr.get("motor_tau_range") if dr_on else None,
        )
        self.obs_noise_std = dr.get("obs_noise_std", 0.0) if dr_on else 0.0
        fc = env_cfg["fault"]
        self.fault = FaultInjector(
            num_envs=num_envs,
            n_rotors=n_rot,
            fault_prob=fc["prob"],
            time_range_s=fc["time_range_s"],
            dt=self.dt,
            device=self.device,
            rotors=fc.get("rotors"),
        )

        self.reward_scales = dict(reward_cfg["reward_scales"])
        self.reward_functions, self.episode_sums = {}, {}
        for name in self.reward_scales:
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros(num_envs, device=self.device)

        z = lambda *shape, **kw: torch.zeros(*shape, device=self.device, **kw)  # noqa: E731
        self.target = z(num_envs, 3)
        self.v_cmd = z(num_envs, 3)  # C4: 故障前の水平移動の指令速度 (world, z=0)。既定は 0 (ホバリング)
        self.init_alt = z(num_envs)
        self.actions = z(num_envs, 4)
        self.raw_actions = z(num_envs, 4)  # クリップ前の行動 (飽和の罰則用)
        self.last_actions = z(num_envs, 4)
        self.rew_buf = z(num_envs)
        self.reset_buf = torch.ones(num_envs, dtype=torch.bool, device=self.device)
        self.episode_length_buf = z(num_envs, dtype=torch.long)
        self.faulted = z(num_envs, dtype=torch.bool)
        self.prev_world_vel = z(num_envs, 3)
        self.prev_tilt = z(num_envs)
        self.success = z(num_envs, dtype=torch.bool)
        # 中間指標 (ログのみ、報酬には使わない。): 故障後の最大ヨー角速度 [rad/s]
        self.max_yaw_fault = z(num_envs)
        self.extras = {}
        self._gravity_world = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(num_envs, 3)

        # C2 (stage=4): 故障 one-hot の代わりの短い履歴。
        # 重力方向 (3) + 角速度 (3) + 並進速度 (3) + モータ RPM 比 (4) = 13 次元を k サンプル分
        hc = obs_cfg.get("history")
        self.hist_enabled = hc is not None
        if self.hist_enabled:
            self.hist_k = hc["k"]
            self.hist_stride = hc["stride_steps"]
            self.hist_dim = 13
            self.hist_buf = z(num_envs, self.hist_k, self.hist_dim)

        self.reset()

    # ------------------------------------------------------------------ step
    _LATCHED = ("success", "landed_fail", "ground_unfaulted", "crashed", "out_of_bounds", "faulted",
                "prev_world_vel", "prev_tilt", "impact_speed")

    def step(self, actions):
        """方策の 1 ステップ = 物理の decimation ステップ (同じ行動を保持する)。

        decimation>1 (方策を物理より低い周期で動かす) でも、接地・墜落の判定は
        物理の 1 ステップごとに行い、各 env が最初に終わった物理ステップの値 (接地直前の速度など) を
        保持して返す。終わった env はその後の物理ステップの報酬を加えない。decimation=1 では従来と同じ
        """
        self.raw_actions = actions.detach()
        self.actions = torch.clip(actions, -1.0, 1.0)
        n_sub = self.decimation
        rew = torch.zeros(self.num_envs, device=self.device)
        done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        timeout_l = torch.zeros_like(done)
        latched = {}
        for k in range(n_sub):
            self._substep_idx = k
            r, d, timeout = self._physics_substep(live=~done)
            rew += torch.where(done, torch.zeros_like(r), r)
            newly = d & ~done
            for name in self._LATCHED:
                v = getattr(self, name)
                if k == 0:
                    latched[name] = v.clone()
                else:
                    m = newly.view(-1, *([1] * (v.dim() - 1)))
                    latched[name] = torch.where(m, v, latched[name])
            timeout_l |= timeout & newly
            done |= d
        for name in self._LATCHED:
            # 終わっていない env は最後の物理ステップの値、終わった env は終わったときの値
            v = getattr(self, name)
            m = done.view(-1, *([1] * (v.dim() - 1)))
            setattr(self, name, torch.where(m, latched[name], v))
        timeout = timeout_l & done
        self.rew_buf[:] = rew
        self.reset_buf = done
        self.extras["time_outs"] = timeout.float()

        log = self._termination_stats(timeout)
        self.extras.pop("episode", None)
        # auto_reset=False (録画スクリプト専用、既定 True): 墜落の判定が付いた直後にリセットすると、
        # 実際に地面へ落ちる様子を描けない (tilt>crash_tilt_deg は姿勢を失った瞬間の判定で、
        # 接地より前に成立するため。)
        if self.env_cfg.get("auto_reset", True):
            self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape(-1))
        if "episode" in self.extras:
            # rsl-rl の Logger は extras["episode"] があると extras["log"] を見ないので、こちらに入れる
            self.extras["episode"].update(log)
        self._compute_state()

        obs = self._get_obs()
        self.last_actions[:] = self.actions
        return obs, self.rew_buf, self.reset_buf, self.extras

    def _physics_substep(self, live):
        """物理を 1 ステップ進め、その時点の報酬・終了・タイムアウトを返す (リセットはしない)。"""
        ec = self.env_cfg
        frac = (self.actions + 1.0) * 0.5
        rpm = self.motor.step(self.motor.max_rpm * torch.sqrt(frac))
        rpm, _ = self.fault.apply(rpm, self.episode_length_buf)
        self.ds.set_propellers_rpm(rpm, absent=self.fault.absent_mask())

        self.prev_world_vel[:] = self.ds.drone.get_vel()
        self.prev_tilt[:] = self.ds.tilt_deg()
        self.ds.step()
        self.episode_length_buf += 1

        # C4: 故障前は目標を v_cmd で動かす (ホバリングのときは v_cmd=0 なので変わらない)。
        # 故障した瞬間に目標が「その場」で固定される (直後の newly の処理) ので、
        # 水平方向の移動中に故障すると、そこから減速して戻るタスクになる
        if ec.get("translate", {}).get("enabled", False):
            self.target[~self.faulted, :2] += self.v_cmd[~self.faulted, :2] * self.dt

        # 故障した瞬間: 目標を「その場の真下に降りる」に切り替える。landing_offset が有効なら
        # 無作為な方向・距離だけずらした地点にする (将来の安全着陸
        # 準備。2026-10-19)
        newly = self.fault.triggered & ~self.faulted
        if newly.any():
            self.target[newly, :2] = self.ds.base_pos[newly, :2]
            lo = ec.get("landing_offset", {})
            if lo.get("enabled", False):
                n = int(newly.sum())
                dist = rand_uniform(0.0, lo["max_dist"], n, self.device)
                heading = rand_uniform(0.0, 2 * math.pi, n, self.device)
                offset = torch.stack([dist * torch.cos(heading), dist * torch.sin(heading)], dim=1)
                self.target[newly, :2] += offset
        self.faulted[:] = self.fault.triggered

        self._compute_state()
        yaw_abs = torch.abs(self.ds.base_ang_vel[:, 2])
        self.max_yaw_fault = torch.where(live & self.faulted, torch.maximum(self.max_yaw_fault, yaw_abs),
                                         self.max_yaw_fault)

        grounded = self.ds.grounded_mask()
        tilt = self.ds.tilt_deg()
        pos = self.ds.base_pos
        out_of_bounds = (torch.norm(pos[:, :2] - self.target[:, :2], dim=1) > ec["max_xy_dist"]) | (
            pos[:, 2] > self.init_alt + ec["max_climb"]
        )
        crashed = (tilt > ec["crash_tilt_deg"]) | out_of_bounds

        # 着地の判定は接地の直前の値で行う (接地したステップは接触力で速度が変わっている)
        v_prev = self.prev_world_vel
        sc = ec["success"]
        ok = (
            (-v_prev[:, 2] < sc["max_vz"])
            & (torch.norm(v_prev[:, :2], dim=1) < sc["max_vxy"])
            & (self.prev_tilt < sc["max_tilt_deg"])
        )
        self.landed = grounded & self.faulted
        self.success = self.landed & ok
        self.landed_fail = self.landed & ~ok
        self.ground_unfaulted = grounded & ~self.faulted
        self.crashed = crashed & ~grounded
        self.out_of_bounds = out_of_bounds & ~grounded  # 評価の分類用 (crashed に含まれる)
        self.impact_speed = torch.norm(v_prev, dim=1)

        timeout = (self.episode_length_buf > self.max_episode_length) & ~grounded & ~self.crashed
        done = timeout | grounded | self.crashed

        rew = torch.zeros(self.num_envs, device=self.device)
        for name, fn in self.reward_functions.items():
            r = fn() * self.reward_scales[name]
            rew += r
            self.episode_sums[name] += torch.where(live, r, torch.zeros_like(r))
        tr = self.reward_cfg["terminal"]
        # 軽微な失敗 (land_fail) が、速度次第で重大な失敗 (crash) より悪い罰則になることがある
        # (2026-10-18
        # 下回る)。crash+1 を下限にする修正 ("floor") を 3 シードで検証したが、平均成功率は変わらない
        # まま重大な失敗の割合が一貫して増えた (10〜17% vs 修正前 1〜5%) ので不採用、
        # 下限なしの元の式 ("linear") に戻した。既定は "linear" (従来どおり)。
        # 2026-10-20(C2 の衝突エネルギー分析) を受け、floor が重大な失敗を増やした
        # 原因が「衝突速度 2.5 m/s 以降、勾配が完全に無くなる (飽和)」ことなのかを検証するため、
        # crash より常に軽いことは保つが勾配が飽和しない式 ("asymptote") を追加した
        mode = tr.get("land_fail_mode", "linear")
        if mode == "linear":
            land_fail_penalty = tr["land_fail"] - tr["land_fail_per_speed"] * self.impact_speed
        elif mode == "floor":
            land_fail_penalty = torch.clamp(
                tr["land_fail"] - tr["land_fail_per_speed"] * self.impact_speed, min=tr["crash"] + 1.0
            )
        elif mode == "asymptote":
            # crash + (land_fail - crash) * exp(-k*v)。v=0 で land_fail、v→∞ で crash に漸近するが
            # 超えない (land_fail_mode=floor と同じ「常に crash より軽い」性質)。v=0 での傾きが
            # linear 版 (-land_fail_per_speed) と揃うように k を選ぶ (k = land_fail_per_speed / (land_fail-crash))
            span = tr["land_fail"] - tr["crash"]
            k = tr["land_fail_per_speed"] / span
            land_fail_penalty = tr["crash"] + span * torch.exp(-k * self.impact_speed)
        else:
            raise ValueError(f"unknown land_fail_mode {mode}")
        # 着地目標 (landing_offset でずらした点) への接近は、毎ステップの整形報酬
        # (descent_xy/descent_xy_far) だけでは descent_vz に埋もれて学習されなかった
        #。接地の瞬間の
        # 残距離に比例する終端罰則を追加する (既定 0.0、既存の実験には影響しない)
        landing_pos_penalty = tr.get("landing_pos_penalty", 0.0)
        dist_to_target = torch.norm(pos[:, :2] - self.target[:, :2], dim=1)
        terminal = (
            tr["success"] * self.success.float()
            + land_fail_penalty * self.landed_fail.float()
            + tr["ground_unfaulted"] * self.ground_unfaulted.float()
            + tr["crash"] * self.crashed.float()
            - landing_pos_penalty * dist_to_target * self.landed.float()
        )
        return rew + terminal, done, timeout

    def _termination_stats(self, timeout):
        ended_fault = (self.reset_buf & self.faulted).sum().item()
        log = {}
        if ended_fault > 0:
            log["fault_success_rate"] = self.success.sum().item() / ended_fault
            log["fault_crash_rate"] = (self.crashed & self.faulted).sum().item() / ended_fault
            # 中間指標: 成功 0% でも進み具合が分かるように。故障後に何秒
            # 終了せずに続いたか、故障後の最大ヨー角速度、接地したものの接地時の降下速度・傾き
            ended = self.reset_buf & self.faulted
            t_since = (self.episode_length_buf - self.fault.fault_step).float() * self.dt
            for sec in (1.0, 3.0):
                log[f"fault_alive_{sec:.0f}s_rate"] = (ended & (t_since >= sec)).sum().item() / ended_fault
            log["fault_max_yaw_mean"] = self.max_yaw_fault[ended].mean().item()
            landed = ended & (self.success | self.landed_fail)
            if landed.any():
                log["fault_touchdown_vz_mean"] = (-self.prev_world_vel[landed, 2]).mean().item()
                log["fault_touchdown_tilt_mean"] = self.prev_tilt[landed].mean().item()
        ended_nominal = (self.reset_buf & ~self.faulted).sum().item()
        if ended_nominal > 0:
            log["nominal_timeout_rate"] = (timeout & ~self.faulted & ~self.crashed).sum().item() / ended_nominal
        return log

    # ----------------------------------------------------------------- state
    def _compute_state(self):
        ds = self.ds
        inv_q = inv_quat(ds.base_quat)
        self.gravity_body = transform_by_quat(self._gravity_world, inv_q)
        err_world = self.target - ds.base_pos
        err_world = torch.where(self.faulted.unsqueeze(1), err_world * torch.tensor([1.0, 1.0, 0.0], device=self.device), err_world)
        self.err_world = err_world
        self.err_body = transform_by_quat(err_world, inv_q)
        self.alt = ds.base_pos[:, 2]
        self.vz = ds.drone.get_vel()[:, 2]
        vc = self.env_cfg["descent"]
        v_ref = -torch.clamp(torch.sqrt(vc["v_land"] ** 2 + 2 * vc["a_plan"] * torch.clamp(self.alt, min=0.0)), max=vc["v_max"])
        self.v_ref = torch.where(self.faulted, v_ref, torch.zeros_like(v_ref))
        self.tilt = ds.tilt_deg()

    def _get_obs(self):
        s = self.obs_cfg["obs_scales"]
        ang_vel_s = torch.clip(self.ds.base_ang_vel * s["ang_vel"], -5, 5)
        lin_vel_s = torch.clip(self.ds.base_lin_vel * s["lin_vel"], -5, 5)
        motor_frac = self.motor.rpm / self.motor.max_rpm
        # alt_obs_max: 評価だけで使う、方策に見せる高度の上限 [m] (既定 None = 打ち切らない)。学習した初期高度 (8〜20 m)
        # より高い所からの評価で、観測が学習時の範囲を外れないようにする (2026-10-28)
        alt_obs = self.alt if self.env_cfg.get("alt_obs_max") is None else torch.clamp(self.alt, max=self.env_cfg["alt_obs_max"])

        parts = [
            self.gravity_body,
            ang_vel_s,
            lin_vel_s,
            torch.clip(self.err_body * s["pos_err"], -1, 1),
            (alt_obs * s["alt"]).unsqueeze(1),
            torch.clip(self.vz * s["lin_vel"], -5, 5).unsqueeze(1),
            (self.v_ref * s["lin_vel"]).unsqueeze(1),
            self.last_actions,
            motor_frac,
        ]

        if self.obs_cfg.get("fault_flag", True):
            fault_onehot = torch.zeros((self.num_envs, 4), device=self.device)
            rows = self.faulted.nonzero(as_tuple=False).squeeze(-1)
            if len(rows) > 0:
                fault_onehot[rows, self.fault.fault_rotor[rows]] = 1.0
            parts.append(fault_onehot)

        if self.hist_enabled:
            # C2 (stage=4): fault_flag=False のときの代わりの情報。
            # 重力方向・角速度・並進速度・モータ RPM 比を、エピソード開始からの一定間隔
            # (hist_stride ステップごと) でためておき、古い方から詰めて最新を末尾に追加する
            hist_sample = torch.cat([self.gravity_body, ang_vel_s, lin_vel_s, motor_frac], dim=-1)
            fresh = (self.episode_length_buf == 0).nonzero(as_tuple=False).squeeze(-1)
            if len(fresh) > 0:
                self.hist_buf[fresh] = hist_sample[fresh].unsqueeze(1).expand(-1, self.hist_k, -1)
            # 前回の観測から hist_stride の倍数をまたいだら 1 つ詰める (decimation>1 で
            # episode_length_buf が飛び飛びに進んでも取りこぼさない。decimation=1 では従来と同じ)
            prev_len = self.episode_length_buf - self.decimation
            push = (self.episode_length_buf > 0) & (
                torch.div(self.episode_length_buf, self.hist_stride, rounding_mode="floor")
                > torch.div(prev_len, self.hist_stride, rounding_mode="floor"))
            push = push.nonzero(as_tuple=False).squeeze(-1)
            if len(push) > 0:
                self.hist_buf[push] = torch.cat([self.hist_buf[push, 1:], hist_sample[push].unsqueeze(1)], dim=1)
            parts.append(self.hist_buf.reshape(self.num_envs, -1))

        obs = torch.cat(parts, dim=-1)
        if self.obs_noise_std > 0.0:
            # B2: センサノイズを模した加算ガウス雑音。観測はすでに
            # おおよそ [-1, 1] に揃うようスケールしてあるので、全チャンネルに同じ std を使う
            obs = obs + torch.randn_like(obs) * self.obs_noise_std
        return TensorDict({"policy": obs}, batch_size=[self.num_envs])

    def get_observations(self):
        return self._get_obs()

    # ----------------------------------------------------------------- reset
    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return
        n = len(envs_idx)
        ec = self.env_cfg
        alt = rand_uniform(*ec["init_alt_range"], n, self.device)
        self.init_alt[envs_idx] = alt
        self.ds.base_init_pos[envs_idx] = torch.stack([torch.zeros_like(alt), torch.zeros_like(alt), alt], dim=1)
        self.ds.reset_idx(envs_idx)  # 速度をゼロに戻す (zero_all_dofs_velocity を含む)
        self.motor.reset_idx(envs_idx)
        self.fault.reset_idx(envs_idx)

        # C4: 故障前に水平移動させておく (ds.reset_idx の後。速度をゼロに戻した直後に上書きする)
        tc = ec.get("translate", {})
        self.v_cmd[envs_idx] = 0.0
        if tc.get("enabled", False):
            speed = rand_uniform(*tc["speed_range"], n, self.device)
            heading = rand_uniform(0.0, 2 * math.pi, n, self.device)
            self.v_cmd[envs_idx, 0] = speed * torch.cos(heading)
            self.v_cmd[envs_idx, 1] = speed * torch.sin(heading)
            vel6 = torch.zeros((n, 6), device=self.device)
            vel6[:, :2] = self.v_cmd[envs_idx, :2]
            self.ds.drone.set_dofs_velocity(vel6, envs_idx=envs_idx)

        self.target[envs_idx] = self.ds.base_init_pos[envs_idx]
        self.faulted[envs_idx] = False
        self.max_yaw_fault[envs_idx] = 0.0
        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        ep = {}
        for key in self.episode_sums:
            ep["rew_" + key] = torch.mean(self.episode_sums[key][envs_idx]).item() / ec["episode_length_s"]
            self.episode_sums[key][envs_idx] = 0.0
        self.extras["episode"] = ep

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self._compute_state()
        return self._get_obs()

    # --------------------------------------------------------------- rewards
    def _reward_hover_pos(self):
        r = torch.exp(-(torch.norm(self.err_world, dim=1) / 0.3) ** 2)
        return torch.where(self.faulted, torch.zeros_like(r), r)

    def _reward_hover_upright(self):
        r = torch.cos(torch.deg2rad(self.tilt))
        return torch.where(self.faulted, torch.zeros_like(r), r)

    def _reward_hover_angvel(self):
        r = torch.sum(self.ds.base_ang_vel**2, dim=1)
        return torch.where(self.faulted, torch.zeros_like(r), r)

    def _reward_descent_vz(self):
        r = torch.exp(-(((self.vz - self.v_ref) / 0.3) ** 2))
        return torch.where(self.faulted, r, torch.zeros_like(r))

    def _reward_descent_xy(self):
        r = torch.exp(-(torch.norm(self.err_world[:, :2], dim=1) / 1.0) ** 2)
        return torch.where(self.faulted, r, torch.zeros_like(r))

    def _reward_descent_xy_far(self):
        """descent_xy (幅 1 m) は 2〜3 m 以上離れると勾配がほぼ無い。幅を広げて遠くでも引き戻す
        (2026-10-19 に 5 m で導入。2026-10-20
        学習されなかったため、`landing_offset.max_dist` (既定 8 m) まで勾配が残るよう幅を変更可能にした)。"""
        scale = self.env_cfg.get("descent_xy_far_scale", 5.0)
        r = torch.exp(-(torch.norm(self.err_world[:, :2], dim=1) / scale) ** 2)
        return torch.where(self.faulted, r, torch.zeros_like(r))

    def _reward_descent_tilt(self):
        r = torch.clamp(self.tilt - 30.0, min=0.0) / 60.0
        return torch.where(self.faulted, r, torch.zeros_like(r))

    def _reward_descent_rp_rate(self):
        r = torch.sum(self.ds.base_ang_vel[:, :2] ** 2, dim=1)
        return torch.where(self.faulted, r, torch.zeros_like(r))

    def _reward_descent_yaw_excess(self):
        """ヨー角速度が yaw_free を超えた分 ((超過 / yaw_free)^2)。relaxed hover の 35〜50 rad/s は罰さない。"""
        free = self.reward_cfg.get("yaw_free", 60.0)
        r = (torch.clamp(self.ds.base_ang_vel[:, 2].abs() - free, min=0.0) / free) ** 2
        return torch.where(self.faulted, r, torch.zeros_like(r))

    def _reward_descent_yaw_rate_cont(self):
        """ヨー角速度を故障直後から連続的に罰する (閾値なし。2026-10-21、ユーザー提案)。

        `descent_yaw_excess` は yaw_free (既定 60 rad/s) を超えるまで罰則が無く、racer の暴走
        (6p 節、ヨー角速度が 0→59 rad/s まで増え続けたが閾値にほぼ達しなかった) では発動しなかった。
        正帰還が大きくなる前に抑えるための早期信号。既定の重み 0.0 (既存の実験には影響しない)。
        """
        r = (self.ds.base_ang_vel[:, 2] / 60.0) ** 2
        return torch.where(self.faulted, r, torch.zeros_like(r))

    def _reward_descent_climb(self):
        """上昇速度 (vz > 0) を故障直後から連続的に罰する (2026-10-21、ユーザー提案)。

        目標降下速度 v_ref は常に負 (下向き) なので、`descent_vz` (v_ref との近さ) だけでは
        「上昇している」ことを積極的には罰さない (的から遠いだけで報酬が 0 に近づくだけ)。
        高度の誤差 (結果) ではなく上昇速度 (先行指標) を直接罰することで、racer の暴走上昇
        (6p 節) を早期に止める狙い。既定の重み 0.0。
        """
        r = torch.clamp(self.vz, min=0.0) ** 2
        return torch.where(self.faulted, r, torch.zeros_like(r))

    def _reward_descent_speed_mag(self):
        """水平速度の大きさを故障直後から連続的に罰する (2026-10-21、ユーザー提案)。

        `descent_xy`/`descent_xy_far` は着地目標との距離 (結果) を見ているが、水平速度
        そのもの (先行指標) は直接罰していない。racer の暴走では水平速度が 0→16.4 m/s まで
        増え続けた (6p 節)。既定の重み 0.0。
        """
        r = torch.sum(self.ds.drone.get_vel()[:, :2] ** 2, dim=1)
        return torch.where(self.faulted, r, torch.zeros_like(r))

    def _reward_descent_action_sat(self):
        """行動が ±1 付近に張り付く (飽和) ことを故障直後に罰する (2026-10-21、ユーザー提案)。

        補正→過剰反応→さらに大きな補正、という正帰還を、行動の飽和自体を罰することで
        止める狙い。`smooth` (行動の変化量) とは別の軸 (変化が無くても飽和したままなら
        `smooth` は働かない)。既定の重み 0.0。

        2026-10-26 修正 (第三者の分析で発覚): 以前はクリップ後の行動 (|a| ≤ 1) に掛けていたため
        1 次元あたり最大 0.01 にしかならず、実質効いていなかった (この修正より前の run の `action_sat_weight` は
        ほぼ無効)。クリップ前の行動で測り、超過分を 1 で頭打ちにする (std≈50 の方策でも値が爆発しないように)。
        """
        r = torch.mean(torch.clamp(self.raw_actions.abs() - 0.9, min=0.0, max=1.0) ** 2, dim=1)
        return torch.where(self.faulted, r, torch.zeros_like(r))

    def _reward_touchdown_calm(self):
        """接地直前 (高度が touchdown_calm_alt を下回る領域) ほど、ヨー角速度を強く罰する (案 3)。

        A3 の仮説 (未検証): 対角推力による傾きは機体座標系にほぼ固定され、
        ヨー回転につれて世界座標系で向きを変え続けるため、水平速度が位相に依存して振動し
        land_fail_vxy の原因になっているのではないか。接地直前だけ回転を落とせば、
        この振動が止まった状態で接地できるはず、という設計。
        """
        alt0 = self.reward_cfg.get("touchdown_calm_alt", 3.0)
        ramp = torch.clamp(1.0 - self.alt / alt0, min=0.0, max=1.0)
        free = self.reward_cfg.get("touchdown_calm_yaw_free", 10.0)
        excess = torch.clamp(self.ds.base_ang_vel[:, 2].abs() - free, min=0.0) / free
        r = ramp * excess**2
        return torch.where(self.faulted, r, torch.zeros_like(r))

    def _reward_smooth(self):
        # 行動の変化は方策のステップごとに 1 回だけ数える (decimation>1 の 2 回目以降の物理ステップは
        # 同じ行動を保持しているだけ)
        if self._substep_idx > 0:
            return torch.zeros(self.num_envs, device=self.device)
        return torch.sum((self.actions - self.last_actions) ** 2, dim=1)
