"""Genesis シーンのセットアップと、PID/RL 両方から使う状態の取得・接地判定。

接地判定は方法 A (Drone エンティティのまま collision=True) で確定済み
(2026-10-15 に検証)。

空気力学の効果は Genesis に無いので自前で入れる (aero=True。2026-10-17、
Genesis 単体ではヨー回転がほぼ減衰しないことを確認)。
各項:
  (1) 対気回転数: 機体がヨー回転すると、プロペラの空気に対する回転数が変わる。
      Genesis には機体に対する回転数ではなく対気回転数を渡す (推力・反トルクの両方に効く)
  (2) H-force によるヨー減衰: アーム先端のプロペラが横に振り回されて受ける抵抗。
      cf2x.urdf の drag_coeff_xy を使い tau_z = -C_xy * L^2 * sum(対気角速度) * r
  (3) 機体の形状抵抗: tau = -k * |omega| * omega。k は形状からの推定値 (仮定)
  (4) 並進抵抗: ロータ起因。cf2x.urdf の drag_coeff_xy/z を gym-pybullet-drones と
      同じ式 F_body = -C * sum(対気角速度) * v_body で使う
"""
import math

import torch
import genesis as gs
from genesis.utils.geom import inv_quat, transform_by_quat, quat_to_xyz


def _rand_uniform(lo, hi, n, device):
    return (hi - lo) * torch.rand(n, device=device) + lo


class DroneScene:
    def __init__(
        self,
        num_envs: int,
        dt: float = 0.01,
        base_init_pos=(0.0, 0.0, 1.0),
        show_viewer: bool = False,
        visualize_camera: bool = False,
        cam_offset=(0.5, 0.5, 0.2),
        cam_fov: float = 40.0,
        cam_mode: str = "follow",  # "follow": 機体に完全追従 (姿勢の detail 用) / "static": 固定カメラ (地面と落下距離が分かる)
        cam_static_pos=(15.0, 15.0, 10.0),
        cam_static_lookat=(0.0, 0.0, 8.0),
        cam_static_fov: float = 50.0,
        aero: bool = False,
        aero_scale: float = 1.0,
        form_drag_k: float = 2.6e-8,
        rot_drag_coeff: float = 0.0,
        aero_scale_range=None,
        mass_scale_range=None,
        drone_file: str = "urdf/drones/cf2x.urdf",
        drone_arm: float = 0.0397,  # [m]。Genesis は URDF の <properties arm=...> を属性として公開しないので自分で渡す
        drone_lin_drag_coeff=(9.1785e-7, 9.1785e-7, 10.311e-7),  # [x, y, z]。cf2x.urdf の drag_coeff_xy/z
    ):
        self.num_envs = num_envs
        self.dt = dt
        self.device = gs.device
        # B2 ドメインランダム化: env ごとに空気抵抗・質量を引き直す。
        # 質量の per-env 設定には batch_links_info=True が必要 (Genesis の RigidEntity.set_mass)
        self.aero_scale_range = aero_scale_range
        self.mass_scale_range = mass_scale_range

        # scene.build(n_envs=...) で並列化すると、vis_options で指定した env しか
        # 描画されない (指定しないと何も描画されず、無地の背景色だけが返る。
        # 2026-10-16 に確認)。可視化が要るのは基本的に env 0 だけなので、
        # それ以外は描画対象から外して軽くする (hover_env.py の rendered_env_num と同じ考え方)。
        rendered_env_num = min(num_envs, 10)
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=dt),
            rigid_options=gs.options.RigidOptions(
                dt=dt / 2, enable_collision=True, batch_links_info=mass_scale_range is not None
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(rendered_env_num))),
            show_viewer=show_viewer,
        )
        self.plane = self.scene.add_entity(gs.morphs.Plane())
        # pos をモーフ生成時に渡す (reset_idx の set_pos だけに任せない)。
        # follow_entity は scene.build() 前 (= まだ原点にいる時点) のエンティティ
        # 位置を基準にカメラのオフセットを固定するため、ここで渡さないと
        # 後で set_pos しても大きくずれたカメラになる (2026-10-16 に確認)。
        self.drone = self.scene.add_entity(
            morph=gs.morphs.Drone(file=drone_file, collision=True, pos=base_init_pos)
        )

        # 第三者視点の観測用カメラ (ドローン搭載ではない。シーンを外から見る)。
        # 機体は実機サイズ (アーム長 4cm 程度) で、ホバリング高度を上げても大きさは
        # 変わらないため、固定位置のカメラでは遠すぎて数ピクセルにしかならない
        # (2026-10-16 に 20 m ホバリングで確認)。follow_entity で追従させ、
        # 機体からの相対距離を一定に保つ。
        self.cam = None
        self._cam_follow = cam_mode == "follow"
        if visualize_camera:
            if cam_mode == "follow":
                # 機体に完全追従。実機サイズ (アーム長 4cm 程度) なので近距離に置く。
                # 姿勢の detail を見せる用で、地面や落下距離は画面に入らない。
                cx, cy, cz = base_init_pos
                cam_pos = (cx + cam_offset[0], cy + cam_offset[1], cz + cam_offset[2])
                self.cam = self.scene.add_camera(res=(960, 720), pos=cam_pos, lookat=base_init_pos, fov=cam_fov, GUI=False)
                # follow_entity は scene.build() より前に呼ぶ (Genesis
                # examples/rendering/follow_entity.py の作法。後だと追従が効かず
                # カメラが初期位置に固定されたまま背景だけが映る)。
                self.cam.follow_entity(self.drone, fix_orientation=True)
            elif cam_mode == "static":
                # 追従しない固定カメラ。地面が画面に入るので、落下して高度が下がる様子が分かる
                # (follow は機体中心に画面を合わせ続けるので、どれだけ落ちたかが伝わらない。2026-10-17)。
                self.cam = self.scene.add_camera(
                    res=(960, 720), pos=cam_static_pos, lookat=cam_static_lookat, fov=cam_static_fov, GUI=False
                )
            else:
                raise ValueError(f"unknown cam_mode {cam_mode}")

        self.scene.build(n_envs=num_envs)

        self.base_init_pos = torch.tensor(base_init_pos, device=self.device).expand(num_envs, 3).contiguous()
        self.base_init_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand(num_envs, 4).contiguous()

        self.base_pos = torch.zeros((num_envs, 3), device=self.device)
        self.base_quat = torch.zeros((num_envs, 4), device=self.device)
        self.base_lin_vel = torch.zeros((num_envs, 3), device=self.device)  # 機体座標系
        self.base_ang_vel = torch.zeros((num_envs, 3), device=self.device)  # 機体座標系
        self.base_euler = torch.zeros((num_envs, 3), device=self.device)  # roll, pitch, yaw [deg] (PID 用)

        self.n_rotors = self.drone.n_propellers
        # n_rotors*KF*rpm^2 = m*g (ホバリングの釣り合い)。KF は Genesis が URDF の
        # <properties kf=...> から読んで drone.KF として公開している (drone_arm は公開されないので
        # 引数で渡す必要がある、2026-10-20)
        mass0 = self.drone.get_mass()
        mass0 = float(mass0[0]) if mass0.dim() > 0 else float(mass0)
        self.hover_rpm = math.sqrt(mass0 * 9.81 / (self.n_rotors * self.drone.KF))

        self.aero = aero
        # (2)(3)(4) の抵抗係数をまとめて何倍かにする (評価・頑健性の確認用。(1) は厳密なので掛けない)。
        # (num_envs, 1) にして tau_body/f_body へブロードキャストする。aero_scale_range があれば
        # reset_idx のたびに env ごとランダムに引き直す (B2)
        self.aero_scale = torch.full((num_envs, 1), aero_scale, device=self.device)
        self.form_drag_k = form_drag_k  # [N*m*s^2/rad^2] 形状抵抗 (推定値)
        self.rot_drag_coeff = rot_drag_coeff  # [N*m*s/rad] 追加の線形回転抵抗 (感度試験用。既定 0)
        self.lin_drag_coeff = torch.tensor(drone_lin_drag_coeff, device=self.device)
        self.arm = drone_arm  # [m]
        self.spin = self.drone.propellers_spin.to(self.device).float()
        self._rpm_air = torch.full((num_envs, self.n_rotors), self.hover_rpm, device=self.device)
        self._base_link_idx = [self.drone.base_link_idx]
        if self.mass_scale_range is not None:
            self._base_mass = self.drone.get_mass()  # (num_envs,)。ランダム化前は全 env 同じ cf2x の質量

        self.reset_idx(torch.arange(num_envs, device=self.device))

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return
        self.drone.set_pos(self.base_init_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.drone.set_quat(self.base_init_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.drone.zero_all_dofs_velocity(envs_idx)
        n = len(envs_idx)
        if self.aero_scale_range is not None:
            self.aero_scale[envs_idx, 0] = _rand_uniform(*self.aero_scale_range, n, self.device)
        if self.mass_scale_range is not None:
            scale = _rand_uniform(*self.mass_scale_range, n, self.device)
            self.drone.set_mass(self._base_mass[envs_idx] * scale, envs_idx=envs_idx)
        self._refresh()

    def set_propellers_rpm(self, rpm: torch.Tensor, absent=None):
        """rpm: 機体に対するモータの回転数 (num_envs, n_rotors)。

        absent: プロペラが無い (故障で脱落した) rotor のマスク (num_envs, n_rotors)。
        aero=True のとき、機体の回転で対気回転数が生じるのはプロペラがある rotor だけなので、
        故障した rotor は対気回転数の補正後に 0 に戻す。
        """
        rpm_air = rpm
        if self.aero:
            # プロペラは自分が機体に与える反トルクと逆向きに回る (反トルクの向き = spin)。
            # 機体がヨー角速度 r で回ると、対気回転数は rpm - spin * r になる。
            r_rpm = self.base_ang_vel[:, 2:3] * (60.0 / (2.0 * math.pi))
            rpm_air = torch.clamp(rpm - self.spin * r_rpm, min=0.0)
        if absent is not None:
            rpm_air = torch.where(absent, torch.zeros_like(rpm_air), rpm_air)
        self._rpm_air = rpm_air
        self.drone.set_propellers_rpm(rpm_air)

    def _apply_drag(self):
        if not self.aero and self.rot_drag_coeff <= 0.0:
            return
        w_body = self.base_ang_vel
        tau_body = torch.zeros_like(w_body)
        force = None
        if self.rot_drag_coeff > 0.0:
            tau_body = tau_body - self.rot_drag_coeff * w_body
        if self.aero:
            omega_sum = (self._rpm_air * (2.0 * math.pi / 60.0)).sum(dim=-1, keepdim=True)  # [rad/s]
            # (2) H-force によるヨー減衰
            tau_body[:, 2:3] -= self.aero_scale * self.lin_drag_coeff[0] * self.arm**2 * omega_sum * w_body[:, 2:3]
            # (3) 形状抵抗
            tau_body = tau_body - self.aero_scale * self.form_drag_k * torch.norm(w_body, dim=-1, keepdim=True) * w_body
            # (4) 並進抵抗
            f_body = -self.aero_scale * self.lin_drag_coeff * omega_sum * self.base_lin_vel
            force = transform_by_quat(f_body, self.base_quat).unsqueeze(1)
        torque = transform_by_quat(tau_body, self.base_quat).unsqueeze(1)
        self.scene.sim.rigid_solver.apply_links_external_wrench(
            force=force, torque=torque, links_idx=self._base_link_idx, ref=gs.link_ref_frame.link_COM
        )

    def step(self):
        self._apply_drag()
        self.scene.step()
        if self.cam is not None and self._cam_follow:
            self.cam.update_following()
        self._refresh()

    def capture_frame(self, antialiasing: bool = False):
        """レポート・動画用に現在のシーンを 1 枚レンダリングする (RGB array, HWC)。

        visualize_camera=True で構築していない場合は None を返す。
        antialiasing=True は見栄えが良くなるが遅い (静止画の保存用。動画は False で十分)。
        """
        if self.cam is None:
            return None
        rgb, _, _, _ = self.cam.render(rgb=True, antialiasing=antialiasing)
        return rgb

    def _refresh(self):
        self.base_pos[:] = self.drone.get_pos()
        self.base_quat[:] = self.drone.get_quat()
        inv_q = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.drone.get_vel(), inv_q)
        self.base_ang_vel[:] = transform_by_quat(self.drone.get_ang(), inv_q)
        self.base_euler[:] = quat_to_xyz(self.base_quat, rpy=True, degrees=True)

    def tilt_deg(self) -> torch.Tensor:
        """機体 z 軸と鉛直軸 (world z) のなす角 [deg]。

        roll/pitch オイラー角ではなく、こちらで評価する:
        ヨー回転中でも定義が崩れない。
        """
        w, x, y, z = self.base_quat.unbind(-1)
        r_zz = 1.0 - 2.0 * (x * x + y * y)
        return torch.rad2deg(torch.acos(torch.clamp(r_zz, -1.0, 1.0)))

    def contact_force_mag(self) -> torch.Tensor:
        """base_link にかかる接触力の大きさ (num_envs,)。地面に当たっているかの目安。"""
        force = self.drone.get_links_net_contact_force()  # (num_envs, n_links, 3)
        base_force = force[:, 0, :]
        return torch.norm(base_force, dim=-1)

    def grounded_mask(self, contact_eps: float = 0.01, alt_threshold: float = 0.05) -> torch.Tensor:
        """接地したとみなす env のマスク。

        高高度からの落下は 1 ステップの移動量が地面の薄い当たり判定を超えて
        「貫通」することがある (2026-10-16, 20 m からの自由落下で確認。1 m では
        発生しなかった)。接触力だけに頼らず、高度でも判定する。
        """
        return (self.contact_force_mag() > contact_eps) | (self.base_pos[:, 2] < alt_threshold)
