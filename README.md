# quadrotor-fault-landing-rl

強化学習 (PPO) により、クアッドロータの**プロペラ 1 枚が完全に止まったときの軟着陸**を学習する。
故障していない通常の飛行 (ホバリング・水平移動) と、故障後の着陸を 1 つのポリシーで行う。
シミュレータは [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) 1.4.4。

## 結果

| 機体 | 結果 (2 m/s で移動中の故障、評価の乱数 5 通り) |
|---|---|
| cf2x (27 g、Crazyflie 相当) | 固定 100 条件で成功率 89% / 92% (2 シード、各 500 回)。同じ条件の PID は 0% |
| racer (830 g)、高度 10〜20 m | (i) cf2x の設定のままでは 0%。(ii) 故障直後の暴走を止める罰則・ランダム化なし・機体の左右対称のデータ増強で、4 シード全てが両側の故障を学習し 67〜79% (1500 回、空気抵抗の係数は cf2x の値の流用)。(iii) 係数を ×30・×88 にして学習し直すと 83%・61% (3000〜4500 回の 4 つのチェックポイントの平均。×1 も同じく延ばすと 82%。各 1 シード) |

**限界**: 成功は接地の柔らかさだけで判定していて、着地目標 (故障地点からずらした点) へはほとんど移動しない。
racer の空気抵抗の係数は同定値ではなく、結果はその値に強く依存する。

## 何をしているか

- プロペラが 1 枚止まると反トルクが釣り合わなくなり、機体はヨー回転を止められない。
  回転を許容したまま姿勢と降下速度を制御し、柔らかく接地させる
- **故障は知らせない**: ポリシーは故障したプロペラの番号を受け取らず、直近の観測の履歴から自分で気づく
  (`--stage 4`)
- **移動中の故障**: 水平に移動している最中 (0〜4 m/s) に故障させ、減速しながら着陸させる (`--translate`)
- 比較の基準として、同じ条件でカスケード PID を動かす (PID は故障後に毎回墜落する)
- **機体の規模**: 30 倍重い racer (`--drone racer`) で、何が壊れ、どう直すかを調べた。鏡像の関係にある故障の片側だけを
  学ぶ局所解を、機体の左右対称を使ったデータ増強 (`--symmetry`、`src/fdrone/symmetry.py`) で解消した

## Genesis サンプルからの発展点

ベースは Genesis の `examples/drone/` の `hover_env.py`・`hover_train.py`・`hover_eval.py`・`quadcopter_controller.py` (Apache-2.0)。これらを元にした
ファイルは冒頭の docstring にその旨を書いてある。追加した主なもの:

| 追加 | 内容 | 場所 |
|---|---|---|
| 物理の周期 | 400 Hz (方策は 100 Hz、`--decimation 4`) | `src/fdrone/configs.py` |
| モータモデル | 1 次遅れ (時定数 0.05 s) と推力重量比による回転数の上限 | `src/fdrone/envs/motor.py` |
| 空気力学モデル | Genesis に無いヨー回転の減衰・並進抵抗 (対気回転数・H-force・形状抵抗) | `src/fdrone/envs/drone_scene.py` |
| 故障注入 | ランダムなプロペラ・時刻で 1 枚を完全停止 | `src/fdrone/envs/fault.py` |
| 接地判定 | 接触力 + 高度 (高高度からの落下で地面をすり抜けるため) | `src/fdrone/envs/drone_scene.py` |
| 報酬 | 故障前は位置の維持、故障後は降下速度・傾き・接地時の衝突速度 | `src/fdrone/envs/fault_land_env.py` |
| 評価 | 固定条件 (高度 5 × プロペラ 4 × 故障時刻 5 + 故障なし 5 = 105 条件) | `scripts/eval_grid.py` |
| ROS 2 ノード | 学習したポリシーの推論を ROS 2 Humble のノードとして動かす (方策は `scripts/export_policy.py` で書き出す) | `ros2_ws/src/fdrone_bridge` |

## 動かし方

すべて Docker の中で動かす (ホストに Python の依存を入れない)。

| 用途 | イメージ |
|---|---|
| 動作確認・評価 (CPU) | `docker/Dockerfile` (`docker/docker-compose.yml`) |
| 学習・録画 (GPU) | `docker/Dockerfile.gpu` (`docker/docker-compose.gpu.yml`) |
| ROS 2 ノード | `docker/Dockerfile.ros2` (`docker/docker-compose.ros2.yml`) |

```bash
# CPU イメージを作る
docker build -t quadrotor-fault-landing-rl:dev -f docker/Dockerfile .

# 動作確認 (CPU で 2 iteration だけ学習する)
docker run --rm -v $(pwd):/workspace -w /workspace quadrotor-fault-landing-rl:dev \
  python3 scripts/train.py --stage 4 --cpu -b 8 --max-iterations 2 -e smoke
```

### 学習 (GPU)

cf2x の最終モデルの設定 (故障を知らせない・移動中の故障・着地目標のずらし・ドメインランダム化・方策 100 Hz):

```bash
python3 scripts/train.py --stage 4 --variant yawpen --translate --domain-rand --landing-offset \
  --decimation 4 --land-fail-mode floor --seed 1 -b 4096 --max-iterations 1500 -e d4g_c2_lffloor
```

racer (830 g) の設定 (故障直後の上昇・水平速度の罰則、ランダム化なし、高度 8〜20 m、30 s、対称性の増強):

```bash
python3 scripts/train.py --stage 4 --drone racer --translate --decimation 4 --land-fail-mode asymptote   --climb-weight -0.3 --speed-mag-weight -0.3 --init-alt-min 8 --episode-s 30 --symmetry   --seed 1 -b 4096 --max-iterations 1500 -e d4g_racer_A2c4_sym_s1
# 空気抵抗の係数を 88 倍 (形状抵抗は機体の形からの見積もり) にして学習する場合は、上に次を足す
#   --lin-drag-scale 88 --form-drag-k est
# 続きから学習を延ばす (別の -e 名にする。同じ名前だとログが消える)
#   --resume-from logs/<exp>/model_1499.pt --max-iterations 1500 -e <exp>_long
```

RTX 5090 で 1500 iteration 約 95 分 (単独)。チェックポイントは `logs/<exp>/model_<iteration>.pt`、
学習条件は `logs/<exp>/cfgs.pkl` と `run_info.txt`。

### 評価

```bash
# 固定条件 105 通り。Gaussian の方策は学習時と同じ確率的な行動 (stoch) で評価する
python3 scripts/eval_grid.py -e d4g_c2_lffloor --ckpts 1499 --speed 2.0 --action-mode stoch --eval-seeds 0,1,2,3,4 --tag s2h0_stoch
# 移動の向きを変える: --headings 0,90,180,270
# 1 シナリオを録画 (GPU)。racer は --racer-visual を足すとプロペラが回って見える (見た目だけ差し替え)
python3 scripts/eval.py -e d4g_c2_lffloor --ckpt 1499 --alt 10 --fault-rotor 0 --fault-time 2.0 \
  --speed 2.0 --action-mode stoch --record --gpu
# PID で同じ条件
python3 scripts/eval_pid_grid.py --stage 4 --speed 2.0 --decimation 4 --landing-offset   # 引数は --help
```

- 評価は**探索ノイズ込みの行動 (`--action-mode stoch`) を基準にする**。Gaussian の方策は学習中に
  std が大きくなり、ノイズなし (`det`) では学習時と別の制御器になるため
- 比較には評価の乱数を複数使う (`--eval-seeds 0,1,2,3,4`)。1 回の評価 (高度 10〜20 m の 60 条件) は約 ±12 ポイントばらつく
- 物理の確認 (学習なし): `scripts/verify/check_racer_fault_physics.py` (故障直後の回転の立ち上がり)、
  `scripts/verify/check_mirror.py` (左右の鏡映が物理と一致するか)
- 成功の判定は接地の柔らかさ (降下速度 0.5 m/s 未満・水平速度 0.5 m/s 未満・傾き 45° 未満) だけを見る。
  着地目標に届いたかは別に測る (`scripts/eval_reach_range.py`)

## 使用バージョン

| | バージョン |
|---|---|
| Python | 3.10 |
| genesis-world | 1.4.4 |
| rsl-rl-lib | 5.5.1 |
| PyTorch | 2.14.1 (GPU: CUDA 13.0 の wheel) |
| ROS 2 | Humble |

## 参考

- Genesis `examples/drone/` (hover_env / hover_train, Apache-2.0) — 環境・学習設定のベースライン
- M. W. Mueller and R. D'Andrea, "Stability and control of a quadrocopter despite the complete loss of
  one, two, or three propellers," ICRA 2014.
- gym-pybullet-drones (racer の URDF、MIT) / RotorS (AscTec Pelican のパラメータ、Apache-2.0、`assets/urdf/pelican.urdf`)
- rsl-rl (BSD-3-Clause)。対称性の増強は M. Mittal et al., ICRA 2024

## ライセンス

[Apache License 2.0](LICENSE)。Genesis のサンプル (Apache-2.0) を元にしたファイルを含む。元にしたファイルと出どころは [NOTICE](NOTICE) に書いてある。
