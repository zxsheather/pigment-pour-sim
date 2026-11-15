# 持瓶移动：从仿真到 Wall-OSS 微调

本阶段负责已经夹住瓶子之后的移动。先建立能执行、能判断失败、能导出示范的环境，再测预训练模型，再微调。抓取与倒液体分别由另外两个子任务接入。

## 2026-11-15 第一轮已验证结果

| 环节 | 实际结果 | 能说明什么 |
|---|---|---|
| 程序教师 | 2 方向 × 4 颜色 × 3 起点，共 24/24 成功；31.85–39.2 秒到位，位置误差 2.05–2.10 cm，最大滑移 5.63 mm | 当前固定布局的小扰动搬运基线可用 |
| 失败检测 | 原地不动→超时；松爪→失去夹持；前进撞桌→碰撞 | 评估器不会把这些失败记为成功 |
| 数据 | 4 条录制示范，2,572 帧，LeRobot v3 导出后重新读取成功 | 图像、状态、动作和时间戳对齐 |
| 官方数据管线 | action `[1,32,26]`，state `[1,1,26]`，底盘维度启用、padding 屏蔽，数值有限 | 没有漏掉移动动作或重复转换状态 |
| 预训练模型 | 权重 SHA256 一致；实际仿真输入推理成功；一条 seed=100 的闭环试跑 75 秒超时，距目标 1.81 m | 当前适配下该条试跑未完成任务，不代表模型总体成功率 |
| 微调 | 20 次动作专家参数更新，loss/梯度有限，每步约 1 秒；检查点保存并重载推理成功 | 训练管线已跑通，还没有证明搬运能力提升 |

Thor 工作目录为 `/data/hlou/pigment-pour-study`。示范在 `datasets/carry-smoke-v1`，训练日志与检查点在 `runs/carry-smoke-v1`，已保存检查点为 `runs/carry-smoke-v1/checkpoints/0_19`。零样本试跑在 `runs/wall-zero-shot-v1`，重载验证在 `runs/wall-ft-probe-v1`。动作头权重最大变化约 `1.84e-4`，抽查的冻结视觉权重变化为 0；未据此推断成功率提升。

## 平滑搬运与视频优化

默认教师改为 `smooth`：在每段路线内固定前进/倒车选择，按朝向误差连续调节速度，临近航点时减速，并将航点容差从 2.5 cm 收紧到 1.2 cm。保留 `--teacher stop-turn` 复现原控制器。

原来的 24 组测试全部通过。另用未参与调参的 seed 100–109、两个方向、起点 x/y ±5 cm、朝向 ±10°，对原版与新版做 20 对匹配测试：

| 指标 | 原版 stop-turn | 新版 smooth |
|---|---:|---:|
| 成功 | 20/20 | 20/20 |
| 平均完成时间 | 34.21 s | 29.19 s |
| 平均到位误差 | 2.07 cm | 1.01 cm |
| 最大夹持滑移 | 5.59 mm | 6.74 mm |

平均用时减少 14.7%，代价是最大滑移增加约 1.15 mm；这是当前接触模型和固定路线下的结果。数据见 `simulation/generated/carry-compare-heldout-v1/summary.json`，不属于 Wall 模型成绩。

新增 `simulation/make_carry_video.py`，将全景、腕部视角与逐帧速度/误差/滑移/接触状态组合为中文标注视频。标注只用于演示，不进入策略训练图像。新示范另保存对齐的 `metrics` 数组及 `metrics_columns`。

```bash
# 在 Thor 工作根目录，录制并生成容易观看的 2 倍速视频。
cd pigment-pour-sim
MUJOCO_GL=egl ../.venv-sim/bin/python simulation/run_carry.py \
  --record --image-size 640 --seed 100 --start-jitter .05 --heading-jitter-deg 10 \
  --output simulation/generated/my-smooth-demo
../.venv-sim/bin/python simulation/make_carry_video.py \
  --episode simulation/generated/my-smooth-demo/episode_0000 \
  --output simulation/generated/my-smooth-demo.mp4 --speed 2

# 复现独立对比。
../.venv-sim/bin/python simulation/validate_carry.py --compare --colors blue \
  --seed 100 --seeds 10 --start-jitter .05 --heading-jitter-deg 10 \
  --output simulation/generated/my-comparison
```

本机仓库根目录已有 `搬运演示_优化版.mp4`，可用 Finder 双击或在终端运行 `open 搬运演示_优化版.mp4`。这是本地生成文件，不提交到 Git。

## 已实现的任务

`simulation/carry_env.py` 在生成的静态场景上添加执行器、夹爪接触垫、任务重置、20 Hz 控制接口和终止条件。修改只作用于内存中的运行模型。

- 起点：重置时将指定颜色的瓶子放在右夹爪内，闭合并等待接触稳定。这里不验证抓取动作。
- 控制：输入 `[前进速度 m/s, 偏航角速度 rad/s]`；双臂、升降和夹爪由关节伺服保持搬运姿态。允许倒车，速度限幅为 ±0.18 m/s、±0.6 rad/s，另有限加速度。
- 物理：瓶子是自由刚体，两指摩擦夹持，没有 weld、等式约束或运行时逐帧绑定。液体只作为负载质量，不模拟自由液面。
- 底盘：平面关节速度伺服；轮子按命令转动，但轮胎与地面的牵引力尚未建模。重力补偿、接触参数也需真机校准。
- 成功：到目标停靠位距离 <4 cm、朝向误差 <0.06 rad，低速停稳至少 0.5 秒。到位指底盘停靠，不代表瓶口已经对准量筒。
- 失败：机器人/瓶子撞场景、瓶子相对夹爪滑动超过 4 cm、双侧接触持续丢失、瓶身倾斜超过 0.45 rad、数值异常或 75 秒超时。

`WaypointTeacher` 使用仿真的真实底盘位姿和预设路线，是采集示范的程序控制器。它的成功率不能记为 Wall-OSS 的成功率。颜色改变视觉和任务文字，四个瓶子的物理参数相同。

## 先运行，不需要真机

在仓库根目录，按 README 获取机器人资产、安装依赖并生成 `scene.xml` 后：

```bash
.venv/bin/python simulation/run_carry.py --color blue --goal B --record
.venv/bin/python simulation/validate_carry.py --output simulation/generated/validation-my-run
```

无显示器的 Linux 服务器在命令前加 `MUJOCO_GL=egl`。macOS 本地可直接离屏渲染；本机 CPU 足以运行这个脚本控制基线。模型推理和微调放到 GPU 服务器。`--goal A` 是返回取瓶侧，`--start-jitter .03` 将起点 x/y 各扰动 ±3 cm。

每条示范生成：

```text
episode_0000/
  trajectory.npz           # state/action/base_pose/bottle_position/timestamp
  metadata.json            # 指令、终止结果、参数、脚本 SHA256
  face_view.mp4            # 头部相机
  left_wrist_view.mp4
  right_wrist_view.mp4
  overview.mp4             # 仅供观察的全局相机，不输入策略
```

每帧先记录观测 `o_t`，再执行命令 `a_t`。动作标签是实际发出的命令，不是下一帧的测量速度。只有成功且录齐三路相机的轨迹可以被导出脚本接收。

## 自己的策略接在哪里

最小接口如下；可将 `my_policy` 换成导航算法或学习模型：

```python
from carry_env import CarryEnv  # 从 simulation 目录运行

env = CarryEnv(render=True)
obs = env.reset(color="blue", goal="B", seed=100, start_jitter=0.03)
while not env.done:
    action = my_policy(obs["images"], obs["state"])
    obs, info = env.step(action)  # [vx, yaw_rate]
env.close()
```

观测里的 `base_pose` 和 `bottle_position` 是给程序教师、评估器用的仿真真值。视觉策略只接收图像和本体状态；不要无意中把世界坐标瓶位喂给模型，然后当作纯视觉结果。

保存的 state/action 都是 23 维：

| 索引 | 内容 | 约定 |
|---|---|---|
| 0:3 / 10:13 | 左/右末端位置 | 底盘坐标系，米 |
| 3:9 / 13:19 | 左/右末端旋转 | 旋转矩阵前两行展开，6D |
| 9 / 19 | 左/右夹爪开度 | 两指间距 / 0.0975 m |
| 20:23 | 底盘速度 | 本体坐标系 `[vx, vy, yaw_rate]` |

state 为测量量；action 的双臂部分是绝对保持目标，底盘部分为加速度限制后的命令，当前 `vy=0`。Wall-X 训练加载器按 `_relative` 配置把双臂绝对目标转为相对当前状态的量，**只转换一次**。底盘速度仍是速度命令，不做位移差分。

## Wall-OSS 与训练流程

采用 [官方 Wall-X 源码](https://github.com/X-Square-Robot/wall-x) 的固定版本 `ce2a1ae31d9ee1028b97ca40e3f66034cf8c0e5f`，以及 [Wall-OSS-0.5](https://huggingface.co/x-square-robot/wall-oss-0.5) 固定版本 `b7dc23a82dbf70a65859c9fdd596703234597028`。模型权重 SHA256 为 `37f8f327d3f9c27f056ed078db43c0e39b8caf3df0dc0163cdb8cf92c167968a`。

该版本视觉 SDPA 的 `forward` 少接收调用端传入的 `max_seqlen` 参数。使用本仓库的 `patches/wall-x-vision-sdpa.patch` 修正接口；SDPA 仍根据 `cu_seqlens` 生成掩码。进入 Wall-X 源码目录后运行 `patch -p1 < ../pigment-pour-sim/patches/wall-x-vision-sdpa.patch`，仅应用一次。

Wall-OSS 在这里的工作方式：输入当前相机图像、状态和“把蓝瓶搬到 B 托盘”指令，输出未来 32 帧动作。执行前几帧后重新拍照、重新预测。微调是用示范的观测/动作对训练这个映射；仿真本身不会自动把模型训练好。这个仓库目前先以程序教师生成示范，不涉及强化学习。

训练和推理环境与 MuJoCo 环境分开。服务器是 ARM64 Thor，需使用支持其 GPU 的 CUDA PyTorch；不要直接照搬 x86 的安装命令。现有 `.venv-wall` 使用独立依赖目录，只读复用已有 PyTorch 2.9.0+cu130。官方依赖要求与这一组合存在版本差异，因此以实际导入、数据、前后向测试结果为准；不要把环境导入成功当成模型已经验证。

在 Thor 的 `/data/hlou/pigment-pour-study` 下执行，第一次先做小规模流程测试：

```bash
# 1. 将已录制成功示范导为 LeRobot v3；输出目录必须不存在。
.venv-wall/bin/python pigment-pour-sim/simulation/export_carry_dataset.py \
  --episodes pigment-pour-sim/simulation/generated/carry-demo-B-01/episode_0000 \
             pigment-pour-sim/simulation/generated/carry-demo-A-01/episode_0000 \
             pigment-pour-sim/simulation/generated/carry-demo-A-01/episode_0001 \
             pigment-pour-sim/simulation/generated/carry-demo-green-01/episode_0000 \
  --output datasets/carry-my-data

# 2. 生成适合本任务的配置；20 步只验证训练管线。
.venv-wall/bin/python pigment-pour-sim/simulation/prepare_wall_config.py \
  --dataset datasets/carry-my-data --model models/wall-oss-0.5 \
  --output runs/carry-my-run --steps 20

# 3. 用官方统计脚本。相对动作统计与加载器采用相同转换。
.venv-wall/bin/python wall-x/scripts/compute_norm_stats.py \
  --train_config runs/carry-my-run/train.yml

# 4. 真正走一遍官方加载器和 collator；不需要模型权重。
.venv-wall/bin/python -m torch.distributed.run --nproc_per_node=1 --master_port=29876 \
  pigment-pour-sim/simulation/validate_wall_data.py \
  --config runs/carry-my-run/train.yml --output runs/carry-my-run/data-validation.json

# 5. 权重下载并校验完成后，才开始训练。
.venv-wall/bin/python -m torch.distributed.run --nproc_per_node=1 --master_port=29877 \
  wall-x/wall_x/trainer/fsdp_trainer/train_fsdp.py --config runs/carry-my-run/train.yml
```

配置保留 20 维双臂 + **3 维底盘速度**，再加 3 维无损失 padding，合计 26 维，兼容预训练动作空间。官方桌面双臂示例把尾部全部补齐，会漏掉本任务要学的移动。先冻结视觉语言主干，仅训练动作专家，batch=1、SDPA、关闭 WandB，设定明确步数上限。

官方训练日志从 step 0 开始计数，保存检查发生在步号递增之前。本配置将保存间隔设为 `steps-1`，使 20 次更新在 step 19 保存，避免到步数上限时直接退出而不留检查点。

`run_wall_carry.py` 接实际仿真图像和状态，绕过仅适用 LIBERO 单臂的状态编码。它记录模型全部输出，但只执行底盘 `vx` 与 `yaw_rate`；双臂由原保持控制器接管，预测的侧向速度不执行。所以它测的是“Wall + 持瓶控制器”的移动能力。

```bash
MUJOCO_GL=egl .venv-wall/bin/python pigment-pour-sim/simulation/run_wall_carry.py \
  --model models/wall-oss-0.5 --probe-only --output runs/wall-probe-my-run

# 去掉 --probe-only 才执行闭环搬运。
# --config 可指定微调配置，--model 指向相应检查点目录。
```

未微调时暂用 `ex_normal` 归一化，这是适配假设。仿真末端坐标、夹爪开度定义、相机安装位姿还没有对齐官方原机器人；零样本失败不能单独证明预训练模型“不会”。先测接口，再测闭环，并保留失败轨迹。

第一轮指令使用 A/B 字母，而场景里没有对应标牌。现在默认指令改为“量筒旁的托盘”或“放着其他颜色瓶子的托盘”，并明确要求保持夹持、不放下瓶子。A/B 仍用于 CLI 和评估元数据。`run_wall_carry.py --instruction` 可自定义指令，也支持与教师一致的位置/朝向扰动参数；历史零样本成绩仍对应旧指令。

首次实际推理使用两路 448 像素图像，首个 32 帧动作块约 2.26 秒；完整试跑后续约 1.6–1.9 秒/块。当前仿真在推理期间暂停，20 Hz 是动作采样频率，不代表已实现真机实时闭环。真机接入还需要异步推理、动作缓存和延迟处理。

## 与队友、真机的接口

抓取子任务输出选定瓶子、实际稳定夹持状态和可搬运姿态；搬运子任务在指定停靠位停稳，并把机器人状态、瓶子姿态交给倾倒子任务。后续串联必须延续同一个仿真状态，不能再次调用本环境的初始化来替代真实交接。

下一轮先增加起点、朝向、目标停靠位、瓶重、摩擦和视觉变化，收集几十到几百条覆盖这些变化的成功示范。这个数量是采集起步范围，不是保证成功的训练门槛。不同路线与起点必须整条 episode 留作独立闭环测试。

官方加载器会先选训练 episode，再在其中划分相邻帧作为训练/验证；日志里的验证损失不能替代独立 episode 成功率。正式实验应只用训练集计算统计；当前 4 条全量统计仅供流程测试。

真机先标定相机/末端/底盘坐标、运动限制和夹持姿态，再用相同接口采集真实遥操作。保留仿真与真机数据来源标签，用真机数据继续微调，最后在机器人上运行推理和低层控制。无需一开始就在真机上训练大模型。
