# 单瓶抓取实验（本地分支）

此目录用于在重建场景中验证抓取。`main` 分支保持静态场景内容，实验分支不自动推送。

第一轮执行结果和已知限制见 [RESULTS.md](RESULTS.md)。

后续的自动示范采集、数据检查、动作专家试训与配对评价流程见 [TRAINING.md](TRAINING.md)。

场景参数复用 `simulation/scene_config.json`。模型只接收头部、右腕 RGB 和机器人自身状态；物体真值只用于评价。底盘、升降、头部与左臂固定。默认右臂和夹爪均按模型输出执行；可选阶段控制接管夹爪保持并切换文字指令，额外的 `--program-lift` 可明确启用程序相对抬升。这两种组合控制分别记录，不使用物体吸附或抓取焊接。

## 准备

从仓库根目录获取机器人资产，并在独立 Python 环境中安装 `requirements-sim.txt`：

```bash
python3 scripts/bootstrap_sources.py
python3 -m pip install -r experiments/grasp/requirements-sim.txt
python3 experiments/grasp/prepare_scene.py
```

准备脚本复用静态场景生成器，只在忽略目录中增加实验控制器和夹爪接触模型。原始 URDF / STL 与静态场景源码保持不变。

## 已定位的仿真问题

1. 夹爪电机占位关节的惯性约为 `1e-9 kg·m²`。在记录的模型动作下，原设置产生约 `1e10 rad/s²` 加速度并重置仿真。实验入口增加 `1e-4 kg·m²` 数值转动惯量后，同一段 32 步动作稳定回放；该值不是实物标定值。
2. 单一凸包表示弯曲手指时，会填入原网格的凹处。在完全开爪、瓶子对准时，两指的凸包各穿入瓶体约 18 mm。左右手指碰撞 STL 分别含三个和两个独立凸部件，将它们分别作为碰撞体即可保留原几何的凹处；不改视觉网格、尺寸或手指质量。

拆分前会验证每个连通部件闭合且为凸体。`generated/convex_fingers/` 缓存拆出的碰撞部件，不纳入 Git。

## 检查

```bash
python3 experiments/grasp/test_stability.py
python3 experiments/grasp/test_contact_geometry.py
python3 experiments/grasp/validate_grasp.py
python3 experiments/grasp/test_observation_contract.py
python3 experiments/grasp/test_grasp_stages.py
```

第一个检查回放曾触发重置的实际动作块，要求仿真时间连续推进；第二个检查对准的瓶体能放进完全打开的夹爪；第三个用预设轨迹验证实际抬起与保持；第四个检查头部状态顺序、三轴小幅移动和夹爪开合；第五个检查阶段切换不会被空爪、单侧受力或短暂受阻触发。各检查均用退出状态报告是否通过。

`validate_grasp.py` 使用预设轨迹和物体真值检查夹爪接触，结果只用于验证环境，不能算模型成功。`probe_contacts.py` 与 `replay_trace.py` 为保留的诊断工具。

## 模型入口

模型代码采用固定的官方 Wall-X 版本 `ce2a1ae31d9ee1028b97ca40e3f66034cf8c0e5f` 和 Wall-OSS-0.5 权重。安装模型依赖、准备源码及权重后，显式传入本机路径：

```bash
python3 experiments/grasp/patch_wallx_runtime.py "$WALLX_SOURCE"
python3 experiments/grasp/grasp_walloss.py \
  --source "$WALLX_SOURCE" --checkpoint "$MODEL_DIR" \
  --output experiments/grasp/runs/blue_seed0
```

加载器检查所有模型参数已加载，复用官方提示词、归一化、动作生成与动作序列化；使用原生 SDPA，保留 checkpoint 的原始视觉 MLP 宽度。未训练或修改 checkpoint。

末端坐标原点及工具点可通过 `--state-origin` 和 `--tcp` 显式设置，用于接口假设对比。真实 SDK 的坐标标定尚未确认，各次结果必须保留所用配置。

成功判据为：目标瓶底离托盘至少 5 cm，同时保持双指接触至少 1 秒。数值异常会立即终止并标记为无效执行。录像、权重、运行输出与依赖缓存均保存在忽略目录。

## 单瓶对照

可以保持原有桌子、托盘、机器人和蓝瓶的位置，仅移除其他三瓶，生成独立的对照场景：

```bash
python3 experiments/grasp/prepare_scene.py \
  --only-bottle blue --output experiments/grasp/generated/blue_only
python3 experiments/grasp/grasp_walloss.py \
  --source "$WALLX_SOURCE" --checkpoint "$MODEL_DIR" \
  --scene experiments/grasp/generated/blue_only/scene.xml \
  --config experiments/grasp/generated/blue_only/scene_config.json \
  --tcp grasp --chunks 48 \
  --output experiments/grasp/runs/blue_only_head_fixed_seed0
```

派生配置与场景一并生成，不更改主分支场景。头部观测遵循官方 SDK 的 `[pitch, yaw]` 顺序，使用仿真实测关节角。

## 执行对照参数

- `--tcp flange|grasp`：分别解释为法兰点或假定抓取中心；真实 SDK 标定仍待核实。
- `--smooth-actions`：开启官方运行时的动作平滑，含夹爪；不修改权重。
- `--ready-pose .10 0 .25`：改用相对各自臂基座的固定起始手位，改善这一场景的手腕视野；不使用物体位置生成起始动作。
- `--staged-grasp`：当模型要求闭合，且两指开口仍大于 1.8 电机单位、闭合方向受力各超过 1 N 并持续至少 0.12 s 时，保留模型最后给出的闭合指令，并请求模型抬起蓝瓶。输入仅为两指位置、驱动力和模型命令，不读取物体坐标或接触体身份；程序不会生成抬起轨迹。这是组合控制实验，不能计为模型独立完成。阈值只适用于当前未标定仿真，双侧受阻也可能来自非目标物。
- `--program-lift`：配合 `--staged-grasp`，在触发保持后，用程序相对当时的实测机械臂位置垂直抬升 12 cm，保持姿态及夹爪闭合。此时抬升轨迹来自程序，结果明确标记为 `model_grasp_with_program_lift`，`model_only_success` 为假。不会用物体坐标选择或调整轨迹。

`telemetry.json` 保留实际执行指令、模型原始指令、目标和实际工具点、跟踪误差、夹爪自身反馈及评价指标。组合控制运行另保存 `stage_events`，便于复核何时发生阶段切换。

实时模型抓取与程序抬升的组合入口：

```bash
python3 experiments/grasp/grasp_walloss.py \
  --source "$WALLX_SOURCE" --checkpoint "$MODEL_DIR" \
  --scene experiments/grasp/generated/blue_only/scene.xml \
  --config experiments/grasp/generated/blue_only/scene_config.json \
  --tcp grasp --ready-pose .10 0 .25 \
  --staged-grasp --program-lift --chunks 48 \
  --output experiments/grasp/runs/blue_only_hybrid_program_lift_seed0
```

`validate_policy_grasp.py` 另提供“记录的模型抓取动作回放＋程序相对抬升”的诊断对照，用于检查已形成的接触是否能承重。它不运行实时模型，其成功不能算模型独立抓取成功。

## 浏览脚本专家示范

`build_demo_gallery.py` 从已有数据归档生成离线浏览页、32 条独立录像及同屏总览。输入目录需包含 `expert_dataset_v1.tar.gz`（清单与轨迹）和 `expert_videos_v1.tar`（各条原始 `rollout.mp4`）。只需 Python 标准库和 FFmpeg；原始归档不会被修改。

```bash
python3 experiments/grasp/build_demo_gallery.py
python3 experiments/grasp/build_demo_gallery.py --serve
```

浏览地址为 `http://127.0.0.1:8843/`，也可直接打开生成的 `experiments/grasp/runs/demo_gallery_v1/index.html`。页面支持逐条播放、三种相机视角、训练候选/留出分组，以及点击瓶位散点切换录像；预览服务支持视频拖动定位。

编号 01–08 来自首批示范，09–24 来自训练补充批次，25–32 来自独立留出批次。前 24 条是训练候选池，官方数据加载器还会进行内部分割。32 条均由读取物体真实位置的脚本专家完成，不能计为 Wall-OSS 模型成功。每条的展示指标直接读取原始清单和遥测，物体位置不作为模型观测输入。

原始录像按 20 fps 编码，而数据以 25 Hz 采集；浏览版本调整至 25 fps，含首尾画面共 152 帧，约 6.08 秒。所有生成数据和媒体都在被 Git 忽略的 `runs/` 中，仅浏览器生成器与页面模板保存在本地实验分支。
