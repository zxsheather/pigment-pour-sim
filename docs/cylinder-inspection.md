# 量筒拿近与图像读数：固定官方平行手指

搬运阶段已经独立保存：Git 提交 `67c86c1`、标签 `carry-v1-20261115`；服务器快照在 `/data/hlou/pigment-pour-study/snapshots/carry-v1-20261115`，包含代码、实验数据与录像归档、大权重路径和 SHA256 清单。过程见 `docs/experiments/2026-11-15-carry.md`。

## 当前优化与 Wall 训练进度（2026-11-17）

本轮保持官方手指 STL/URDF 几何、滑动轴、行程、20 N 执行器力限不变；不新增指垫，不提高摩擦系数，不启用 NoSlip。每次成功示范及模型评测都审计实际运行环境中的手指几何。

固定配置见 `simulation/configs/cylinder_official_teacher_dev_v1.json`。改动是抓取工具点沿手指方向从 135 mm 调到 115 mm、闭合半开度命令从 10 mm 调到 3 mm；同时将 MuJoCo 时间步从 2 ms 改为 0.5 ms、`impratio` 从 1 改为 10。抓取高度仍为 225 mm，`condim=3`、摩擦 `[1,0.005,0.0001]`、`noslip_iterations=0`。闭合命令只是位置伺服目标，不会穿过刚体，实际接触仍受 20 N 力限约束。可选力控制和持续真值扶正曾试验但未被此配置采用。

调整时间步与接触阻抗属于数值模型变更，不能把全部改善归因于动作控制。MuJoCo 的 [Preventing slip 文档](https://mujoco.readthedocs.io/en/3.3.7/modeling.html#preventing-slip) 说明软接触模型的滑移及 `impratio` 的影响。本轮保留较细步长验证记录，不把模拟成功解释成真实材料已标定。

| 阶段 | 通过 | 最大滑移 | 最大读数误差 |
| --- | ---: | ---: | ---: |
| 程序教师：原 8 组 | 8/8 | 10.58 mm | 0.246 mL |
| 程序教师：额外开发 12 组，seed 500..511 | 12/12 | 15.31 mm | 0.134 mL |
| 程序教师：训练数据采集 | 44/48 | 保留逐条记录 | 4 条读数阶段失败被排除 |
| Wall 微调 256 次更新，执行前 8 帧，25 mL / seed 100 | 0/1 | 未建立抓取 | 60 s 超时 |
| 同一检查点，执行完整 16 帧 | 0/1 | 未建立抓取 | 60 s 超时 |
| 微调 Wall：原 8 组与独立 12 组 | 待验 | 待验 | 前向接口检查不计任务成功 |

两套教师验证的松爪下落、空白/遮色标/遮液面/旋转图像拒绝读数与旧搬运回归均通过。结果见 [原 8 组](experiments/2026-11-17-official-teacher-original-v1.json)、[额外 12 组](experiments/2026-11-17-official-teacher-development-v1.json)。后者用于开发，不再称为最终独立测试。

48 条采集使用 5/10/15/20/25/30/35/40 mL，各 6 个起点，xy ±8 mm、yaw ±8°。44 条成功轨迹包含三路 448px RGB、动作前的 23D 机器人状态和 23D 绝对末端动作。动作标签由实际发出的关节目标做 FK 得到，保留请求的夹爪闭合预载，不用未来实测位姿代替控制目标。视频以原 H.264 码流导出，避免第二次有损压缩；导出器仍检查时间戳、帧数和相机对齐，拒绝失败示范。

`run_wall_cylinder.py` 独立执行 Wall 的右臂、夹爪、底盘前进与转向输出；左臂保持，平台不支持的侧向速度只记录。输入只有三路 RGB 与机器人本体状态，不提供量筒位姿、任务阶段或液量，也不接入教师的真值扶正和图像高度控制。物体真值与标定 RGB 读数器只负责评分。`cylinder_actions.py` 与官方 Wall 加载器采用同一个旋转约定：前两行 6D 旋转，`R_abs = R_rel @ R_state`；相对位移只加到动作块开始时的状态，夹爪与底盘速度保持绝对值。

Wall 首轮训练采用官方 Wall-OSS 原权重、冻结视觉语言主干、只更新动作专家。模型测试原 8 组与开发 12 组后，最后使用预先固定的 seed 800..811 独立集合验收。训练损失或单次前向通过不代表动作完成。程序教师成绩、保存动作重放与真实模型闭环成绩分别记录。

首轮数据复核发现：固定版本的官方统计脚本在连接 episode 后形成相对动作窗口，会将上一段结尾与下一段起点混合。因此量筒使用 `compute_cylinder_norm_stats.py`，仅在每条训练示范内部形成窗口，末端重复最后一个动作，与 LeRobot 的实际加载行为一致。该脚本不读取测试案例。官方加载器的 `train_test_split` 还会先选择靠前的 episode；本任务全部 44 条已经是训练集合，所以明确设为 1.0，使用 `patch_wall_train_only.py` 放开官方配置校验中的上界限制，记录补丁前后 SHA256。加载器内部仍随机留出 5% 相邻帧，这类验证损失不算独立任务验证。原始统计的首轮初始化被主动停止，记录保留；修正后的训练目录为 `/data/hlou/pigment-pour-study/runs/cylinder-official-bc-v2`。

该环境的词表扩展初始化还曾造成约 75 GB RSS，尚未训练便出现内存压力；用 `patch_wall_checkpoint_resize.py` 关闭紧接着会被检查点完整覆盖的临时词向量协方差初始化，保留原始日志与补丁哈希。修复后已完成真实梯度更新。`audit_wall_checkpoint.py` 用于逐张量核对冻结部分与原权重在运行时 dtype 转换后完全一致，同时确认动作专家权重发生更新；日志中按名称是否含 `action` 归类的参数范数不等于冻结核验。

未微调原权重在相同任务归一化下的单例测试（25 mL、seed 100）未完成抓取，16.8 秒时因模型目标不可达终止，见 [闭环记录](experiments/2026-11-17-cylinder-zero-shot-single-v2.json)。原先八案例批量推理因初始化内存压力下延迟很高而中断，标为未完成评测，不计为 0/8。新评测器支持独立设置推理微批量，记录中断、代码哈希和真实执行前缀。

第一检查点 `0_255` 的 [权重核验](experiments/2026-11-17-wall-bc-v2-step255-weight-audit.json) 通过：257 个动作专家张量改变，581 个冻结张量在运行时 dtype、官方视觉 MLP 补零及已声明词嵌入共享关系下完全一致，新增补零区也逐值检查。源码中的 MLP 保存钩子没有消去子模块补零，所以不能直接把原始检查点与新文件的矩阵形状差异视为训练修改。

该模型的 [8 帧闭环](experiments/2026-11-17-wall-bc-v2-step255-prefix8.json) 和 [16 帧闭环](experiments/2026-11-17-wall-bc-v2-step255-prefix16.json) 都未抓起量筒；因此滑移 0 不是夹持稳定证据。固定教师观测的 [离线诊断](experiments/2026-11-17-wall-bc-v2-step255-imitation.json) 显示，“approach”和“present”的 16 帧平均末端位置误差分别为 37.4、37.9 mm；这不是闭环任务成绩。当前先继续训练并比较后续检查点的轨迹精度。首轮真实模型失败视频为本机 `量筒演示_Wall256步_失败.mp4`，与程序教师视频分开。

另修正推理补齐约定：训练 collator 在归一化后补三个 0，推理原先却将它们归一化成 -1。现在显式还原为 0。相同检查点、相同随机种子和上述十组教师观测的配对诊断没有发现预测数值变化，故没有将该修正当作本轮成功率改善或主要失败原因。

输入数据快照在 `/data/hlou/pigment-pour-study/snapshots/cylinder-official-bc-v2-inputs-20261117`（约 1.06 GiB，923 个文件附 SHA256）：含源码归档、实际 Wall Python 源码、48 条完整采集记录、44 条训练导出和训练配置。它冻结训练输入，模型最终结果仍以 `runs` 中的后续闭环记录为准。时间步再减半到 0.25 ms 的单例检查仍成功，滑移 8.75 mm、读数误差 0.061 mL，见 [数值检查](experiments/2026-11-17-official-dt-convergence-025-v1.json)；不是大样本真机结论。

复现程序教师开发配置：

```bash
MUJOCO_GL=egl ../.venv-sim/bin/python simulation/validate_cylinder.py \
  --controller-config simulation/configs/cylinder_official_teacher_dev_v1.json \
  --output simulation/generated/my-fixed-official-original

# 首轮实际模型训练和评测说明随检查点结果继续更新。
```

在服务器研究目录 `/data/hlou/pigment-pour-study` 中，已导出的训练集为 `datasets/cylinder-official-bc-v2`。新实验目录必须不存在；下面的配置与首轮一致。现有 Wall 源码已经应用 `patch_wall_train_only.py`、`patch_wall_checkpoint_resize.py`，补丁记录在 `results/wall-*-patch.json`；重新安装固定版本源码时先应用这两个补丁，不重复应用到已修改的文件。

```bash
.venv-wall/bin/python pigment-pour-sim/simulation/prepare_wall_config.py \
  --dataset datasets/cylinder-official-bc-v2 --model models/wall-oss-0.5 \
  --output runs/my-cylinder-bc --steps 1021 --batch-size 2 --learning-rate 3e-5 \
  --save-every 255 --horizon 16 --front-resolution 448 --train-episode-fraction 1.0
.venv-wall/bin/python pigment-pour-sim/simulation/compute_cylinder_norm_stats.py \
  --config runs/my-cylinder-bc/train.yml
PYTHONPATH=wall-x .venv-wall/bin/python -m torch.distributed.run \
  --nproc_per_node=1 --master_port=29876 pigment-pour-sim/simulation/validate_wall_data.py \
  --config runs/my-cylinder-bc/train.yml --output runs/my-cylinder-bc/data-validation.json
PYTHONPATH=wall-x .venv-wall/bin/python -m torch.distributed.run \
  --nproc_per_node=1 --master_port=29877 wall-x/wall_x/trainer/fsdp_trainer/train_fsdp.py \
  --config runs/my-cylinder-bc/train.yml

# 等该检查点保存完成，再执行；不是只做前向接口检查。
MUJOCO_GL=egl PYTHONPATH=wall-x .venv-wall/bin/python pigment-pour-sim/simulation/run_wall_cylinder.py \
  --model runs/my-cylinder-bc/checkpoints/0_255 --config runs/my-cylinder-bc/train.yml \
  --norm-key cylinder_official_v2 --policy-batch-size 1 --execute-steps 8 \
  --controller-config pigment-pour-sim/simulation/configs/cylinder_official_teacher_dev_v1.json \
  --cases pigment-pour-sim/simulation/configs/cylinder_policy_original_cases.json \
  --output runs/my-cylinder-eval-original
```

训练与评测日志、失败案例、检查点保存在研究目录下的 `results` 与 `runs`。录制图像和完整示范保存在 `simulation/generated/fixed-official-training-v1`，没有将失败案例覆盖成成功案例。推理期间仿真暂停，20 Hz 是动作标签的采样频率，不代表模型达到真机 20 Hz 实时控制。

## 几何恢复基线（2026-11-17，优化前）

按用户要求，量筒任务默认恢复官方 URDF 提供的平行手指碰撞几何，不增加自定义平面垫、V 槽或托底结构。`CylinderEnv()`、`inspect_cylinder.py` 和 `validate_cylinder.py` 的默认配置均为 `official`。旧 `flat`、`v-pad` 仅可显式选择，属于历史自定义代理模型；旧瓶子搬运实验仍保留其原来的平面代理以便复现，不以其成绩替代本节基线。

几何来自 [sources.json](../sources.json) 固定的 `X-Square-Robot/sdk_robot` 提交 `63610199bbd70eafd1fc087a015ec5c74a0c906c`。官方左/右手指碰撞 STL 分别由 3/2 个分离的凸实体组成。MuJoCo 对整份网格求一个凸包会填掉实体间空隙，因此按原有连通实体分别导入，保留原始形状、位置和尺寸；这不是另设计分段指垫。导入后与源几何的双向顶点误差小于 2 nm，检查容差为 1 µm。原始官方文件不修改。

| 项目 | 当前基线 | 依据 |
| --- | --- | --- |
| 手指几何、滑动轴、行程 | 官方原始值 | 固定版本 URDF/STL；几何审计 |
| 右手单指执行器力上限 | 20 N | URDF 滑动关节 `effort` 字段；不等于实测夹持力 |
| 摩擦参数 | `[1, 0.005, 0.0001]` | MuJoCo 名义默认值，尚未做真机标定 |
| 接触维度、抗滑后处理 | `condim=3`、`noslip_iterations=0` | 撤销上一轮额外滚动/扭转阻力和抗滑设置 |
| 指垫材料、伺服刚度/阻尼 | 待实测；控制器仍使用原名义增益 | 官方资产未提供真实接触材料标定 |
| 抓取点与动作程序 | 保留 225 mm 抓取点及原程序教师 | 本轮检验默认硬件切换，未为成功率调动作 |

这套基线以已有官方资产为依据。真机安装的指垫尺寸、材质和摩擦仍需确认；后续按实物补充参数，而非为提高成功率替换接触形状。历史 V 形改装的成功率不能作为当前默认配置的成绩。

原 8 组条件重新测试为 **0/8**，均因 `lost_grasp` 失败；原瓶子搬运代理模型的回归测试通过。由于没有成功的量筒轨迹，本轮松爪及图像负例检查明确记为跳过，不能计为通过。该结果说明当前动作程序直接迁移到这一几何/接触配置仍不稳定，不能推断真机平行夹爪无法抓取量筒。

本机视频 `量筒演示_官方平行手指基线.mp4` 保留 25 mL / seed 100 的失败试跑。旧 V 形视频继续保留，文件名及本节将其与默认基线分开。完整参数、逐组失败与源文件 SHA256 见 [本轮结果](experiments/2026-11-17-official-parallel-baseline.json)。这次同时恢复了接触默认参数和 URDF 力上限，属于整体基线切换，不能当作只改变指垫形状的严格消融。

复核默认几何、关节范围及力上限：

```bash
../.venv-sim/bin/python simulation/audit_gripper.py \
  --output simulation/generated/my-official-gripper-audit.json
```

## 历史结果：2026-11-16 的 V 形指垫改装

最终代码固定后测试；未放宽原成功判据。程序教师、V 形指垫与已校准视觉读数的实测结果如下：

| 测试 | 通过 | 最大滑移 | 最大读数误差 |
| --- | ---: | ---: | ---: |
| 原 8 组（原版为 2/8） | 8/8 | 4.87 mm | 0.185 mL |
| 扩展回归 20 组 | 20/20 | 14.32 mm | 0.232 mL |
| 新起点独立验证 12 组 | 12/12 | 5.60 mm | 0.239 mL |

原 8 组已观测最大滑移从 25.80 mm（包含失败终止）降到 4.87 mm，最终最大倾斜约 0.35°。扩展工况仍出现 14.32 mm 再定位，虽然双侧承载接触、最终静止和读数均通过，仍是后续应收紧的薄弱点，不应解释为零滑移。

扩展回归使用 5/12/22/32/40 mL、seed 200..219、xy 各 ±8 mm、刻度朝向 ±8°、摩擦与总质量各 0.8..1.2 倍。这批数据参与过开发。最终独立验证使用 6/18/28/40 mL、seed 400..411、xy 各 ±10 mm、刻度朝向 ±10°、摩擦与总质量各 0.85..1.15 倍；测试集合在最终代码运行前固定。seed 300..311 的首轮验证结果归入开发记录。

三套验证中，松爪后量筒下落约 0.82–0.87 m、承载接触归零；空白、遮住色标、遮住液面、旋转图像均拒绝读数，原瓶子搬运回归通过。小规模测试通过不能直接外推为真机成功率；抓取定位仍使用仿真真值，起点扰动未检验视觉定位误差。

本机新视频是仓库根目录 `量筒演示_稳定优化版.mp4`，38.2 秒、2 倍速，依次展示 8 mL / seed 100 与 40 mL / seed 219。两段读数分别为 7.91 和 40.04 mL。原视频及中间首轮视频保留。画面放大与标注依据保存的原始 RGB 观察，任务判分不使用 MP4 压缩图像。

代码保存为本地标签 `cylinder-inspection-stable-v2-20261116`，服务器代码、过程数据、失败记录和视频快照位于 `/data/hlou/pigment-pour-study/snapshots/cylinder-inspection-stable-v2-20261116`，清单包含 SHA256；本轮未推送 GitHub。完整结果见 [实验 JSON](experiments/2026-11-16-cylinder-stability-results.json)。

## 动作与最初原型

量筒从桌面上的自由刚体开始，右夹爪先张开、靠近、合拢，再提起、退离桌边、移到头部相机前。通过关节控制扶正量筒，用图像中的液面位置调整高度，最后读取刻度。运行期间不直接修改量筒位姿，没有 weld/等式约束；松开夹爪后量筒会落下。

首版录制示例：设置 25 mL，图像读数约 **24.9 mL**，计算误差约 **0.07 mL**；全过程约 **33.75 秒**，量筒中心距相机约 **0.65→0.38 m**，最高提起约 **11.6 cm**，最终倾斜约 **1.1°**。这些只是简化仿真中的一次完整示例，不代表真实仪器的测量精度。

首版视频保留为仓库根目录 `量筒演示_拿近与读数.mp4`。左侧显示机器人动作，右侧放大头部相机中的刻度与液面；放大只用于演示，读数算法处理原始 RGB 图像。

## 历史基线：2/8

测试了 8、15、25、35 mL，各两种起点（seed 100/101，x/y 各 ±4 mm）。

- 第一版偏保守的滑移判据下，8 组中通过 1 组。
- 改为检查双侧实际承载接触、允许有限的夹持再定位，并要求最终静止、液面与视线对齐、倾斜不超过 3°后，通过 **2/8**。通过的是 25 mL / seed 100 与 35 mL / seed 101，误差分别约 0.07、0.19 mL。
- 其余失败包括扶正时发生持续滑移、最终不够竖直、刻度色标被遮挡。失败仍被记录为失败，即使某张图能算出一个数值。
- 空白图片、遮住校准色标、旋转图片均拒绝读数。松开夹爪的负例中量筒下落约 0.74 m、双侧接触变为 0；原搬运任务的一条回归测试仍成功。
- 另试过用单个平滑凸圆柱替代分段筒壁作碰撞体，两条试跑都提前倾斜失败，未采用。对应代码和日志保留在生成目录及快照中。

首版结论是：完整链路存在成功例，但抓取与扶正不稳。该版代码、失败数据与录像以 `cylinder-inspection-v1-20261115` 保留。

## 历史改装过程（2026-11-16）

薄圆筒在平面指垫上容易滚动，扶正时的相对运动进一步变成轴向滑移。本轮将右手两侧的平面指垫改成 V 形槽：每个高度两片 ±30° 斜面，上下两层，合计八个接触几何。两侧、多点接触限制前后滚动，上下两层限制倾覆；仍由接触力和摩擦承载量筒。

抓取中心从筒底以上 215 mm 上移到 225 mm，使 40 mL 液面与手指下缘拉开距离；20/35 mL 校准标记仍位于夹爪下方。上下两层指垫仍通过实际接触承载，顶层局部超出筒口但不作挂钩或焊接。夹爪单执行器力限仍为 10 N；默认摩擦、量筒质量、标尺、运动速度和原测试起点均未改变。`--grip-profile flat` 保留平面指垫；历史几何消融采用上移前的 215 mm 抓取高度。

视觉流程每次读数失败后都允许小角度转腕寻找刻度，包括高度调整后的观察；原先只在第一张图失败时搜索。无有效色标或液面仍拒绝输出读数。对齐控制只接收 RGB 读数中的像素偏差，三次稳定读数后才验收。

第一轮扩展测试得到 19/20，失败的是 40 mL / seed 219：握持稳定，但黑色刻度遮住液面中央几列，读成 39.07 mL，误差 0.93 mL。最终读数器检查液体连通区域内更多列，寻找至少三列相邻像素支持的最高边界；对三列取中值抑制单像素噪声，边界整体高差超过 8 px 时仍拒绝读数。它使用 RGB 中实际可见的边界，不根据设定液量补值。

只修改像素边界后，完整动作在同一案例仍有 0.56 mL 偏差；增加可见性验收及转腕后仍有约 0.54 mL 偏差，说明还存在夹爪对高液面的物理遮挡。最终把抓取点上移 10 mm，同时保留“对齐后至少一半可见宽度支持同一边界”的验收。可见性不足时，通过实际关节运动转腕寻找新视角，再重新做高度对齐；粗略对齐阶段允许使用部分边缘，最终读数必须通过可见性验收。被拒绝的观察保存在 `view_rejected_*.png` 和 `view_search_history`，失败记录均保留。

验收与原 2/8 基线相同，并增加可见性验收：双侧实际承载接触；连续失去双侧接触不超过 0.2 s；夹持点相对滑移不超过 25 mm；运输倾斜不超过 0.12 rad；无超过 1 mm 的禁入碰撞；最终倾斜 ≤3°、线速度 ≤5 mm/s、角速度 ≤0.03 rad/s；液面距离画面水平中心 ≤2 px，连续三次读数极差 ≤0.4 mL，评分误差 ≤0.5 mL。没有通过放宽这些阈值取得改善。

新增参数 `--friction-scale`、`--mass-scale`、`--yaw-jitter` 用于接触和读数扰动测试；保存双指法向接触力、最大运输倾斜、接触缺失时长，以及完整最终控制量，便于重放和排查。

消融检查保留了平面指垫，只使用新的视觉搜索逻辑：8 mL / seed 100 仍在扶正时滑移失败（25.43 mm），而原先刻度遮挡失败的 35 mL / seed 100 经一次转腕搜索后通过（读数 34.81 mL，最大滑移 8.37 mm）。这支持“接触几何改善夹持，重新搜索恢复被遮挡刻度”的机制解释；不是所有改善都来自读数程序。

## 动作、视觉和简化边界

`inspect_cylinder.py` 的动作由程序教师生成：使用仿真中的量筒位姿规划抓取、纠正倾斜；用有界逆运动学求关节目标，平滑插值位置与姿态，再通过执行器推进物理仿真。该程序教师录像不代表 Wall-OSS 模型输出；模型动作须通过独立的 `run_wall_cylinder.py` 运行评测。

读数模块 `cylinder_reader.read_volume(rgb)` **只接收 RGB 图像**。量筒增加了明确的校准标记：紫色 20 mL、橙色 35 mL。程序定位两个标记及蓝色液面边界，在已知标尺上插值；不读取设置液量、量筒三维位置或仿真液面高度。预设液量只用于生成场景和评分。它是已知仪器的校准视觉基线，尚非任意量筒的 OCR/VLM 读数。

液面高度按 50 mL/220 mm 标尺生成，液体以刚体负载质量和可见填充几何近似，没有流动、晃动、折射或真实弯月面。真实量筒读数还需要处理相机标定、反光和弯月面，并按液面处的水平视线观察。[读数方法参考](https://www2.chem.wisc.edu/deptfiles/genchem/lab/labdocs/modules/gradcyl/gradread.htm)。

当前从量筒侧停靠位、右手空闲、量筒已装入指定液量开始，尚未串接前一阶段“拿着颜料瓶”的状态。后续必须增加放回颜料瓶或用另一只手抓量筒，并统一倾倒后的液量状态；不能用重置代替真实交接。历史 V 形指垫属于定制硬件改装，当前默认已恢复官方手指几何；其历史成绩不能当作默认配置的软件控制能力。机器人自碰撞、软垫变形、接触参数与液体物理仍需实测标定。

## 复现

在 Thor 的 `/data/hlou/pigment-pour-study/pigment-pour-sim` 中运行：

```bash
# 新动作的额外依赖，和旧搬运环境共用 MuJoCo。
../.venv-sim/bin/python -m pip install -r simulation/inspection_requirements.txt

# 单次完整试跑；默认官方碰撞几何，25 mL，无位置扰动。
MUJOCO_GL=egl ../.venv-sim/bin/python simulation/inspect_cylinder.py \
  --grip-profile official --volume 25 --record --output simulation/generated/my-cylinder-demo

# 只用保存的相机原图重新读数，不传入液量或三维状态。
../.venv-sim/bin/python simulation/cylinder_reader.py \
  simulation/generated/my-cylinder-demo/final_face.png \
  --output simulation/generated/my-cylinder-demo/reread.json

# 组合为中文标注视频。
../.venv-sim/bin/python simulation/make_cylinder_video.py \
  --episode simulation/generated/my-cylinder-demo \
  --output simulation/generated/my-cylinder-demo.mp4 --speed 2

# 原八组扰动、拒绝读数负例、松爪测试及搬运回归。
MUJOCO_GL=egl ../.venv-sim/bin/python simulation/validate_cylinder.py \
  --output simulation/generated/my-cylinder-validation

# 扩展回归：seed 200..219；首轮发现高液量失败后已用于修正读数器。
MUJOCO_GL=egl ../.venv-sim/bin/python simulation/validate_cylinder.py \
  --suite heldout --workers 3 --output simulation/generated/my-cylinder-expanded

# 最终独立测试：seed 400..411，6/18/28/40 mL。
MUJOCO_GL=egl ../.venv-sim/bin/python simulation/validate_cylinder.py \
  --suite fresh --workers 3 --seed-offset 100 --output simulation/generated/my-cylinder-fresh
```

输出目录必须不存在。验证只在全部案例通过且负例、搬运回归通过时返回零退出码。

单次输出包含 `summary.json`、`trajectory.npz`、`final_physics.npz`、相机原图、读数标注图，以及录制时的两路 MP4。状态、动作和视频按动作前观测对齐。`joint_command` 保留 **6 个右臂关节目标 + 夹爪半开度 + 底盘前进/转向速度** 共 9 维；新增 `action` 是 23 维绝对末端目标与底盘速度，可用 `export_carry_dataset.py` 导出。历史原型的 9D 文件不能直接作为 23D 数据导入。

服务器主要记录：

- 成功视频示例：`simulation/generated/cylinder-demo-v1`。
- 第一轮/第二轮扰动测试：`simulation/generated/cylinder-validation-v1`、`cylinder-validation-v2`。
- 抓取、IK、扶正失败尝试：`simulation/generated/cylinder-ik-*`、`cylinder-motion-*`、`cylinder-read-*`。
- 首版结果摘要：`docs/experiments/2026-11-15-cylinder-results.json`。
- 本轮最终测试：`cylinder-stable-original-v4`、`cylinder-stable-expanded-v4`、`cylinder-stable-fresh-v4`。
- 中间失败与代码版本：`cylinder-stable-*-v1/v2/v3`、`cylinder-optimization-sources`。
- 本轮完整结果：`docs/experiments/2026-11-16-cylinder-stability-results.json`。

依赖实测为 MuJoCo 3.3.7、NumPy 2.5.4、SciPy 1.18.1、OpenCV headless 5.0.0.93。逆运动学使用 [SciPy 有界最小二乘](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)，其结果仍需通过真实执行器与接触仿真验证，不能将 IK 可达等同于任务成功。

## 后续工作

以实际平行手指/指垫为准，核实尺寸、材质、闭合力和摩擦。历史 V 形指垫只作改装对照。继续基于固定几何的成功示范训练 Wall 并闭环迭代；读数端后续需要无色标刻度识别、弯月面分割和真实相机数据。
