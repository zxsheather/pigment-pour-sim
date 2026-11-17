# 量筒拿近与图像读数：固定官方平行手指

搬运阶段已经独立保存：Git 提交 `67c86c1`、标签 `carry-v1-20261115`；服务器快照在 `/data/hlou/pigment-pour-study/snapshots/carry-v1-20261115`，包含代码、实验数据与录像归档、大权重路径和 SHA256 清单。过程见 `docs/experiments/2026-11-15-carry.md`。

## 当前优化与 Wall 训练进度（2026-11-17–18）

**2026-11-18 存储清理：** 按用户要求删除旧权重、旧优化器、重复数据导出、旧帧缓存及冗余录像。最新又删除历史 `0_510` 权重及依赖它的失败模型差分归档，共 9.04 GiB；清理期间新 `0_384` 检查点保存，项目实测 39.74 GiB、磁盘可用 188.82 GiB。[清理记录](experiments/2026-11-18-storage-cleanup.json) 列出具体删除项与保留项。当前 52 条训练数据清理前后逐文件 SHA256 相同，59 条当前／后续候选原始成功示范保留。完整权重保留官方预训练、当前父模型 `0_2040`、当前训练的暂定较好／最新检查点；后两者按评测结果轮换。训练及评测继续运行。下文多数历史导出和旧模型路径已删除，只保留其配置、日志和数值实验记录。

本轮保持官方手指 STL/URDF 几何、滑动轴、行程、20 N 执行器力限不变；不新增指垫，不提高摩擦系数，不启用 NoSlip。每次成功示范及模型评测都审计实际运行环境中的手指几何。

固定配置见 `simulation/configs/cylinder_official_teacher_dev_v1.json`。改动是抓取工具点沿手指方向从 135 mm 调到 115 mm、闭合半开度命令从 10 mm 调到 3 mm；同时将 MuJoCo 时间步从 2 ms 改为 0.5 ms、`impratio` 从 1 改为 10。抓取高度仍为 225 mm，`condim=3`、摩擦 `[1,0.005,0.0001]`、`noslip_iterations=0`。闭合命令只是位置伺服目标，不会穿过刚体，实际接触仍受 20 N 力限约束。可选力控制和持续真值扶正曾试验但未被此配置采用。

调整时间步与接触阻抗属于数值模型变更，不能把全部改善归因于动作控制。MuJoCo 的 [Preventing slip 文档](https://mujoco.readthedocs.io/en/3.3.7/modeling.html#preventing-slip) 说明软接触模型的滑移及 `impratio` 的影响。本轮保留较细步长验证记录，不把模拟成功解释成真实材料已标定。

| 阶段 | 通过 | 最大滑移 | 最大读数误差 |
| --- | ---: | ---: | ---: |
| 程序教师：原 8 组 | 8/8 | 10.58 mm | 0.246 mL |
| 程序教师：额外开发 12 组，seed 500..511 | 12/12 | 15.31 mm | 0.134 mL |
| 程序教师：训练数据采集 | 44/48 | 保留逐条记录 | 4 条读数阶段失败被排除 |
| Wall 字符串状态：256/511/766/1021 次更新，各测 25 mL / seed 100 | 各 0/1 | 均未建立有效抓取 | 超时或动作不可达 |
| 字符串状态 `0_255`，执行完整 16 帧 | 0/1 | 未建立抓取 | 60 s 超时 |
| Wall 连续状态：正式 511 次更新，25 mL / seed 100 | 0/1 | 19.73 mm | 抓起约 16 cm，48.75 s 碰桌失败 |
| Wall 连续状态：正式 1021 次更新，25 mL / seed 100 | 0/1 | 25.18 mm | 拿近、退离、扶正均达到；未停稳，83.25 s 滑移超限 |
| 恢复数据＋稳健状态／零中心动作：256 次更新，25 mL / seed 100 | 0/1 | 未建立抓取 | 10.10 s 动作不可达；继续训练下一检查点 |
| 同一训练分支：511 次更新，25 mL / seed 100 | 0/1 | 未建立抓取 | 8.80 s 动作不可达 |
| 里程计＋32 帧窗口：256 次更新，25 mL / seed 100 | 0/1 | 未建立抓取 | 16.50 s 非指部碰撞；碰撞位移不计有效抬起 |
| 同一分支：511 次更新，25 mL / seed 100 | 0/1 | 25.05 mm | 有效夹持并抬起 15.48 cm，41.10 s 滑移超限 |
| 同一检查点：重叠预测平均 4 组 | 0/1 | 10.41 mm | 25.85 s 倾斜超限，未采用 |
| 里程计分支：766 次更新，渲染缓存后完整重跑 | 0/1 | 未建立有效夹持 | 50.60 s 动作不可达；之前的运行中断另记 |
| 微调 Wall：原 8 组与独立 12 组 | 待验 | 待验 | 前向接口检查不计任务成功 |

两套教师验证的松爪下落、空白/遮色标/遮液面/旋转图像拒绝读数与旧搬运回归均通过。结果见 [原 8 组](experiments/2026-11-17-official-teacher-original-v1.json)、[额外 12 组](experiments/2026-11-17-official-teacher-development-v1.json)。后者用于开发，不再称为最终独立测试。

48 条采集使用 5/10/15/20/25/30/35/40 mL，各 6 个起点，xy ±8 mm、yaw ±8°。44 条成功轨迹包含三路 448px RGB、动作前的 23D 机器人状态和 23D 绝对末端动作。动作标签由实际发出的关节目标做 FK 得到，保留请求的夹爪闭合预载，不用未来实测位姿代替控制目标。视频以原 H.264 码流导出，避免第二次有损压缩；导出器仍检查时间戳、帧数和相机对齐，拒绝失败示范。

`run_wall_cylinder.py` 独立执行 Wall 的右臂、夹爪、底盘前进与转向输出；左臂保持，平台不支持的侧向速度只记录。输入只有三路 RGB 与机器人本体状态，不提供量筒位姿、任务阶段或液量，也不接入教师的真值扶正和图像高度控制。物体真值与标定 RGB 读数器只负责评分。`cylinder_actions.py` 与官方 Wall 加载器采用同一个旋转约定：前两行 6D 旋转，`R_abs = R_rel @ R_state`；相对位移只加到动作块开始时的状态，夹爪与底盘速度保持绝对值。

Wall 首轮训练采用官方 Wall-OSS 原权重、冻结视觉语言主干、只更新动作专家。模型测试原 8 组与开发 12 组后，最后使用预先固定的 seed 800..811 独立集合验收。训练损失或单次前向通过不代表动作完成。程序教师成绩、保存动作重放与真实模型闭环成绩分别记录。

首轮数据复核发现：固定版本的官方统计脚本在连接 episode 后形成相对动作窗口，会将上一段结尾与下一段起点混合。因此量筒使用 `compute_cylinder_norm_stats.py`，仅在每条训练示范内部形成窗口，末端重复最后一个动作，与 LeRobot 的实际加载行为一致。该脚本不读取测试案例。官方加载器的 `train_test_split` 还会先选择靠前的 episode；本任务全部 44 条已经是训练集合，所以明确设为 1.0，使用 `patch_wall_train_only.py` 放开官方配置校验中的上界限制，记录补丁前后 SHA256。加载器内部仍随机留出 5% 相邻帧，这类验证损失不算独立任务验证。原始统计的首轮初始化被主动停止，记录保留；修正后的训练目录为 `/data/hlou/pigment-pour-study/runs/cylinder-official-bc-v2`。

该环境的词表扩展初始化还曾造成约 75 GB RSS，尚未训练便出现内存压力；用 `patch_wall_checkpoint_resize.py` 关闭紧接着会被检查点完整覆盖的临时词向量协方差初始化，保留原始日志与补丁哈希。修复后已完成真实梯度更新。`audit_wall_checkpoint.py` 用于逐张量核对冻结部分与原权重在运行时 dtype 转换后完全一致，同时确认动作专家权重发生更新；日志中按名称是否含 `action` 归类的参数范数不等于冻结核验。

未微调原权重在相同任务归一化下的单例测试（25 mL、seed 100）未完成抓取，16.8 秒时因模型目标不可达终止，见 [闭环记录](experiments/2026-11-17-cylinder-zero-shot-single-v2.json)。原先八案例批量推理因初始化内存压力下延迟很高而中断，标为未完成评测，不计为 0/8。新评测器支持独立设置推理微批量，记录中断、代码哈希和真实执行前缀。

第一检查点 `0_255` 的 [权重核验](experiments/2026-11-17-wall-bc-v2-step255-weight-audit.json) 通过：257 个动作专家张量改变，581 个冻结张量在运行时 dtype、官方视觉 MLP 补零及已声明词嵌入共享关系下完全一致，新增补零区也逐值检查。源码中的 MLP 保存钩子没有消去子模块补零，所以不能直接把原始检查点与新文件的矩阵形状差异视为训练修改。

该模型的 [8 帧闭环](experiments/2026-11-17-wall-bc-v2-step255-prefix8.json) 和 [16 帧闭环](experiments/2026-11-17-wall-bc-v2-step255-prefix16.json) 都未抓起量筒；因此滑移 0 不是夹持稳定证据。固定教师观测的 [离线诊断](experiments/2026-11-17-wall-bc-v2-step255-imitation.json) 显示，“approach”和“present”的 16 帧平均末端位置误差分别为 37.4、37.9 mm；这不是闭环任务成绩。字符串状态版本完成 1021 次更新后，四个检查点仍均未建立有效抓取，见 [训练迭代记录](experiments/2026-11-17-wall-learning-progress.json)。首轮真实模型失败视频为本机 `量筒演示_Wall256步_失败.mp4`，与程序教师视频分开。

随后使用 Wall-X 已有的连续本体状态编码选项，仍提供同样的 23 个机器人状态量。原始发布检查点没有该投影，因此 `extend_wall_state_projection.py` 保留所有父检查点张量，增加一个种子固定的 `2048×52` 状态投影，共 106,496 个参数。这是基于 Wall-OSS 的输入表示微调实验，不能称为未改架构的原始预训练模型。20 次更新的管线检查确认新增投影有真实梯度更新；后续正式训练 1021 次，动作专家与状态投影共 258 个张量改变、581 个冻结张量保持一致。正式运行以字符串版本 `0_510` 加投影后的 20 步检查点为父模型，累计更新次数不能只看新目录名。

连续状态正式 `0_510` 在真实闭环中建立双侧承载夹持并抬起约 16 cm，但移近后再次下降，48.75 s 时量筒底部碰到桌面，仍失败。其 `0_255` 是非指部碰撞，不能把该次 4.3 cm 位移称为成功抬起；`0_765` 又出现动作不可达。后续训练不保证单调改善，按物理闭环结果选择检查点。真实模型视频为工作区 `量筒演示_Wall连续状态_抓起后碰桌.mp4`。

闭环还暴露出状态归一化问题：连续 `0_510` 的右腕两路旋转分量约 93%/98% 观测超出训练 1%/99% 分位范围；夹爪开度约 90% 超出范围。新 `robot_margin` 仅调整输入预处理：使用训练状态 min/max，并对位置各侧扩 30 mm、旋转分量各侧扩 0.1，最小跨度为 100 mm/0.5；开度使用物理归一化范围 `[0,1]`，底盘速度使用控制器限值。这些 q01/q99 槽位存的是明确指定的稳健边界，不能再称为经验分位数。

`recover_cylinder_demonstration.py` 将模型真实访问的 MuJoCo integration state、控制缓存、求解器 warm-start 完整恢复，只重置采集时钟，再由特权教师生成恢复示范。物理恢复的下一步状态与原环境一致，初始机器人状态误差为 0。这种示范只用于训练，绝不计为模型成功。已保存三个空夹爪恢复成功、一个起点不适合被拒绝，以及两个持筒运输恢复成功、一个较晚持筒状态倾斜失败。两个成功持筒恢复从模型仿真第 12.8/16.0 s 开始，新增滑移 3.74/3.85 mm，读数误差 0.041/0.125 mL；滑移参考从恢复起点计，不能替代原模型累计滑移。

新导出 `datasets/cylinder-recovery-v2` 保留原 44 条，追加 5 条成功恢复，共 49 条、31,040 帧；原始导出和失败记录不覆盖。`prepare_recovery_weights.py` 对恢复最初 2 s/随后 4 s/剩余帧分别加权 64/8/2，原示范为 1，预期约 40.5% 采样来自恢复示范；移除验证帧后的真实训练采样比例约 41.2%。`patch_wall_recovery_sampling.py` 仅在配置指定权重时替换训练 sampler；验证集合仍排除在抽样外，模型推理和物理评分不变。已验证随机种子复现、跨 rank 切片一致和验证帧排除。

连续 `0_1020` 的最终记录将问题进一步缩小到停止与稳定：已退离 0.65 m、量筒距头部相机 0.36 m、保持竖直，但线速度约 0.034 m/s、角速度约 0.20 rad/s，未满足静止条件。此时还未通过合格的三次连续读数。其经验底盘速度动作边界是 `[-0.1,0]`，停止对应归一化区间端点；右臂部分相对位移、旋转也严重不对称。这是持续偏移的一种可能来源，尚未做因果消融。

因此将短暂启动的 `runs/cylinder-recovery-robust-bc-v1` 标为 `superseded`，保留停止原因、日志和输入快照；它没有完成评测，不记失败成功率。后续 `runs/cylinder-recovery-centered-bc-v1` 从连续 `0_1020` 继续，使用相同 49 条数据和恢复抽样、稳健状态边界，以及新增 `--action-profile zero_centered`：相对位置零、相对旋转单位矩阵位于边界中心；半径为经验分位偏差的 1.2 倍，最小 20 mm/0.05，底盘采用对称控制限值，开度 `[0,1]`。这些是指定边界，不能叫经验分位数。[零动作校验](experiments/2026-11-17-wall-zero-action-validation.json) 确认所有静止位姿与底盘速度维度精确映射为零。批量 4、4 个数据 worker，最多 2041 次更新。数据 worker 的短测只证明加载开销减少，不代表整个训练同比加速，见 [加载短测](experiments/2026-11-17-wall-data-loading-benchmark.json)。

另提供可选 `--action-filter-seconds`，默认 0 时逐值透传模型指令；启用后，仅对机器人坐标系中的末端平移、SO(3) 旋转和底盘速度做指数低通，夹爪预载直通。它不读取物体位姿、液量或阶段，保留原始模型输出与实际执行命令。连续 `0_1020` 加 0.4 s 滤波未建立抓取，30.95 s 时动作不可达；滞后与提前后退改变了部署时的状态分布，因此未采用。两个连续检查点改成执行完整 16 帧，也都未完成任务，具体失败与前缀设置保留在 [迭代记录](experiments/2026-11-17-wall-learning-progress.json)。

继续补采“已经退离桌面并持筒”的读数恢复示范：`--start-holding --reading-only` 跳过已经完成的底盘退离，直接教移近、停稳、对齐液面。该选项仅用于特权教师采集，模型评测不会调用；起点要求双侧承载、已退离至少 0.2 m、无禁碰和离地，允许持筒高度低于原托盘，这类状态对恢复有用。从连续模型第 40 s/60 s 真实状态出发，分别 9.95/12.45 s 内成功，新增滑移 3.54/2.43 mm、读数误差 0.065/0.120 mL。后者在原横向 0.14 m 观察位置曾因色标遮挡失败，改为 0.18 m 后成功，两个尝试都保留，见 [晚期恢复记录](experiments/2026-11-17-late-reading-recoveries.json)。两条另导出为 `datasets/cylinder-recovery-v3` 的第 50/51 条，不改正在训练的 49 条输入。尚无模型完整任务通过记录。

零中心版本 `0_255` 的 [权重核验](experiments/2026-11-17-wall-centered255-weight-audit.json) 通过，258 个动作／状态投影张量改变、581 个冻结张量未变，但真实闭环仍在 10.10 s 因位姿不可达失败，未建立抓取。[教师观测离线诊断](experiments/2026-11-17-wall-centered255-imitation.json) 的退离速度误差约 0.053 m/s，仍较大。当前预处理、恢复抽样和父检查点同时发生变化，不能单独归因于某一项；继续训练并保留该失败。原 8 组是开发回归集，其中 25 mL / seed 100 已通过恢复示范参与训练，最终独立 seed 800..811 仍未使用。

下一组输入实验准备加入底盘位姿：`export_carry_dataset.py --include-base-odometry` 从原始示范的机器人 `base_x/base_y/base_yaw` 三关节坐标追加状态，动作仍为原 23 维。它是理想仿真里程计代理，未建模真实轮式里程计漂移。51 条、31,488 帧的新导出 `datasets/cylinder-recovery-odometry-v1` 已逐值检查：前 23 个状态与全部动作完全一致，新增三维与记录的机器人关节坐标一致，见 [导出检查](experiments/2026-11-17-odometry-export-validation.json)。没有物体位姿、液量或阶段标签进入观测。

`prepare_wall_config.py --base-odometry` 使用原来三个状态 padding 槽，状态投影仍为 `2048×52`，不新增参数；动作 padding 继续屏蔽。推理仅对真正的状态 padding 清零，不能误清新增里程计。按 32 帧窗口做的 [官方加载器检查](experiments/2026-11-17-odometry-data-validation.json) 已确认 26 维状态全部可见、23 维动作有效、末端补齐仍正确，恢复采样约 48.1%。`runs/cylinder-odometry-data-probe-v1` 只检查数据管线，没有模型梯度更新或任务成绩。

另测试 IK 前的可选机器人伺服限速：每个 50 ms 步长按实际机器人末端位姿限制平移／旋转增量，底盘和夹爪指令直通。默认关闭时保持原执行行为；不读取物体或阶段。该试验使用 `--cartesian-speed-limit .04 --angular-speed-limit .1`，保存原始模型动作及实际执行动作。`0_255` 加限速仍未建立抓取，32.55 s 时动作不可达，不计为成功或选用方案；成绩与原始控制器分开记录。

还提供 `--flow-noise zero` 的推理解码诊断，默认仍为官方高斯初始噪声。零噪声模式重建第一帧 flow action embedding 和后续积分初始值，再执行原神经网络的完整 10 步生成；没有手工生成动作。真实加载连续 `0_1020` 后，用完全相同 RGB／机器人状态、两个不同随机种子检查，输出逐值一致，见 [真实前向检查](experiments/2026-11-17-zero-noise-forward-probe.json)。这只证明解码确定性，不证明其优于高斯采样；闭环成绩仍需另测。

该零噪声闭环最终未建立抓取，22.95 s 时动作不可达，未采用。零中心分支 `0_510` 的权重核验仍通过，但 8.80 s 动作不可达、未建立抓取，见 [后期离线诊断](experiments/2026-11-17-wall-centered510-imitation.json)、[权重核验](experiments/2026-11-17-wall-centered510-weight-audit.json)。其后停止该训练作输入实验切换，保留日志、两个检查点和停止原因。

`runs/cylinder-odometry-bc-v1` 从零中心 `0_510` 开始实际梯度更新：51 条示范、32 帧动作窗口、26 维机器人状态、标准高斯初始噪声、执行前 8 帧；动作专家学习率 `3e-5`、状态投影 `3e-4`，批量 4、4 个数据 worker、seed 46，最多 2041 次更新。实际训练的 [数据检查](experiments/2026-11-17-odometry-formal-data-validation.json) 确认里程计可见、动作补齐不计损失。原 8 组与独立 12 组验收仍待模型候选通过单例完整动作后运行。

另检验提前中止是否仅来自 IK 的 0.002 加权残差上限：`--ik-max-residual .01` 允许在原关节范围内取最近解，记录 `norm([工具点误差(m),0.25*旋转误差(rad)])`。用同一个保存的失败状态／模型目标复现了 0.002651 残差，严格配置拒绝、最近解配置接受，关节仍在官方范围内，见 [目标复核](experiments/2026-11-17-nearest-ik-target-validation.json)。但完整闭环仍未建立抓取，10.20 s 时模型目标误差增至约 0.01171 并终止。这项执行配置未采用，默认仍为 0.002；物理碰撞、夹持滑移、静止和读数成功条件始终未变。

里程计分支输入快照在 `/data/hlou/pigment-pour-study/snapshots/cylinder-odometry-bc-v1-inputs-20261117`：约 569 MiB、204 个文件及 SHA256，包含 51 条训练导出、晚期恢复成功／失败、源码归档 `18b5700`、实际 Wall Python 源码、配置和验证记录。它记录训练输入，尚不是最终模型结果；此前原始 48 条采集和早期恢复的快照仍保留。

该分支首个检查点 `0_255` 的 [权重核验](experiments/2026-11-17-odometry255-weight-audit.json) 通过；[新增列核验](experiments/2026-11-17-odometry-column-update.json) 确认里程计状态及新 mask 列发生实际更新。闭环仍在 16.50 s 非指部碰撞失败，约 4.3 cm 的碰撞位移不是有效抬起。[离线误差](experiments/2026-11-17-odometry255-imitation.json) 使用 32 帧窗口，不能直接把它与此前 16 帧均值误差作同口径比较。继续训练并按保存的后续检查点评测。

为排查接口偏差，新增 `validate_wall_input_parity.py`：对三条完全相同的解码 RGB／机器人状态，分别运行真实训练 collator 与推理预处理，逐项比较 token、attention、图像张量、图像网格、归一化状态、两种 mask 和 MoE 路由；[结果全部逐值一致](experiments/2026-11-17-wall-input-parity.json)。该检查不加载模型权重或执行机器人动作。实际闭环记录另确认模型输入为 26 维、输出窗口为 32 帧，未复用旧模型配置。

进一步加载真实权重检查模型内部状态 token，发现上述预处理一致没有覆盖投影前的精度转换：实际训练先将状态和 mask 转成 token 的 BF16，再由 FP32 状态投影处理；原生 flow 推理直接投影 FP32 状态。[原生差异记录](experiments/2026-11-18-state-embedding-native-discrepancy.json) 在两帧中确认差异。现在 `run_wall_cylinder.py` 默认 `--state-embedding-precision training`，在模型边界补上同样的转换；`native` 保留旧路径以供对照，字符串状态模型不受影响。新脚本 `validate_wall_state_embedding.py` 使用保存的同一 RGB／机器人状态、真实训练 scatter 方法和相同随机种子检查生产推理路径：[两帧状态 token 均逐值一致](experiments/2026-11-18-state-embedding-alignment.json)，原生路径分别有 332/200 个分量不一致；同输入重复原生前向逐值一致。`0_510` 前 8 帧右末端位置变化最大约 0.52/0.70 mm。该检查不推进物理，不计任务成功。

修复精度后，同一 `0_510`、25 mL / seed 100 的完整闭环在 16.55 s 因倾斜失败，抬起仅 11.93 mm，[结果](experiments/2026-11-18-odometry510-training-precision.json) 没有达到有效抬起条件。后续 `0_1275` 使用相同新精度约定，建立夹持并有效抬起 6.94 cm，但 20.80 s 丢失抓持，最大滑移 50.79 mm，[结果](experiments/2026-11-18-odometry1275-training-precision.json) 仍失败。更早记录默认走原生精度，不能把跨约定的动作变化全部归因于继续训练。几何、物理参数、51 条训练输入和验收门槛不变；尚无模型完整通过单例，最终独立测试仍未使用。

对 `0_1275` 的 [实际指令诊断](experiments/2026-11-18-odometry1275-gripper-command-diagnosis.json) 显示，20.0 s 的半开度目标约 2.5–3.1 mm，20.4 s 下一组预测突然要求 32.1–34.0 mm，随后接触丢失；这是模型张爪指令的证据，不能据此排除其他误差。恢复 20.4 s 的完整物理状态与控制缓存，不重置位姿，教师保持原夹持后在 15.45 s 内完成运输与读数，新增滑移 3.52 mm。该 [候选示范检查](experiments/2026-11-18-odometry1275-held-recovery-validation.json) 包含 309 帧、三路 448px 视频及时间对齐；26 维机器人状态和全部 32 帧相对动作均在当前归一化边界内。只保留原始数据和小型检查记录，未新增重复导出，未修改正在训练的 51 条输入。未来候选原始示范共 57 条；该教师恢复不计神经网络成功。

对保存的 19.2/20.0/20.4 s 三组 RGB／机器人状态，每组另采样 8 次真实模型前向，输入不变，不推进物理。[采样记录](experiments/2026-11-18-odometry1275-held-sampling.json) 及 [夹爪摘要](experiments/2026-11-18-odometry1275-gripper-sampling.json) 显示：前 8 个执行帧中，出现半开度大于 15 mm 的采样数分别为 0/3/5，20.4 s 还有 4 次采样的平均半开度超过 15 mm。15 mm 仅是诊断分界，没有新增执行限幅或放宽验收。这里的随机流没有复用原闭环演进到该时刻的 RNG 状态，不能称为原输出逐值重放或新增任务失败次数；该记录支持补充持筒状态下正确闭合指令的示范。

另提供默认关闭的 `--gripper-open-confirmations` 控制对照。设置为 2 时，执行前缀的夹爪目标中位数若比上次接受目标增大超过每指 5 mm，需要连续两次独立观测后的模型预测确认；闭爪立即执行，逐渐张爪仍可执行。它只读预测指令与初始机器人开度，不读取物体位姿、接触或任务阶段，不调整任何几何、力限或验收门槛。原始指令和实际执行指令分别保留，摘要记录每次干预。5 mm 是显式选择的实验参数，不是官方硬件标定值。[指令检查](experiments/2026-11-18-gripper-opening-confirmation-validation.json) 验证默认逐值透传、闭爪立即执行、持续张爪可以通过、闭爪重置确认计数以及其他动作维度不变；在保存的 52 组 `0_1275` 预测上仅最后一组被拦截。该检查不推进物理，不等于任务成功。

这项 [完整闭环对照](experiments/2026-11-18-opening-confirmation-closed-loop.json) 最终在 17.85 s 丢失抓持，最大抬起 7.22 cm、滑移 43.86 mm，未采用，基线继续使用 0。[逐帧比较](experiments/2026-11-18-opening-confirmation-comparison.json) 发现前缀中位数规则漏过了第 43 组末尾两帧 20.28/14.12 mm 半开度尖峰；下一组虽然被拦截，保持的已是过松的 14.12 mm 目标，仍然掉落。不能把本轮失败仅归为连续两次张爪，也不能用它否定尚未测试的逐帧确认规则。两次运行的首帧 RGB 本身有稀疏 1/255 差异，首组右末端动作最大差约 0.82 mm，因此也不是严格相同前缀的因果重放。它是“神经网络加指令确认”的失败，不是新训练模型或成功成绩；评测队列已恢复，主训练未中断。

里程计 `0_1530` 的 [权重核验](experiments/2026-11-18-odometry1530-weight-audit.json) 通过：258 个动作／状态投影张量改变，581 个冻结张量不变；[离线诊断](experiments/2026-11-18-odometry1530-imitation.json) 单独保留。实际神经网络 [闭环](experiments/2026-11-18-odometry1530-closed-loop.json) 在 15.85 s 因量筒底座与托盘底面碰撞失败，最大抬起 21.40 mm，没有达到有效抬起 40 mm 的门槛。没有启用额外夹爪确认，当前仍无神经网络完整单例成功。

碰撞前 [第 39 组状态](experiments/2026-11-18-odometry1530-precollision-states.json)（15.6 s）仍有双侧接触，无禁止碰撞，实际抬起 20.89 mm。为这类局部持筒状态，恢复教师新增默认 0 的 `--held-initial-lift-m`：可先实际向上移动，再水平撤离。此例使用 40 mm 的额外抬升，实测增加 39.35 mm；未改动物体位姿或机器人几何。首条恢复在最后一次读数漂到 2.5 px，因原有 2 px 限制 [失败](experiments/2026-11-18-odometry1530-recovery-reading-failure.json)，没有纳入训练。

第二条仅将读数前的教师居中目标设为 `--eye-alignment-target-px .75`（默认仍为 2），稳定读数验收仍为误差不超过 2 px。该恢复在 40.15 s 完成，新增滑移 17.63 mm、最终居中误差 0.5 px、RGB 估读误差 0.059 mL。[原始数据检查](experiments/2026-11-18-odometry1530-lift-recovery-validation.json) 验证了 803 帧、三路视频、时间对齐及源物理状态／控制缓存逐值保留。但两个相对旋转分量各有约 8.39% 的 32 帧标签超出现有归一化边界（前 8 帧各约 0.65%），所以暂不加入重建配方或当前训练；另存为未选候选。原始成功数据共保留 58 条，现有候选重建配方仍为 57 条，正在训练的 51 条不变。以上恢复均使用特权程序教师，不计神经网络成绩。

`validate_raw_cylinder_recovery.py` 可直接检查原始示范，不必为检查再复制一份 LeRobot 导出；数据完整性和归一化兼容性分别记录。[失败示范拒绝记录](experiments/2026-11-18-failed-raw-recovery-rejection.json) 确认没有产出训练候选，[已有 309 帧成功恢复的复核](experiments/2026-11-18-odometry1275-raw-recovery-validation.json) 与先前独立检查的状态／动作覆盖结果逐值相同，且全部标签仍在原边界内。失败恢复的冗余训练录像随后清理，数值轨迹、失败结果与读数图片保留。

下一轮输入 `datasets/cylinder-hold-recovery-v1` 已暂存为 52 条、31,797 帧，仅追加上述 309 帧持筒恢复，没有混入旋转标签越界的候选。[追加检查](experiments/2026-11-18-hold52-append-coverage.json) 确认原 31,488 帧状态／动作／索引／来源不变，新增状态与动作也逐值等于原始记录；沿用现有归一化，新增标签零越界。[原输入检查](experiments/2026-11-18-hold52-original-inputs.json) 另确认四个旧视频文件逐字节一致，51 条旧视频时间元数据全部相同，当前训练目录的 12 个文件 SHA256 均未改变。

[官方加载器检查](experiments/2026-11-18-hold52-official-loader.json) 通过，并显式加载新增示范的第 0/154/308 帧，核对相对动作、状态、32×26 补齐与 mask。检查器先按源帧排序，避免把 `random_split` 的乱序索引误当时间顺序；新增片段有 297 个训练锚点，起始帧及前 2 s 的 40 帧全部进入训练。恢复组 [权重设定](experiments/2026-11-18-hold52-sampling-weights.json) 占一半抽样概率，排除验证帧后 [实际概率](experiments/2026-11-18-hold52-official-loader.json) 约 50.33%，新增片段约 6.18%。这些是启动前的数据检查，[检查配置](experiments/2026-11-18-hold52-purpose.json) 没有模型前向、反向或梯度更新；后续正式训练另采用下文的 `0_2040` 父模型。

暂存时四个不可变旧视频用硬链接共享同一物理文件，其他元数据各自独立，[空间记录](experiments/2026-11-18-hold52-shared-storage.json) 确认省去约 531 MiB 的相同副本，内容未改变。随后新训练切换至 52 条导出，原 51 条目录已淘汰，视频内容仍完整保留在新目录中。

里程计 `0_1785` 的 [权重核验](experiments/2026-11-18-odometry1785-weight-audit.json) 通过，258 个动作／状态投影张量改变、581 个冻结张量不变。[闭环](experiments/2026-11-18-odometry1785-closed-loop.json) 建立夹持并抬起 13.13 cm，最大滑移 3.17 mm，但 20.75 s 的模型目标加权 IK 残差 0.00605 超出原 0.002 门槛，仍失败；[离线诊断](experiments/2026-11-18-odometry1785-imitation.json) 单独保留。

`diagnose_cylinder_ik.py` 从最后一份实际物理状态重放已执行的 7 帧指令，机器人状态逐值相同，复现了上述残差。然后每种关节边界使用 33 个起点、最多 400 次迭代：[多起点检查](experiments/2026-11-18-odometry1785-ik-diagnosis.json) 在原范围下最好仍为 0.0060506，使用完整官方关节范围也仅到 0.0060331，第三关节停在下限。这不是全局不可达的数学证明；几何上界未给出有效排除证据。没有扩大执行关节范围、修改求解器或放宽验收，重放也不计一次新的模型任务成绩。

从该模型第 20.4 s 的持筒状态得到另一条 [教师恢复](experiments/2026-11-18-odometry1785-held-recovery-validation.json)：20.80 s 内完成撤回与读数，416 帧、新增滑移 5.57 mm、RGB 估读误差 0。但 [边界检查](experiments/2026-11-18-odometry1785-boundary-details.json) 发现，43.51% 的状态帧中右末端横向坐标低于旧下界，最大超出约 19.65 mm；相对动作同维度的 32 帧标签约 0.218% 越界，前 8 帧全部在界内。因此这条数据尚未追加到下一轮 52 条输入。

为检查状态裁剪的信息损失，提供默认关闭的 `--unclipped-robot-state`：仅对机器人状态保留同一仿射数值尺度、去除 `clamp(-1,1)`，动作归一化不变，不更新模型权重，也不改变几何或物理验收。[真实归一化检查](experiments/2026-11-18-state-unclipping-contract.json) 证明原 31,488 帧训练状态的归一化输出逐值相同；原 `0_1785` 运行中有 7 组实际输入（18.0..20.4 s）被旧边界裁剪。该选项目前只支持 26 维连续机器人状态。

同一检查点的 [不裁剪状态闭环](experiments/2026-11-18-state-unclipping-closed-loop.json) 在 18.20 s 因 IK 残差 0.00252 失败，未采用。[对照记录](experiments/2026-11-18-state-unclipping-comparison.json) 确认它实际访问了 4 组越界状态，选项确已起作用；首帧 RGB 同时存在稀疏 1/255 差异，首组末端动作差最大约 0.64 mm，不能把轨迹差异全部归因于裁剪选项。主训练输入与基线默认设置保持原样，评测队列已恢复。

原 51 条训练完成全部 2041 次更新，[最终权重核验](experiments/2026-11-18-odometry-final-weight-audit.json) 通过，所有训练／评测进程正常结束。最后 `0_2040` 的 [实际闭环](experiments/2026-11-18-odometry2040-closed-loop.json) 抬起 13.96 cm、滑移 7.24 mm、退离 17.17 cm，但仍在 21.20 s 因 IK 残差 0.00303 失败，没有达到 20 cm 退离和完整读数要求。本轮没有神经网络完整单例成功，也未运行最终独立测试。

已从 `0_2040` 启动真实的 `runs/cylinder-hold-correction-bc-v1` 微调，[父模型选择](experiments/2026-11-18-hold-correction-parent-selection.json) 记录其退离比 `0_1785` 更远；这是开发单例的暂定选择。数据为已验证的 52 条、31,797 帧，只新增零越界的 309 帧恢复；归一化统计逐字节沿用旧参考，保留状态裁剪、单次高斯 flow、原 IK 和无额外夹爪过滤设置。[实际配置](experiments/2026-11-18-hold-correction-train.yml)：batch 4、32 帧窗口、4 workers、seed 48，动作专家／状态投影学习率 `3e-5`／`3e-4`，最多 1025 次更新，每 128 步保存。[正式数据检查](experiments/2026-11-18-hold-correction-formal-data-validation.json) 通过后，[训练进程记录](experiments/2026-11-18-hold-correction-progress.json) 已确认实际梯度训练至第 44 步，首个新检查点尚未评测。这不是程序教师成绩。

[训练脚本](experiments/2026-11-18-hold-correction-train-job.py) 与 [评测脚本](experiments/2026-11-18-hold-correction-evaluations.py) 保存具体调用。每份完整新检查点先核验真实权重变化，再运行完整开发单例；首次完整成功只产生待回归候选，仍须通过原 8 组、开发 12 组和未用过的独立 12 组。[保留规则](experiments/2026-11-18-hold-correction-retention-policy.json) 优先完整单例成功，再比较开发进度；保留父模型、最新模型和暂定较好模型，及时淘汰已完成评测的旧权重／优化器。

首个新检查点 `0_128` 的 [权重核验](experiments/2026-11-18-hold-correction128-weight-audit.json) 确认 258 个动作／状态投影张量实际更新、581 个冻结张量不变。但 [完整闭环](experiments/2026-11-18-hold-correction128-closed-loop.json) 在 10.00 s 因 IK 残差 0.00427 失败，没有建立夹持或抬起；零滑移不代表抓持稳定。当前训练按既定设置继续，不能凭训练损失或这一次失败判断整轮效果。

新 `0_256` 的 [权重核验](experiments/2026-11-18-hold-correction256-weight-audit.json) 同样通过；[正式闭环](experiments/2026-11-18-hold-correction256-closed-loop.json) 抬起 12.06 cm、最大滑移 2.98 mm、退离 19.70 cm，但 20.65 s 因 IK 残差 0.00255 失败，尚未拿近并停稳读数。它只是开发单例的暂定较好模型；旧 `0_128` 权重与优化器已淘汰，最终独立测试仍未使用。

另做一次 [IK 容许残差 0.01 的单独诊断](experiments/2026-11-18-hold-correction256-ik-diagnostic.json)，16.75 s 因残差 0.01080 失败，也没有完成拿近读数。该诊断不计正式候选，正式评测仍使用 0.002；没有改变物理成功门槛，评测协调队列已恢复。

针对反复出现的笛卡尔目标不可达，新增 `reconstruct_cylinder_joints.py` 检查另一种数据表示是否具备可用观测。它重放已记录且经过限速的 9 维关节／夹爪／底盘指令，仅保存右臂关节位置和速度，不重新编码视频。[562 帧完整示范](experiments/2026-11-18-joint-reconstruction-base.json) 与 [309 帧恢复示范](experiments/2026-11-18-joint-reconstruction-held.json) 均完整重建，原 23 维机器人状态、量筒位置和姿态差异均为零；后者从原始完整物理状态及控制缓存恢复。这里只验证了这两段数据，没有检查全部 52 条，也没有训练关节动作策略；当前模型及动作表示不变，重放不计模型任务成绩。

另外使用该检查点采集 15/35 mL、seed 101 的两段短开发轨迹，最大 25 s，仅供从实际访问状态补充恢复示范；分别于 18.4/21.4 s 碰撞终止。它们不算原 8 组或最终独立 12 组成绩。从各自第 4/10 s 状态得到的四条教师恢复均通过，新增滑移 10.15–10.38 mm、最终读数误差 0.061–0.183 mL，见 [恢复记录](experiments/2026-11-17-multivolume-recoveries.json)。这些特权教师结果只用于训练，不计模型成功。

四条恢复另存为 `datasets/cylinder-recovery-odometry-v2`，共 55 条、33,795 帧，正在训练的 51 条输入不变。[追加校验](experiments/2026-11-17-odometry55-append-coverage.json) 证实原 31,488 帧的状态、动作、索引与来源逐值不变；沿用旧归一化时，新状态没有越界，新相对动作的两个分量分别有约 0.27%/0.68% 超出旧边界，其余为零。没有重新定义数值尺度以掩盖模型误差。新增 `--target-recovery-fraction .5` 将恢复组总抽样概率固定为一半，并保留组内 64/8/2 时间优先级；[实际官方加载检查](experiments/2026-11-17-odometry55-data-validation.json) 在排除验证帧后约为 49.91%，[权重清单](experiments/2026-11-17-odometry55-sampling-weights.json) 保存 SHA256。这里只验证了下一轮候选数据，尚未用它训练模型。

里程计 `0_510` 已建立双侧夹持并有效抬起 15.48 cm，最终靠近头部相机至 0.314 m，但运输抖动、未停稳，41.10 s 时累计滑移 25.05 mm 超限，仍失败。恢复保存的物理状态后，[夹持力诊断](experiments/2026-11-17-odometry510-gripper-forces.json) 在 20/28/36/40 s 的双指法向力约为 7.5–9 N，力限仍为 20 N。这个诊断说明当时有实际承载，不证明夹持力已最优或真机材料已标定。

另提供默认关闭的 `--temporal-ensemble`：对同一未来控制时刻的重叠绝对末端预测平均，旋转重新正交化，夹爪使用最新指令；保留原始模型指令和实际执行指令，不读取物体真值。时间对齐、过期预测移除、夹爪直通、旋转有效性和默认逐值透传均通过 [检查](experiments/2026-11-17-temporal-ensemble-validation.json)。但相同 `0_510`、相同单例和默认 IK 下，平均四组预测在 25.85 s 因倾斜失败，抬起仅 8.04 cm；因此未采用。两种真实闭环、几何审计和代码 SHA 见 [对照记录](experiments/2026-11-17-odometry510-execution-comparison.json)。截至这些记录，尚无任何神经网络策略完整通过单例任务。

`diagnose_wall_sampling.py` 用保存的同一组三路 RGB／26 维机器人状态重复运行真实模型，不推进物理。在原模型第 20/28 s 的两组观测，各采样 8 次标准高斯 flow：前 8 个执行帧的右末端位置 RMS 标准差分别为 1.64–4.74 mm、1.72–8.08 mm，旋转 RMS 标准差约 0.28–0.44°，见 [重复前向记录](experiments/2026-11-17-odometry510-sampling-variation.json)。这只测出随机采样带来的输出变化，不等于任务误差，不能单独证明哪种滤波或解码能提高成功率。

另外从这个模型第 20/28 s 的真实持筒状态补采两条教师恢复。前者在 36.35 s 内完成读数，新增滑移仍有 20.06 mm、读数误差 0.061 mL；后者在 9.60 s 因新增滑移 25.08 mm 失败，见 [两条记录](experiments/2026-11-17-odometry-held-recoveries.json)。失败恢复保留并排除；成功恢复暂存为 `datasets/cylinder-recovery-odometry-v3` 第 56 条，共 34,522 帧，尚未训练。其 [追加检查](experiments/2026-11-17-odometry56-append-coverage.json) 确认旧数据不变、机器人状态均在原边界内，但两个相对旋转分量有约 14.23% 超出旧动作归一化边界，另两个约 0.89%；因此不能仅凭 [数据加载通过](experiments/2026-11-17-odometry56-data-validation.json) 就声称动作标签完全保留。当前训练继续使用冻结的 51 条。

恢复物理状态后仅作几何接触诊断，量筒局部接触高度约从第 16 s 的 188–199 mm 变到第 40 s 的 216–218 mm，均位于 250 mm 筒壁内，见 [接触位置记录](experiments/2026-11-17-odometry510-contact-locations.json)。该记录包括接触位置和距离，不是另一条执行结果，也不替代滑移／承载力评分。

对上述第 20 s 同一持筒起点，仅将教师最终侧向观察位置从 0.14 m 改为 0.18 m 再试，32.85 s 时仍因色标不可见失败，新增滑移 15.61 mm，见 [失败记录](experiments/2026-11-17-odometry-held-read18-failure.json)。这次更改未采用，失败示范不导出为训练成功样本；不能因同一设置曾帮助另一个起点而推断普遍有效。

基于重复前向中的采样变化，新增可选 `--prediction-samples`，默认 1 保持原始单次 flow 预测；大于 1 时对完全相同的当前观测独立采样后平均，旋转重新正交化，不使用过往观测的预测。所有独立输出另存 `sampled_chunks.npy`，平均后的输出保存在 `predicted_chunks.npy`。在实际保存的神经网络输出上，[检查](experiments/2026-11-17-same-observation-average-validation.json) 确认默认值及数据顺序逐值不变、平均旋转有效。这只是准备好的解码实验；此处没有多采样平均的物理任务成绩，基线评测继续使用 1。

里程计 `0_765` 权重核验通过（258 个可训练张量变化、581 个冻结张量不变），但首轮闭环在第 10 s 附近出现数分钟无进展。对本任务进程的原生调用栈检查落在 NVIDIA EGL 库的释放/内存映射中，不能把它计为模型物理失败。中断后 `finally` 完整保存了截至 13.2 s 的动作、机器人观测、物理恢复状态和视频；当时尚无有效夹持，见 [中断记录](experiments/2026-11-17-odometry765-renderer-interruption.json)。主训练未停止。

`CarryEnv.render` 原先只有一个 Renderer，每次 448px 模型观测与 768px 录像/评分交替都会关闭并重建。现按分辨率缓存 Renderer，退出时全部关闭，模型、录像和评分仍使用原分辨率。对两个保存物理状态、三路相机和两种分辨率做 [12 组对照](experiments/2026-11-17-renderer-cache-validation.json)，integration state 逐值不变，头部 RGB 读数结果相同，额外 40 次分辨率切换没有重新创建渲染器。图像并非逐像素完全一致：先前严格逐值/1 灰度级检查失败，保留 [初始检查](experiments/2026-11-17-renderer-cache-initial-tolerance-failure.json) 和 [重复渲染诊断](experiments/2026-11-17-egl-repeat-pixel-variation.json)；同一个未经修改的 EGL Renderer 重复输出也有稀疏的 2 灰度级变化，因此图像检查采用最大 2/255、变化像素不超过 0.1%，且读数必须完全相同。这个容差只用于渲染回归检查，物理任务成功条件没有改变。

该检查点在新目录 `runs/cylinder-odometry-step765-single-v2` 完整重跑仍失败：未建立双侧夹持，50.60 s 时目标 IK 残差 0.00333 超过原 0.002 限值，见 [闭环结果](experiments/2026-11-18-odometry765-closed-loop.json)、[权重核验](experiments/2026-11-18-odometry765-weight-audit.json)。约 11.95 mm 的物体位移和 0 mm 滑移不能证明有效提起或稳定抓持。之前的运行中断不再计一次模型失败。后续队列为 `results/cylinder-odometry-later-evaluations-v2.py`，仍为单次采样、原高斯 flow、前 8 帧执行。

这次重跑使用同一检查点和相同 Torch seed 47，但不是完全相同输入的数值重放：[前 17 组预测对照](experiments/2026-11-17-renderer-cache-neural-prefix-comparison.json) 中，首步位置分量差最大约 0.275 mm，随闭环状态分化后最大约 2.82 mm。因此不能把重跑动作变化称为模型学习改善或严格配对的控制消融；真实任务结果仍需完整运行并使用后续固定案例验证。

额外测试 `runs/cylinder-odometry-step510-sameobs4-v2` 使用能建立夹持的 `0_510`，对同一观测的 4 次独立预测平均，模型微批量仍为 1、执行前 8 帧，其余滤波关闭。最终未建立抓持，29.95 s 时动作不可达，见 [结果](experiments/2026-11-18-odometry510-sameobs4.json)；该解码方式未采用，基线继续单次采样。首次启动因当前内核未提供 `/proc/.../children` 而未运行模型，不计一次任务失败。评测协调队列已经恢复，主训练继续。

另从 `0_765` 运行中断前第 13.2 s 的真实物理状态补采教师恢复：重新张开、对准并抓持，27.25 s 完成拿近和读数，新增滑移 5.89 mm，RGB 估读 25.0 mL，见 [记录](experiments/2026-11-18-odometry765-grasp-recovery.json)。这仍是使用仿真物体位姿的教师示范，不计模型成功。曾单独导出候选数据并检查加载，重复导出现已清理，原始示范保留；没有加入正在训练的 51 条输入。

空间一度仅余约 56 GiB，先对 6 份早期权重做过经 SHA256 流式重建验证的无损归档。随后按用户要求删除了这 6 份旧差分归档及不用的完整基准。最后一份历史失败模型 `cylinder-continuous-state-bc-v1/0_1020` 的 0.68 GiB 差分备份及其专用 `cylinder-odometry-bc-v1/0_510` 基准也已一并淘汰，释放约 9.04 GiB。[历史归档记录与当前可用性](experiments/2026-11-18-checkpoint-archives.json) 保留哈希记录；这些旧版本均已不能恢复。

完整权重保留官方预训练模型、当前最好／最新模型及训练核验所需的父模型；优化器现在仅保留最新完整检查点的一份，用于恢复训练。最新检查点 `0_1020` 完整保存并核验后，已删除无有效抓持的旧 `0_765` 权重和优化器。后续继续清理已评测淘汰的旧版本，保留待评测和候选模型，避免实验目录重新堆积。

`0_1275` 完整保存、权重核验及物理评测完成后，又删除旧 `0_1020` 的权重和优化器（约 9.86 GiB）。之后依次清理了旧优化器约 3.00 GiB，以及被完整保存／核验的 `0_1785` 替代的 `0_1530` 权重和优化器约 9.86 GiB。按 [开发单例的暂定取舍](experiments/2026-11-18-model-retention-1785-versus1275.json)，`0_1785` 抬起更高、滑移更小，旧 `0_1275` 权重也已删除约 8.36 GiB；这不是多案例模型质量排名。此时项目约 38.73 GiB、磁盘可用约 189.90 GiB，配置、哈希与数值轨迹保留，详细删除项见 [清理清单摘要](experiments/2026-11-18-storage-cleanup.json)。该阶段仍使用 51 条训练输入，52 条作为下一轮输入暂存；之后已完成下述切换。下文较早的独立导出和输入快照属于历史记录，其重复数据载荷已删除，不能再当作完整可用副本；小型配置、哈希和实验结果仍保留。

全部旧审计结束且新微调已实际运行后，又淘汰了不再依赖的零中心父权重、旧 `0_1785` 权重／优化器及原 51 条导出目录；原 51 条输入完整包含在当前 52 条导出中，另存原始来源与 SHA256 记录。此时项目约 30.39 GiB、磁盘可用约 198.08 GiB，尚未保存新训练检查点；这是历史时点的测量。原零中心父权重和旧 `0_1785` 现已不能直接加载。

新 `0_128` 完整保存并核验后，已用它替代最近完整恢复点，删除 `0_2040` 的旧优化器约 1.50 GiB，父模型权重仍保留作后续核验。此时项目约 38.77 GiB、磁盘可用约 189.70 GiB；后续新权重按上述保留规则轮换，避免重新累积。

归档工具及 [小文件验证](experiments/2026-11-18-checkpoint-archive-validation.json) 仍保留，覆盖逐字节恢复、默认保留原文件、拒绝覆盖、错误基准和损坏差分拒绝恢复（包括 Python `-O`）。历史恢复核验曾通过，但差分载荷及其基准现在均已删除，不能再执行旧恢复命令。

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
