# 仿真演示与动作专家试训

本地实验分支记录，基础模型始终为 `X-Square-Robot/wall-oss-0.5`。当前完成的是小规模训练链路试验，尚未验证抓取性能提升；本轮没有使用 LoRA。

## 数据

`collect_expert.py` 用预设专家在单蓝瓶场景中采集演示。专家读取仿真物体位置生成轨迹，物体真值仅在专家和评价侧使用，不进入 `observation.state`。模型特征为头部/右腕 RGB、机器人状态、任务文字；动作标签是实际下发的笛卡尔目标，不是模型失败输出，也不是下一帧实测位置。

每行先采集图像和状态，再执行动作。原始状态/动作各为 20 维：左臂 xyz/rpy/夹爪、右臂同样 7 维、底盘速度 3 维、升降 1 维、头部 pitch/yaw 2 维。官方训练加载器将欧拉角展开成 6D 旋转，并按配置计算相对动作，得到 26 维模型表示。

数据使用 **25 Hz**，每次控制对应 10 个 0.004 s 物理步，避免把原先实际 0.048 s 的周期错误声明为 20 Hz。每条演示 150 帧、6 s，瓶位在原位置的 x/y 各 ±2 cm 内变化，机器人、瓶体尺寸、质量和摩擦不变。只有通过瓶底抬高至少 5 cm、连续双指接触至少 1 s 的演示才导出为训练数据。

已生成：

- `expert_pilot_v1`：8 条成功演示，含原位置；
- `expert_train_extra_v1`：16 条额外成功演示；
- `expert_heldout_v1`：8 条独立瓶位演示。

24 条组成训练候选池，8 条外部留出；32 个位置互不重复。留出集不参与归一化统计计算。官方加载器还会在训练候选池内另作训练/验证划分。`audit_expert_data.py` 检查原始轨迹与执行记录的时间顺序、动作对应关系、图像数量、训练/留出位置隔离，以及 LeRobot 导出后的图像/状态/动作读回一致性。

原始数据、图像、归一化文件、训练日志、权重及录像均在忽略的 `runs/` 下，不推送数据或模型。`export_lerobot.py` 仅创建本地数据集，不调用上传接口。本轮数据工具使用 LeRobot 0.4.3，输出 v3 数据格式。

数据文件读回一致不代表后续归一化无损。额外的动作编解码诊断发现一个起始目标被 q01 边界裁剪约 5.36 mm，其余采样锚点恢复一致；见结果记录。正式训练前应继续检查动作分布和裁剪影响。

## 采集与导出

在准备好的仿真环境中，从仓库根目录运行。每个输出目录必须尚不存在，以免覆盖已有数据。

```bash
python3 experiments/grasp/collect_expert.py \
  --output experiments/grasp/runs/expert_pilot_v1 --episodes 8
python3 experiments/grasp/collect_expert.py \
  --output experiments/grasp/runs/expert_train_extra_v1 \
  --episodes 16 --seed 2027 --no-center
python3 experiments/grasp/collect_expert.py \
  --output experiments/grasp/runs/expert_heldout_v1 \
  --episodes 8 --seed 2028 --no-center
```

以下步骤需要独立、已核验的 Wall-X 环境和 `requirements-training.txt` 中的数据工具，不应直接替换已有推理环境的 PyTorch。脚本依赖的源码版本为 `ce2a1ae31d9ee1028b97ca40e3f66034cf8c0e5f`。

```bash
python3 experiments/grasp/export_lerobot.py \
  --input experiments/grasp/runs/expert_pilot_v1 experiments/grasp/runs/expert_train_extra_v1 \
  --output experiments/grasp/runs/lerobot_train_v1 --repo-id local/pigment_grasp_train
python3 experiments/grasp/export_lerobot.py \
  --input experiments/grasp/runs/expert_heldout_v1 \
  --output experiments/grasp/runs/lerobot_heldout_v1 --repo-id local/pigment_grasp_heldout
python3 experiments/grasp/audit_expert_data.py \
  --train experiments/grasp/runs/expert_pilot_v1 experiments/grasp/runs/expert_train_extra_v1 \
  --heldout experiments/grasp/runs/expert_heldout_v1 \
  --dataset experiments/grasp/runs/lerobot_train_v1 \
  --report experiments/grasp/runs/expert_data_audit_v1.json
```

## 试训

`prepare_training.py` 生成当前实验的配置，不是通用的充分训练配方。本轮仅更新官方动作专家参数组，冻结其他参数；batch=1、图像 224、BF16、单卡 DDP、非重入梯度检查点、AdamW 学习率 1e-5，共 100 次更新。官方加载器对训练池内部划分后，训练迭代器为 2993 帧。

共享的 `runtime_compat.py` 使用原生 SDPA，并保留 checkpoint 的视觉 MLP 宽度。启动器还保留 checkpoint 的完整词表尺寸，避免无必要地先缩小再扩展词表。训练和推理使用相同的控制标记、状态编码 512 bins 和动作布局。`audit_training_contract.py` 在不加载完整模型权重的情况下检查这些约定。

本轮实测环境为 PyTorch 2.7.0+cu128、Transformers 5.2.0；与官方推荐的 PyTorch 2.10.0 不同，因此记录为该兼容入口的实测结果，不推广为其他安装组合都可运行。

```bash
python3 experiments/grasp/prepare_training.py \
  --checkpoint "$MODEL_DIR" --dataset experiments/grasp/runs/lerobot_train_v1 \
  --repo-id local/pigment_grasp_train \
  --output experiments/grasp/runs/train_action_100_v2 --steps 100
python3 "$WALLX_SOURCE/scripts/compute_norm_stats.py" \
  --train_config experiments/grasp/runs/train_action_100_v2/train.yml \
  --data_root experiments/grasp/runs/lerobot_train_v1 \
  --output_path experiments/grasp/runs/train_action_100_v2/norm_stats.json
python3 experiments/grasp/audit_training_contract.py \
  --source "$WALLX_SOURCE" --checkpoint "$MODEL_DIR" \
  --config experiments/grasp/runs/train_action_100_v2/train.yml \
  --report experiments/grasp/runs/training_contract_v2.json
torchrun --standalone --nproc_per_node=1 experiments/grasp/train_expert_smoke.py \
  --source "$WALLX_SOURCE" \
  --config experiments/grasp/runs/train_action_100_v2/train.yml \
  --report experiments/grasp/runs/train_action_100_v2/report.json
```

入口调用官方训练器。为保证达到步数上限时仍有可评价的输出，训练结束后另导出 `evaluation/baseline` 和 `evaluation/adapted`。二者使用同一套数据统计和预处理；baseline 不包含神经网络参数更新，adapted 额外加载 `action_expert.safetensors`。基座文件只读共享，适配文件只保存可训练参数；它们是评价快照，**不含完整优化器续训状态**。

## 回到仿真

两边均在同一个外部留出瓶位、同一随机种子、25 Hz、相同起始手位下测试；本轮评价不使用程序保持或程序抬升，机械臂和夹爪均由模型输出。此 baseline 已使用示范统计量，不能与早期使用 `ex_normal` 的零样本结果直接比较。

```bash
python3 experiments/grasp/grasp_walloss.py \
  --source "$WALLX_SOURCE" \
  --checkpoint experiments/grasp/runs/train_action_100_v2/evaluation/adapted \
  --norm-key local/pigment_grasp_train --control-period .04 --tcp grasp \
  --scene experiments/grasp/runs/expert_heldout_v1/episode_0000/input_scene.xml \
  --config experiments/grasp/generated/blue_only/scene_config.json \
  --chunks 48 --output experiments/grasp/runs/train_action_100_v2/rollout_adapted_e0
```

将 checkpoint 和 output 改为 baseline 对应目录即可运行配对基线。实际结果见 [RESULTS.md](RESULTS.md)：本轮两者都未完成抓取，尚无闭环收益证据。训练损失下降仅证明优化过程能运行，不能代替真实执行评价。

## 延长训练与学习曲线

后续固定数据实验仍使用同一基座、24 条训练候选示范、同一归一化文件、batch=1 和动作专家 AdamW 1e-5。由于上一轮评价快照不含优化器状态，长训练从基座重新开始；1000、3000、6000 步来自同一次连续优化，100 步参考来自上一轮独立运行。保持已有的 21/3 轨迹划分及训练轨迹内的帧划分。

兼容启动入口支持 `--snapshot-steps 1000 3000`，会在对应的优化器更新完成后导出评价快照，再继续训练。最终结果仍写入 `evaluation/`，逐步损失保存在 `learning_curve.jsonl`。中间快照也不包含完整优化器续训状态。固定步数实验将官方整模型保存和自动验证间隔设在步数上限之外，使用下面的独立评价；不以训练损失选择模型。

评价工具：

- `audit_observation_contract.py`：不加载模型，在 CPU 上比较同一示范帧经过训练数据加载器和执行预处理后的实际输入，包括图像张量、归一化状态、提示词前缀、掩码和动作长度。
- `evaluate_expert_openloop.py`：在两个训练候选位置和三个未参与梯度更新的内部位置，各取 12 个阶段帧；比较模型序列化后的右臂动作与专家实际命令，分别报告首动作、前八步及整段的位置/夹爪误差。物体位置不进入模型，已知的归一化裁剪误差仍计入原始命令误差。
- `evaluate_expert_closedloop.py`：每次只加载一个模型，在同样的五个位置独立复位并执行。每次最多 48 个动作块，每块执行前八步，控制周期 0.04 s；全部由模型生成动作，不用脚本抓持或抬升。默认随机种子为 0，不能据此估计跨随机种子的成功率。`--external-holdout` 改为最终的八个外部位置。
- `run_learning_evaluation.py`：依次评价基座、旧 100 步参考及长训练的三个检查点。先按三个内部位置的抓取成功次数选择适配模型；相同时，选内部示范观测上前八步位置误差较小的模型，再相同则选较早步数。选择完成后，才运行八个外部位置。
- `summarize_learning_experiment.py`：从实际训练轨迹与评价报告绘制学习曲线并汇总，不从损失推断抓取成功。

内部三个位置对应导出训练候选池的 episode 21–23，即补充批次的 episode 13–15。它们不参与梯度更新，但沿用上一轮由 24 条候选轨迹计算的归一化统计，因此最终泛化结论应以归一化和模型选择均未使用的外部留出集为准。训练位置探针也可能包含官方内部帧验证划分中的帧，用于检查同一位置上的执行能力，不能称为纯训练帧误差。

```bash
python3 experiments/grasp/audit_observation_contract.py \
  --source "$WALLX_SOURCE" --checkpoint "$BASELINE_EVAL_DIR" \
  --config "$LONG_TRAIN_CONFIG" \
  --output experiments/grasp/runs/learning_eval_v3/observation_contract.json
python3 experiments/grasp/run_learning_evaluation.py \
  --source "$WALLX_SOURCE" \
  --previous experiments/grasp/runs/train_action_100_v2 \
  --experiment experiments/grasp/runs/train_action_6000_v3 \
  --output experiments/grasp/runs/learning_eval_v3
python3 experiments/grasp/summarize_learning_experiment.py \
  --experiment experiments/grasp/runs/train_action_6000_v3 \
  --evaluation experiments/grasp/runs/learning_eval_v3
```

### 更新精度与单帧学习检查

默认的单卡 DDP 路径将大部分模型参数存为 BF16，并直接用 AdamW 更新。小学习率下，参数更新可能小于 BF16 可表示的间隔：不能仅凭“每个参数张量都发生过变化”就认为大部分权重元素都能有效更新。

`--fp32-master-weights` 使用 `master_optimizer.py` 为 BF16 可训练参数建立 FP32 主副本。仍用原来的 BF16 模型前向与反向，梯度仍先由官方训练器在模型上裁剪；优化器将裁剪后的梯度转为 FP32，在主副本上累积更新，再将结果舍入回计算模型。原有 FP32 参数保留原样。该入口只支持本实验的 BF16 DDP 路径，从基座权重文件开始训练；仍不提供完整优化器续训恢复。

`test_master_optimizer.py` 与独立的 FP32 AdamW 参考比较小更新的累积结果，并检查梯度清零。不要用直接将模型可训练参数改成 FP32 的方式替代主副本方案：当前模型的部分线性层要求计算输入和权重 dtype 一致。

`--repeat-frame 0` 是诊断模式，只反复学习第一条训练候选轨迹的首帧；每步重新采样 flow 时间。它用于检查学习和推理链路，不能作为完整抓取性能或泛化结果。对应的单帧评价可传 `--frames 0 --probe-ids 0`。普通整套数据训练不传 `--repeat-frame`。

如将更新精度对照纳入学习曲线比较，`run_learning_evaluation.py` 可传 `--additional-checkpoint`、`--additional-steps 1000`，对照标签默认为 `fp32_master`。该候选仍按相同的三个内部位置成功次数和动作误差参与选择，选定后才测试外部留出。汇总图可另传 `--additional-experiment`，用菱形标出该对照，并显示两种更新方式的训练曲线。

## 状态触发与恢复示范

后续试验使用 `reactive_expert.py`，用实际手位、方向、夹爪开度和双指接触触发靠近、闭合、抬升与重新对齐。靠近采用连续的预抓取目标，夹爪未张开或方向未对准时先退开。闭合后要求短暂双指接触确认，再抬升；不再使用旧专家在固定时刻突然换阶段的长等待。接触确认仍有 0.08 s 去抖时间，最终成功仍要求抬高至少 5 cm 且连续双指接触至少 1 s。

教师能够读取物体位置和接触身份，学生仍只接收两路 RGB、自身状态和同一任务指令。四类训练起点为通用准备位、手臂偏移、抓偏后部分闭合、真实滑落。扰动通过物理命令产生，不用绑定或瞬移；准备动作在记录起点前完成，之后清空录像、遥测和成功计数，只将教师纠正动作作为标签。滑落初始化先用脚本夹住并抬升，再制造接触丢失和向下滑动，因此该组只用于恢复评价，不能计作从桌面完整抓起。

训练采集 44 次尝试，接纳 32 条成功示范，每类 8 条，共 1835 帧；失败仍保存在原始目录中，但不导出。另采集独立验证 8 条（正常/手臂偏移各 4）、最终正常抓取留出 8 条，以及三类恢复留出各 2 条。四个集合的瓶位互不重复。`audit_reactive_data.py` 检查原始时序和标签，并各重放一个模式的初态；本轮四个重放初态的状态误差均为 0。导出读回检查仍由 `audit_expert_data.py` 执行。

```bash
python3 experiments/grasp/collect_reactive_expert.py \
  --output experiments/grasp/runs/reactive_train_v4 \
  --episodes 64 --accepted-per-mode 8 --seed 2041
python3 experiments/grasp/collect_reactive_expert.py \
  --output experiments/grasp/runs/reactive_validation_v4 \
  --episodes 8 --modes clean hand_offset --seed 2042 --no-center
python3 experiments/grasp/collect_reactive_expert.py \
  --output experiments/grasp/runs/reactive_holdout_v4 \
  --episodes 8 --modes clean --seed 2043 --no-center
python3 experiments/grasp/collect_reactive_expert.py \
  --output experiments/grasp/runs/reactive_recovery_holdout_v4 \
  --episodes 30 --accepted-per-mode 2 \
  --modes hand_offset grasp_miss slip --seed 2044 --no-center
python3 experiments/grasp/export_lerobot.py \
  --input experiments/grasp/runs/reactive_train_v4 \
  --output experiments/grasp/runs/lerobot_reactive_train_v4 \
  --repo-id local/pigment_grasp_reactive_v4
python3 experiments/grasp/prepare_training.py \
  --checkpoint "$MODEL_DIR" --dataset experiments/grasp/runs/lerobot_reactive_train_v4 \
  --repo-id local/pigment_grasp_reactive_v4 \
  --output experiments/grasp/runs/train_reactive_3000_v4 --steps 3000
python3 "$WALLX_SOURCE/scripts/compute_norm_stats.py" \
  --train_config experiments/grasp/runs/train_reactive_3000_v4/train.yml \
  --data_root experiments/grasp/runs/lerobot_reactive_train_v4 \
  --output_path experiments/grasp/runs/train_reactive_3000_v4/norm_stats.json
```

归一化计算完成并核验后才启动训练。验证和最终留出轨迹均不参与统计。官方加载器另在 32 条训练池内部划分出 28 条训练轨迹，再取内部训练帧；本轮训练迭代器为 1533 个起点。保持同一基座、动作专家参数范围、batch=1、图像 224、学习率 1e-5，使用 FP32 主副本训练 3000 步，在 1000 步另存快照；不是 LoRA，也不使用前一轮训练后的权重初始化。

`audit_action_ranges.py` 单独检查右臂 XYZ 标签的边界裁剪：逐条用官方加载器和 collator 核对首帧，再扫描所有接纳轨迹中每个起点前八步的已有目标，不包括末尾补齐标签。本轮 13784 个位置目标中，192 个的裁剪误差超过 0.1 mm，最大 37.35 mm；这是标签表示误差，不能单独解释模型执行失败。该诊断不检查方向或夹爪，也没有据此改动本轮统计量或训练设置。

```bash
python3 experiments/grasp/audit_action_ranges.py \
  --source "$WALLX_SOURCE" \
  --config experiments/grasp/runs/train_reactive_3000_v4/train.yml \
  --raw experiments/grasp/runs/reactive_train_v4 \
  --output experiments/grasp/runs/reactive_audit_v4/action_ranges.json
```

```bash
python3 -m torch.distributed.run --standalone --nproc_per_node=1 \
  experiments/grasp/train_expert_smoke.py \
  --source "$WALLX_SOURCE" \
  --config experiments/grasp/runs/train_reactive_3000_v4/train.yml \
  --report experiments/grasp/runs/train_reactive_3000_v4/report.json \
  --snapshot-steps 1000 --fp32-master-weights
python3 experiments/grasp/run_reactive_evaluation.py \
  --source "$WALLX_SOURCE" \
  --reference-final-device "$REFERENCE_EVAL_DEVICE"
python3 experiments/grasp/summarize_reactive_experiment.py \
  --experiment experiments/grasp/runs/train_reactive_3000_v4 \
  --evaluation experiments/grasp/runs/reactive_eval_v4
```

评价协调器先保存实验规则，对前一轮 FP32 主副本 1000 步模型、新 1000 步及新 3000 步模型，在相同两个训练探针和八个独立验证起点上比较。只在两个新模型中，按八个验证起点的稳定抓取数选择；相同则比较验证观测上的前八步动作位置误差，再相同则选较早步数。外部留出不参与选择。输入检查增加实际阶段边界抽样，比较训练和执行的状态、图像、提示词与动作长度。

选定后，将新旧模型在相同八个新正常抓取位置及六个恢复起点上配对执行。准备结束后，所有动作都由模型输出，每次执行前八步、25 Hz、最多 48 块。最终结果分别统计完整抓取与三类恢复，不合并为一个成功率。本轮同时改变了教师、初态覆盖、归一化统计和更新次数，只能评价整套修改，不能单独归因于数据数量或某种恢复模式。

### 手腕方向的独立诊断

1000 步模型在首个训练探针出现明显方向偏离后，另登记八个验证起点的诊断：保持同一个 1000 步模型、相同初态、种子和控制周期，仅将下发的右臂 rpy 固定为臂基座坐标中的零向量，其余位置和夹爪命令仍由模型给出。它使用 `evaluate_expert_closedloop.py --fixed-tool-orientation`，报告与录像明确标为带方向约束的组合控制；保留模型原始预测和最终下发命令。不能把这项成功计作模型独立成功，也不据此改动检查点选择或最终外部评价规则。

诊断使用八个验证 case 的独立 JSON 列表，通过 `--cases` 指定；输出为 `runs/orientation_diagnostic_v4/`。汇总时可向 `summarize_reactive_experiment.py` 另传 `--orientation-diagnostic experiments/grasp/runs/orientation_diagnostic_v4`，生成独立图表，对照成功数、实际工具方向和最近抓取距离。该约束不读取物体真值，没有加入脚本抓持、闭合或抬升。

### 查看新的训练示范

`build_reactive_gallery.py` 从 `runs/reactive_train_v4/` 的 manifest、已接纳录像和初始预览图，构建可离线打开的 `runs/demo_gallery_v1/reactive-v4/index.html`。页面按四种初态筛选，显示原始尝试编号、帧数、保持时间与抬升高度。媒体按 25 Hz 仿真速度重新编码为预览副本，原录像不变；不混入模型执行或留出轨迹。已有的示范预览服务也可直接访问 `/reactive-v4/`。

```bash
python3 experiments/grasp/build_reactive_gallery.py
```

## 右臂位置标签的单变量对照

`expand_xyz_action_bounds.py` 仅扩大右臂相对 XYZ 动作的归一化边界，覆盖同一训练池全部 32 步目标；状态统计、方向、夹爪和其他通道保持原值。兼容字段仍名为 `q01/q99`，但这三个动作通道实际存放训练包络，并各加 1e-5 m 数值余量，不再解释为对应分位数。该改动不使用验证、外部位置或接管试采数据。

先用 `audit_action_ranges.py --max-clip-error-mm .1` 在旧统计上复现失败，再构造新统计并检查。旧统计的前八步最大位置截断为 37.35 mm；修正后，前八步 13784 个已有目标和完整 32 步 42848 个已有目标的截断误差均为 0。每条轨迹首帧经过真实官方加载器与 collator 核对。`--prefix-steps 32` 覆盖完整训练段，默认仍为前八步；不统计末尾补齐目标。

```bash
python3 experiments/grasp/expand_xyz_action_bounds.py \
  --statistics "$REFERENCE_TRAIN_DIR/norm_stats.json" \
  --config "$NEW_TRAIN_DIR/train.yml" \
  --raw experiments/grasp/runs/reactive_train_v4 \
  --output "$NEW_TRAIN_DIR/norm_stats.json" \
  --report experiments/grasp/runs/xyz_envelope_v5/envelope.json
python3 experiments/grasp/audit_action_ranges.py \
  --source "$WALLX_SOURCE" --config "$NEW_TRAIN_DIR/train.yml" \
  --raw experiments/grasp/runs/reactive_train_v4 \
  --output experiments/grasp/runs/xyz_envelope_v5/full_horizon_ranges.json \
  --prefix-steps 32 --max-clip-error-mm .001
```

本次新目录为 `train_xyz_envelope_3000_v5`，基准为 `train_reactive_3000_v4`。两者除模型配置文件、统计文件和输出目录路径外，训练配置及模型架构 JSON 逐项一致，包括原有日志/验证/保存间隔；仍从相同基座开始，以相同种子、数据、划分、优化器和学习率训练 3000 步，1000 步快照仅作诊断。不要将 `prepare_training.py` 的默认日志间隔误当成上一轮实际配置；应复制基准的 `logging` 并核对一致性后启动。

`run_xyz_envelope_evaluation.py` 预先固定最终对照为两种统计各训练 **3000 步**，不根据外部结果或 1000 步表现改选。内部仍使用原来的两个训练探针和八个验证起点，原 3000 步模型的内部记录复用；最终两个模型均在新采集的八个正常位置和六个恢复起点上重新配对执行。新采集种子为 2051/2052；先前已经用于诊断的外部录像不再作为本轮新留出结果。所有新旧集合位置隔离，正常抓取与恢复分别统计。

## 从真实模型状态接管的试采

`collect_model_takeovers.py` 只读取已完成评价中的两个训练位置探针。将实际模型动作前 8、24、48、96、192、384 步逐步重放，检查手位及瓶位与原遥测误差小于 1e-6，再在同一个物理状态接管。模型前缀单独保存为 `student_prefix.mp4`，清空记录和成功计数后，才采集教师纠正动作；不瞬移物体或恢复预设抓取位。

`--teacher reactive` 使用旧教师，`--teacher axis` 在瓶子倾斜超过 35°时改用 `takeover_expert.py` 的轴向恢复教师，其他状态仍调用旧教师。轴向教师从瓶体轴线的垂直方向靠近，倒伏时从上方夹取；它使用物体真值，仅用于试采。该试采不进入本轮位置边界对照的训练或统计，不计为模型成功，也未验证抓起后的直立放置、倒液或液体安全。

```bash
python3 experiments/grasp/collect_model_takeovers.py \
  --evaluation experiments/grasp/runs/reactive_eval_v4/closedloop_reactive_003000 \
  --teacher axis --output experiments/grasp/runs/takeover_axis_v5
```

实际试采中，旧教师和轴向教师在相同 12 个状态上均为 4/12，八个倒伏状态均未恢复。两个端部接近的开发试验，TCP 高于瓶体中心 20 mm 与 50 mm，各在相同两个倒伏状态上测试，均为 0/2；`--teacher endon` 保留该实验入口，不能视为已验证的恢复控制器。逆解诊断和接触记录用于排查可达性及桌面干涉，不据此声称所有倒伏状态都不可恢复。

随后使用旧教师在更早的实际状态试采：两个训练位置各重放前 8、12、16、20、24、28、32、36、40 步，纠正时间上限为 8 s。18 次尝试接纳 9 条，共 495 帧；只导出这一批成功样本，不重复合并之前试采中的相同前缀。完整前缀保持在监督数据之外，未成功的尝试仍保留。

```bash
python3 experiments/grasp/collect_model_takeovers.py \
  --evaluation experiments/grasp/runs/reactive_eval_v4/closedloop_reactive_003000 \
  --teacher reactive --prefix-steps 8 12 16 20 24 28 32 36 40 \
  --max-seconds 8 --output experiments/grasp/runs/takeover_precontact_v5
python3 experiments/grasp/export_lerobot.py \
  --input experiments/grasp/runs/takeover_precontact_v5 \
  --output experiments/grasp/runs/lerobot_takeover_precontact_v5 \
  --repo-id local/pigment_takeover_precontact_v5
python3 experiments/grasp/audit_model_takeovers.py \
  --input experiments/grasp/runs/takeover_precontact_v5 \
  --dataset experiments/grasp/runs/lerobot_takeover_precontact_v5 \
  --report experiments/grasp/runs/xyz_envelope_v5/takeover_audit.json
```

审计检查初态重放、原始时序、实际命令对应、物体真值未进入特征，以及每条导出轨迹首帧的图像/状态/动作读回。这九条只来自两个原始训练场景，未来划分应按 `source_batch + source_episode` 分组，不能随机拆分其相关前缀后宣称泛化。当前轮训练和归一化仍只用原先 32 条轨迹；新增九条仅准备供后续试验。新采集目录另保存执行时的教师及采集源文件快照。

`build_takeover_gallery.py` 为九条接纳示范生成 A/B 预览，默认输出在既有预览服务下的 `runs/demo_gallery_v1/takeover-v5/`。页面分别显示录制的模型动作前缀和教师纠正，不将教师结果计成模型成绩；原录像不改写。

## 将九条接管纠正加入续训

`prepare_takeover_mix.py` 将原 32 条与九条接管纠正合并为 41 条、2330 帧。先放九条纠正，再放原来的 28 个训练 episode，最后保留原四个内部验证 episode；来源文件只读。官方加载器按 episode 列表的前缀选训练集，因此合并后必须使用 `train_test_split: 0.9024390243902439`（37/41），不能沿用 0.875。九条的两个来源场景都属于原训练组，相关前缀不跨场景划分。

```bash
python3 experiments/grasp/prepare_takeover_mix.py \
  --original experiments/grasp/runs/reactive_train_v4 \
  --corrections experiments/grasp/runs/takeover_precontact_v5 \
  --output experiments/grasp/runs/mixed_takeover_v6 \
  --report experiments/grasp/runs/takeover_training_v6/group_split.json
python3 experiments/grasp/export_lerobot.py \
  --input experiments/grasp/runs/mixed_takeover_v6 \
  --output experiments/grasp/runs/lerobot_mixed_takeover_v6 \
  --repo-id local/pigment_grasp_reactive_v4
```

两组均从 `train_xyz_envelope_3000_v5/evaluation/adapted/action_expert.safetensors` 的 3000 步动作专家权重开始，各再更新 3000 步。控制组仍用原 32 条池、28 个训练 episode；实验组用合并池、37 个训练 episode。复制上一轮实际配置和架构，保留 batch=1、种子 2026、1e-5 恒定学习率、BF16 计算、FP32 主副本及 `logging`；只调整数据路径、episode 比例和输出路径。二者都使用覆盖 41 条完整动作目标的同一份 `common_norm_stats.json`，状态、旋转及夹爪统计不变；新增纠正仅需将原相对 Z 上界再扩大约 0.109 mm。新外部位置不进入统计。

`--initial-action-expert` 在官方基座权重加载完成后、创建优化器之前覆盖全部 257 个可训练张量，并检查名称、形状及数值完全相同。这是保留训练后权重、使用新优化器的额外微调，不能解释为恢复了旧 AdamW 动量。快照分别记录新增更新数和累计权重更新数。导出的 `baseline` 目录仍是公开基座，不代表续训起点；评价只读取 `adapted`。

`--dataset-plan` 读取实际加载的 episode 及官方帧级 95%/5% 划分，写出每条示范进入梯度训练的帧数，要求九条都非零。虽然来源场景分组保持一致，两个长度不同的数据集的内部帧级随机划分并非逐帧配对。额外 3000 步超过两组一个完整训练 epoch，单进程、batch=1、无放回 sampler 会实际访问每个选中的训练帧。先在独立目录执行一次更新的短检查，再启动正式训练。

```bash
python3 -m torch.distributed.run --standalone --nproc_per_node=1 \
  experiments/grasp/train_expert_smoke.py \
  --source "$WALLX_SOURCE" --config "$CONTINUATION_DIR/train.yml" \
  --report "$CONTINUATION_DIR/report.json" \
  --initial-action-expert experiments/grasp/runs/train_xyz_envelope_3000_v5/evaluation/adapted/action_expert.safetensors \
  --dataset-plan "$DATASET_PLAN" --fp32-master-weights
```

`prepare_takeover_evaluation.py` 预先固定各加训 3000 步后比较，不按外部成绩选择检查点；以种子 2061/2062 准备八个新正常位置和六个恢复起点，并核对与先前各组位置隔离。`run_takeover_training_evaluation.py --cohort control|augmented` 对两组分别执行相同的开环和闭环评价，最后运行 `summarize_takeover_training.py` 核对训练、权重加载、数据成员和全部试验有效性，生成汇总图表。

内部评价包含原两个训练探针、八个验证起点，另加两个训练中见过的纠正初态。`replay_takeover_start.py` 重放旧模型前 24 步，检查机器人及瓶子状态与采集时一致，再清空记录和抓持计时；此后的全部动作由被测模型产生。这两个案例明确计为见过场景的前缀恢复，不能算从准备位开始的完整抓取或新位置泛化。九条只覆盖倒伏前状态，不能据此声称已训练倒伏恢复。

## 执行频率与抓持分解诊断

`run_feedback_diagnostic.py` 固定上述累计 6000 步模型，使用原内部列表前四个开发起点：两个训练探针和两个已用于评价的验证起点，各含一个正常初态、一个手臂偏移初态。它们不再称为新留出。模型输入、控制周期和物理参数保持原值，每组最多执行 384 个控制步，即 **15.36 秒仿真**；8 步组最多推理 48 次，2 步组最多 192 次。成功时可提前结束，其他案例必须用满相同控制预算。

四个主组为 `direct8`、`direct2`、`hold8`、`hold2`。辅助组在双指连续接触至少 80 ms、模型仍请求闭合时，冻结当时已发出的夹爪命令；手臂六维命令继续由模型产生。触发读取仿真接触身份，因此属于有物理真值辅助的诊断，不能记作模型独立或真机成绩。它不增加闭合力度，不改变提示词或手臂目标，也不瞬移物体。

```bash
python3 -m unittest discover -s experiments/grasp -p test_feedback_trace.py
python3 experiments/grasp/run_feedback_diagnostic.py \
  --cases "$FEEDBACK_CASES" --output "$REPLAY_OUTPUT" --replay-only
python3 experiments/grasp/run_feedback_diagnostic.py \
  --source "$WALLX_SOURCE" \
  --checkpoint experiments/grasp/runs/train_takeover_augmented_3000_v6/evaluation/adapted \
  --cases "$FEEDBACK_CASES" --output "$DIAGNOSTIC_OUTPUT" \
  --execute-steps 2 --max-control-steps 384
```

`--contact-latched-gripper` 开启辅助组。`DiagnosticEnvironment` 只在每次原始物理步进后读取测量，不改变基础环境的积分、夹爪映射或成功判据。先逐步重放四条已记录模型动作，要求右臂、瓶子和成功计时与原遥测误差小于 1e-6；本轮三者实测误差均为 0。随后保存每个物理步的接触、瓶底高度、实际/请求夹爪开度、力反馈及姿态/位置跟踪误差，分别统计连续双指接触和“抬高 5 cm 且双指接触”。`feedback_trace.py` 的三个回归用例区分高度下降、接触丢失和仍在保持的终止状态，防止把计时清零一律解释为松手。

同一模型、种子和初始图像仍可能出现毫米级首段预测差异；另加 `repeat8` 重复对照，不替换主组。`initial_image_audit.json` 保留逐相机精确差异：本轮一个辅助案例的腕部初图只有 12 个颜色通道相差一个灰度级，其余一致。汇总明确记录这种偏差，只接受最大 1 灰度、变动通道比例不超过 1e-4 的稀疏差异，不将其报告为逐像素完全相同。初始预测的运行差异原因尚未确定，不用小幅接触时长变化单独推断改进。

`replay_feedback_probes.py` 在预先选定的第一个训练案例上，重放同一段已有模型命令。双指接触 80 ms 后，分别测试冻结原夹爪命令、改为更紧闭合目标 1.7、固定当前工具位姿、固定工具位姿并更紧闭合。参考教师原闭合目标为 2.2；探针不改摩擦、执行器限幅或物体附着。每种干预都从相同状态开始，除指定通道外保留原命令。模型不再根据变化后的状态重新预测，因此四段均是特权反事实回放，不是模型成功数。

```bash
python3 experiments/grasp/replay_feedback_probes.py
python3 experiments/grasp/summarize_feedback_diagnostic.py
```

所有诊断仍在推理时暂停仿真物理，并保存实际推理延迟。更短执行段带来更多推理和噪声重采样；该对照不等同于已实现更高真机反馈率。目录为 `runs/feedback_diagnostic_v7/`，包含 20 次模型执行、四次原动作重放及四次反事实回放，三类结果分别标注。结论应同时参照阶段到达数、首次接触/高度中断和录像，而非只看最大抬升高度。

## 抓取目标向指根方向修正

`grasp_depth_probe.py` 对同一正常起点的三个已记录接近动作分别测试 0/10/20/30 mm。正值沿模型请求的夹爪局部正 X 轴送进手掌，使瓶子相对夹爪更深入；旋转转换为臂基座坐标后再加到 XYZ 目标上，开始 0.4 s 平滑送入。原 TCP `[.132, 0, 0] m`、FK/IK、真实机器人状态、方向和原夹爪命令保持原定义，不将单侧动作修正伪装成 TCP 重标定。

脚本读取有效碰撞网格并将实际瓶心和接触点转换到夹爪本体坐标，记录距离指尖的余量及掌部接触。抓持确认仍用双指接触 80 ms 和闭合请求，属于特权诊断；确认后固定方向和夹爪 2.2。默认只原位保持；`--controlled-lift` 另运行完整配对组，统一先停 0.2 s、再沿世界 Z 用 2 s 最小加加速度曲线相对抬升 12 cm、最后保持。始终运行 15.36 s，分别统计原 1 s 成功、末尾连续双指保持 ≥3 s，以及至少有手部接触的末尾抬升保留时间。

```bash
python3 experiments/grasp/grasp_depth_probe.py
python3 experiments/grasp/grasp_depth_probe.py --controlled-lift
python3 experiments/grasp/run_grasp_depth_live.py \
  --source "$WALLX_SOURCE" \
  --checkpoint experiments/grasp/runs/train_takeover_augmented_3000_v6/evaluation/adapted
python3 experiments/grasp/run_grasp_depth_live.py \
  --source "$WALLX_SOURCE" \
  --checkpoint experiments/grasp/runs/train_takeover_augmented_3000_v6/evaluation/adapted \
  --depths-mm 20 30 --tag live_extended
python3 experiments/grasp/summarize_grasp_depth.py
```

新推理默认按回放末尾保持数选择最小的最佳非零量，与 0 mm 做三种子对照。确认抓持后不再调用模型，进入相同的程序抬升/保持；此前仍是每次模型预测 32 步、执行前八步，输入只有双相机和机器人状态。所有新推理均归为组合控制，不能计为模型独立抓取。追加深度扫描属于开发调参，不作未见测试；实际送进量也不能仅由名义偏移推断。

`summarize_grasp_depth.py` 核对全部 36 次有效性、零偏移前缀的物理一致性、每步目标只包含声明的偏移，以及确认后相同的世界竖直抬升控制。目录为 `runs/grasp_depth_v8/`。执行过的早期原位保持版、带慢抬升版和新推理版源文件分别保存，以便区分版本。所有局部相机只作观察，不是模型输入；没有改碰撞网格、摩擦、质量、执行器限幅或附着约束。

## 闭合前的自适应几何检查

`guarded_grasp.py` 是显式使用仿真瓶位的局部控制器，不是视觉检测器。模型只接收原来的相机和机器人状态，负责接近；模型请求闭合且瓶子进入局部抓取区域后，程序检查真实瓶心在夹爪中的深度、横向位置、高度及瓶轴与夹爪 Z 轴的夹角。未进入该区域的闭合请求先保持张开，手臂仍执行模型目标。

进入局部控制后，原本已足够深入且对准的情况直接保持；只有深度不合适时修正深度。若还需调整高度、横向或方向，先张开、退到预抓取位置，再对准并送入。张开对准时持续更新当前瓶轴方向，保留原接近方向，避免瓶子回正后仍追踪过时倾角。配置目标瓶心为夹爪本体坐标 `[132, 0, -5] mm`；闭合前允许 X 为 123–142 mm，横向误差 ≤4 mm，高度误差 ≤6 mm，轴向夹角 ≤6°，两指实际开度均 ≥4.1。连续通过检查后才发出 2.2 的闭合命令；接触后的几何容差适度放宽以允许物理沉降。

局部修正的每步平移上限 2.4 mm、转角上限 0.025 rad，控制周期 0.04 s；闭合失败最多重新对准两次，局部阶段超过 5 s 或瓶身倾斜超出范围则停止并张开。抓持确认仍需 80 ms 双指物理接触及闭合受阻，并额外检查实际夹持几何；之后完全复用上一轮的 0.2 s 停留、2 s 世界竖直慢抬升 12 cm、固定方向和夹爪值的控制。没有改变模型权重、TCP、物理参数或附着方式。

```bash
python3 -m unittest discover -s experiments/grasp -p test_guarded_grasp.py -v
python3 experiments/grasp/run_guarded_grasp.py --replay \
  --output experiments/grasp/runs/guarded_grasp_v9/paired
python3 experiments/grasp/audit_policy_repeatability.py \
  --source "$WALLX_SOURCE" \
  --checkpoint experiments/grasp/runs/train_takeover_augmented_3000_v6/evaluation/adapted
python3 experiments/grasp/run_guarded_grasp.py \
  --source "$WALLX_SOURCE" \
  --checkpoint experiments/grasp/runs/train_takeover_augmented_3000_v6/evaluation/adapted \
  --suite normal --output experiments/grasp/runs/guarded_grasp_v9/paired
python3 experiments/grasp/run_guarded_grasp.py \
  --source "$WALLX_SOURCE" \
  --checkpoint experiments/grasp/runs/train_takeover_augmented_3000_v6/evaluation/adapted \
  --suite external --seeds 0 --output experiments/grasp/runs/guarded_grasp_v9/paired
python3 experiments/grasp/summarize_guarded_grasp.py \
  --root experiments/grasp/runs/guarded_grasp_v9/paired \
  --repeatability experiments/grasp/runs/guarded_grasp_v9/repeatability/report.json
```

输出总目录为 `runs/guarded_grasp_v9/`，最终配对组在 `paired/` 下，分别是旧接近动作、正常起点三种子、八个已有验证场景。每次都运行 384 步，分别报告任意时刻的原 1 s 判据和结束时仍连续双指抬高 ≥3 s 的判据。

新推理使用确定性计算设置，并在每次推理前将 Python、NumPy、Torch 的种子设为 `trial_seed * 10000 + control_step // 8`；记录每次输入摘要和种子。同输入重复五次完全一致仍不足以保证两个完整执行过程的输入一致，因此最终采用共享接近段的配对试验：从同一经过物理生成的完整初态开始，只有完整 `mjSTATE_INTEGRATION` 完全相同才共用 RGB；只有该状态和推理种子都相同才共用模型采样。首次控制分叉后按各自状态继续，局部程序接管后不再调用模型。这些是新生成的共同接近动作加不同控制，不能称为每个分支都独立重新推理。共用条件中的物理真值不传入模型；相机原渲染与实际输入的差异另作记录。

汇总逐对核对完整初态、相机图像、首次程序干预前的所有动作、物理状态、输入摘要和种子，并核对两组相同的抬升曲线。之前未满足完整配对检查的开发记录单独保留，不混入最终配对成绩。每组保存执行源码、权重/归一化文件摘要、阶段事件、动作和物理步记录。八个场景在之前轮次用过，不能称为全新留出测试；全部仍是使用仿真真值的组合控制结果。

## 保持过程的缓慢下滑诊断

```bash
python3 experiments/grasp/replay_slip_probe.py
```

默认复放 `guarded_grasp_v9/paired/normal/seed_1_guarded` 的已执行目标，在原抓持确认后分别保持原目标 2.2、改为更紧目标 1.7、保持原目标但启用三次防滑后处理。三组末端目标一致，新增每指实际接触法向力大小之和与相对竖直位移记录。先核对原条件完整复放误差为零，再核对干预前一致性。不调用或训练模型，不更改摩擦、质量、限力；防滑求解属于数值诊断条件，不能作为真机改进。输出 `runs/slip_diagnostic_v10/` 保存最终静态图和轨迹，不生成完整执行录像；不能仅以旧的接触/高度判据称为无滑移稳定抓取。
