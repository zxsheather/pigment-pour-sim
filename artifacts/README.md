# 量筒微调参数与实验录像

截至 2026 年 11 月 24 日，4A6000 最后写入的训练检查点是 **V924 第 1024 步**。实验编号按方案分配，实际完成时间取决于 GPU 排队顺序。该检查点在 25 mL 的完整模型闭环试跑中为 **0/1**，终止原因为碰撞。程序演示的成功次数另见搬运与量筒实验记录。[本次模型评测](cylinder-v924-step1024/evaluation/summary.json)

| 材料 | 用途 |
| --- | --- |
| [V924 实际训练配置](cylinder-v924-step1024/train.yml) | AdamW，动作专家学习率 1×10⁻⁵，状态投影学习率 5×10⁻⁵，每卡批量 4，随机种子 48，动作窗口 8 步 |
| [V924 训练输入](cylinder-v924-step1024/training/)与[运行记录](cylinder-v924-step1024/execution-status.json) | 537 段、373,093 帧。保留采样权重、归一化统计、模型配置、训练代码改动及运行命令。原 YAML 保留 4A6000 的路径 |
| [V924 模型参数](cylinder-v924-step1024/) | 按网络层保存与固定版本官方底模的无损差分，附原始张量布局、分层校验值及恢复脚本 |
| [V1064 两组已训练的对照配置](cylinder-v1064-comparison/) | 保存 nominal 与 extended 两组实际训练配置、运行记录和第 1024 步评测，两组在完整单例上均为 0/1 |
| [V1077 下一轮配置](cylinder-v1077-prepared/train-v1077.yml)与[数据校验](cylinder-v1077-prepared/v1077-verified-focused-open-advance.json) | 568 段、396,009 帧。目录中的 NPY 是逐帧采样权重。数据已通过加载检查，该配置尚未启动训练 |
| [持瓶移动录像](demos/carry-teacher.mp4)与[量筒读数录像](demos/cylinder-25ml/teacher.mp4) | 均由程序控制器执行。量筒演示预设 25 mL，图像读数为 24.94 mL。[逐次读数与物理指标](demos/cylinder-25ml/summary.json) |

## 恢复 V924 完整模型

从 GitHub 获取本分支后，在仓库根目录运行以下命令。恢复需要 Python 3、NumPy、固定版本的官方底模，以及额外约 8.4 GiB 的输出空间。在新机器上可先从 [官方固定版本](https://huggingface.co/x-square-robot/wall-oss-0.5/tree/b7dc23a82dbf70a65859c9fdd596703234597028) 下载 `model.safetensors`。在 4A6000 上可以直接使用下方已有文件。

```bash
bundle=artifacts/cylinder-v924-step1024
base=/data/hlou/pigment-pour-study/models/wall-oss-0.5/model.safetensors
(cd "$bundle" && sha256sum -c SHA256SUMS)
out=$(mktemp -d)
cp -R "$bundle/checkpoint/." "$out/"
python3 "$bundle/restore_weights.py" --base "$base" --output "$out/model.safetensors"
```

恢复脚本在 CPU 上运行，只在 SHA-256 匹配后写入目标文件。原始模型大小为 **8,978,728,800 字节**。已在 4A6000 上从官方底模和分支中的差分完整还原，并与原检查点的 SHA-256 核对一致。

| 文件 | SHA-256 |
| --- | --- |
| 官方底模 | `37f8f327d3f9c27f056ed078db43c0e39b8caf3df0dc0163cdb8cf92c167968a` |
| 恢复出的 V924 完整模型 | `d9a787b0a7eca40d5d253e31455d1e881ba493db1881eddd3decf07264e39eea` |

模型文件和推理配置可用于评测。在 4A6000 上原样续训时，V924 对应的 AdamW 优化器和随机数状态保存在 `/data/hlou/pigment-pour-study/runs/cylinder-joint-early-v718-retreat-bc-v924-a6000/checkpoints/0_1024/`。训练数据位于 `/data/hlou/pigment-pour-study/datasets/`，具体子目录见各自 YAML 的 `data.lerobot_config.root`。[V924 权重审计](cylinder-v924-step1024/checkpoint-1024-weight-audit.json)
