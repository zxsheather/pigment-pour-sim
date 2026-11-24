# 在另一台电脑上运行 Wall-OSS 抓瓶模型

这个分支提供 **Ubuntu x86_64＋NVIDIA GPU 的推理包和量子一号 Pro SDK 桥接**。先跑离线样例，再读取真机图像与状态；完成现场标定后，可以预览模型动作，或向右臂发送一次限幅后的目标。

当前权重是 **Wall-OSS-0.5 的动作专家微调版本 v6**，不是 LoRA，也不是 0.5B 模型。模型来源、SHA256 和训练步数见 [model_manifest.json](model_manifest.json)、[MODEL_CARD.md](MODEL_CARD.md)。代码和权重公开发布；权重放在 [Release](https://github.com/zxsheather/pigment-pour-sim/releases/tag/walloss-grasp-deploy-v1-20261124)，避免大文件进入 Git 历史。

已经验证的范围见 [VALIDATION.md](VALIDATION.md)。**尚未接真实机械臂验证，不能声称已实现真机自动抓瓶。** 仿真完整抓取程序使用物体位置和接触真值，且存在瓶子缓慢下滑；这些程序保留在 `experiments/grasp/` 供复现实验，真机桥接没有移植它们。

## 1. 准备目标电脑

建议 Ubuntu 22.04/24.04 x86_64、Python **3.11**、NVIDIA GPU。离线推理已在 24 GB RTX 4090 上验证。预留至少 30 GB 磁盘和 32 GB 内存；安装支持 CUDA 12.8 PyTorch 的 NVIDIA 驱动，`nvidia-smi` 应能正常运行。使用源代码的 PyTorch 算子回退和 SDPA，不要求安装 FlashAttention 或编译 Wall-X CUDA 扩展。

```bash
git clone --branch deploy/walloss-grasp --single-branch https://github.com/zxsheather/pigment-pour-sim.git
cd pigment-pour-sim

# 示例使用已安装的 conda；也可自行准备 Python 3.11 venv。
conda create -n pigment-walloss python=3.11 -y
conda activate pigment-walloss
sudo apt-get update
sudo apt-get install -y git libgl1 libglib2.0-0
bash deploy/bootstrap.sh
```

脚本安装测试过的推理版本，并从随包的 Git bundle 恢复固定提交的 [Wall-X 源码](https://github.com/X-Square-Robot/wall-x/tree/ce2a1ae31d9ee1028b97ca40e3f66034cf8c0e5f)，无需再次从 GitHub 克隆上游。不执行上游 `pip install -e .`，因此不触发 CUDA 扩展编译。源码位于 `deploy/vendor/wall-x`，只应用可审计的 FlashAttention 可选导入补丁；内存加载与视觉兼容处理来自本仓库的 `walloss_runtime.py` 和 `runtime_compat.py`。

## 2. 下载权重并跑通离线推理

```bash
python -m deploy.artifacts download
python -m deploy.artifacts verify
python -m deploy.infer
```

下载器会：

1. 下载约 **646 MB** 的 Release 微调包，检查整个压缩包 SHA256，然后逐文件校验并解压。
2. 从 Hugging Face 获取固定版本 `X-Square-Robot/wall-oss-0.5@b7dc23a82dbf70a65859c9fdd596703234597028` 的 **8.33 GB `model.safetensors`**，检查 SHA256。
3. 保留微调包内的配置、tokenizer 和归一化参数，不用基础模型配置覆盖它们。基础权重以软链接指向 Hugging Face 缓存，减少重复占用。

推理读取 `deploy/samples/` 中实际仿真生成的头部、右腕 RGB 和机器人状态，输出 `deploy/outputs/offline_prediction.json`。应看到 `follow2_pos` 为 **32×7**，加载审计中 `action_expert_overlay_tensors=257`，缺失、意外、未加载参数列表均为空。这个测试不连接机器人。

`follow2_pos` 每行是右臂**绝对** `[x,y,z,roll,pitch,yaw,gripper]`；位置单位米，姿态为 `Rotation.from_euler("xyz", ...)` 的弧度。归一化键必须为 `local/pigment_grasp_reactive_v4`，不要换成 `ex_normal` 或 `x2_normal`。

首次安装需要联网。之后可断网推理。搬运整个已安装 checkpoint 时，注意 `model.safetensors` 是指向缓存的软链接：用 `cp -L` 复制其内容，或在另一台电脑重新执行下载器。离线拷贝后仍执行 `verify`。

手动下载 Release 包时也可运行：

```bash
python -m deploy.artifacts download --archive /path/to/walloss-grasp-v6-overlay.tar.gz
```

该命令仍需获取基础模型，除非 `deploy/checkpoint/model.safetensors` 已在本地。大文件传输中断后可手动下载完整文件再用 `--archive`；校验失败不会被当成成功安装。

## 3. 安装机器人匹配的官方 SDK

按机器人实际固件选择 [官方 SDK Release](https://github.com/X-Square-Robot/sdk_robot/releases) 的 wheel。桥接参考 API 提交为 `63610199bbd70eafd1fc087a015ec5c74a0c906c`，并检查了 `v1.1.2` wheel 的方法签名。未知固件不自动升级到最新 SDK。已确认兼容 `v1.1.2` 的机器人可这样安装：

```bash
python -m pip install -c deploy/sdk-constraints.txt \
  https://github.com/X-Square-Robot/sdk_robot/releases/download/v1.1.2/x2robot-1.1.2-py3-none-any.whl
python -m pip check
python -c 'from x2robot import connect; from x2robot.geometry_msgs import Pose; from x2robot.sdk import GripperPosition; print("SDK imports OK")'
```

本环境使用 `opencv-python==4.11.0.86` 满足 SDK 依赖。不要同时安装 `opencv-python-headless`，两者提供相同的 `cv2` 文件。如果机器人配套 SDK 版本的方法或返回格式不同，先改适配层并只运行 `observe`。

根据[官方网络与 SDK 说明](https://github.com/X-Square-Robot/sdk_robot/blob/63610199bbd70eafd1fc087a015ec5c74a0c906c/README.md)，把目标电脑接到机器人所在局域网。填写实际 SDK 地址，不要把本地配置提交到 Git。

```bash
cp deploy/robot.example.json deploy/robot.local.json
# 编辑 robot.local.json 的 endpoint，例如 x2://<机器人IP>:<SDK端口>
python -m deploy.robot observe --output deploy/outputs/observe-001
```

`observe` 不需要权重或标定，保存两路图像和原始末端姿态、夹爪、头部、升降、底盘信息。它只发观测请求，SDK 自身会建立连接并发送心跳；不会切换控制模式或下发关节/位姿目标。空图像、无效四元数、RPC 错误会直接报错，不以虚构观测替代。

## 4. 在现场填写坐标与单位标定

**不能把配置里的 null 直接替换成单位阵并视为标定完成。** 用机器人实际 TF、SDK 末端定义和测量结果填入下表。`calibration_reviewed` 默认为 false，`preview` 和 `step` 都会拒绝运行；确认测量与坐标方向后才设为 true。

| 配置 | 含义 |
| --- | --- |
| 左右臂 `T_model_reference_from_sdk_reference` | 4×4 齐次变换，把 SDK 返回姿态的参考坐标系映射到训练时该臂的 `*_arm_base_link`。平移单位米 |
| 左右臂 `T_sdk_tool_from_model_tcp` | 从模型 TCP 到 SDK 末端的坐标映射，满足下方乘法。训练 TCP 位于夹爪基座本地 **+X 方向 0.132 m**，姿态轴沿用夹爪基座；SDK 的 TCP 未在真实硬件上核对 |
| `gripper_model_per_sdk`, `gripper_model_offset` | `g_model = scale * g_sdk + offset`，需要两臂都填。模型夹爪范围约 0–4.5，是仿真 URDF 的电机坐标，不是牛顿或毫米 |
| `sdk_gripper_limits` | 实测/官方确认的右夹爪 SDK 下、上限，按 SDK 原单位填写 |
| `lift_model_per_sdk`, `lift_model_offset` | 升降量转换到模型的米制高度。检查 SDK 是米还是其他单位，以及零点差异 |
| `head_model_per_sdk`, `head_model_offset` | 长度 2，按 `[pitch,yaw]` 将 SDK 原始值映射为模型弧度，包含方向符号。SDK 文本中的角度说明存在差异，必须核对实际单位，不能凭变量名猜测 |
| `stationary_model_posture` | 标定时固定的升降高度、头部角度及容差。仿真数据约为升降 0.30 m、头部 `[0.55,0]` rad，真机对应值需测量，程序不会自动移动到该姿态 |
| `workspace_model_xyz_min/max` | 右臂模型 TCP 的已检查无障碍包围盒，三维米制坐标；不要填整个机械臂理论工作范围 |
| `camera_array_order` | SDK 若返回 ndarray 时明确选择 RGB 或 BGR；压缩图像用 PIL 解码为 RGB |

变换约定是列向量，`T_A_from_B` 把 B 中的点变换到 A。观测与动作使用互逆公式：

```text
T_modelRef_from_modelTCP
  = T_modelRef_from_sdkRef @ T_sdkRef_from_sdkTool @ T_sdkTool_from_modelTCP

T_sdkRef_from_sdkTool
  = inverse(T_modelRef_from_sdkRef) @ T_modelRef_from_modelTCP
    @ inverse(T_sdkTool_from_modelTCP)
```

现场先比较静止姿态和小幅手动移动：SDK 坐标变换后的方向、量级、TCP 位置应与实际机械臂一致。左右臂的参考系各自独立。更改升降、底盘位置或头部/相机布置后，应重新核对标定有效性。

头部与右腕画面需与训练时的视角、方向、物体尺度接近；当前代码使用完整 RGB 帧，由模型配置 resize，未做真实相机内外参匹配、畸变矫正或视角域适配。程序记录的是本地收到观测的时间；还需现场确认 SDK 返回新帧、图像与状态同步，不能把本地时间检查当成传感器同步验证。

## 5. 先预览，再单步试动

```bash
# 只读：预测并显示将被限幅后的右臂目标，零运动命令。
python -m deploy.robot preview --output deploy/outputs/preview-001
```

`preview` 先加载模型，再获取观测、推理、重新读取状态。若标定未填、底盘运动、头/升降姿态不符、推理过久、推理期间机械臂/底盘移动或模型目标超出配置的无障碍区域，直接拒绝。检查保存的图像、原始反馈和 `prediction.json` 中的模型目标与限幅目标。

在机器人官方界面/配套 SDK 中，按现场操作规程启用 **SDK 工作模式＋末端位姿控制模式**。本程序不会自动切模式、回零、站起、移动底盘或改变头部/升降。确认右臂周围和整条连杆运动空间无障碍，现场可随时操作硬件急停，再执行一次：

```bash
python -m deploy.robot step --execute --output deploy/outputs/step-001
```

默认只取模型 **第 1 个目标**，向该目标最多移动 **2 mm、旋转 1°**，发送一次 `right_arm.set_end_pose` 后退出。它会重新获取观测并预测，**不会执行之前 preview 保存的目标**；不会循环执行 32 帧，也不会写左臂、底盘、头或升降。下一步需再次显式运行命令。程序限幅只检查 TCP 包围盒与步长，未实现连杆碰撞、关节限位/奇异位形、力矩或抓握力控制，因此仍需要现场确认运动空间。

默认不写夹爪。确认夹爪单位、方向与范围后，可显式允许右夹爪最多改变 0.05 模型电机坐标：

```bash
python -m deploy.robot step --execute --allow-gripper --output deploy/outputs/step-gripper-001
```

桥接层的观测和目标 RPC 设置 1 秒超时，并检查官方 `ExecutionResult.is_success`；SDK 连接与心跳沿用官方实现。SDK 返回成功只代表接受命令，不证明已经到位。右臂与夹爪不是原子操作；夹爪命令失败时，前面的右臂目标可能已被接受。发生超时或中断时不会盲目重发或自动回零。退出 Python **不会撤销已接受的 SDK 目标，也不等于硬件急停**；运动异常使用机器人硬件急停及官方恢复流程。

## 6. 文件和复现

| 文件/目录 | 用途 |
| --- | --- |
| `deploy/bootstrap.sh`, `requirements.txt` | 推理环境与固定版本源码 |
| `deploy/artifacts.py`, `model_manifest.json` | 下载、完整哈希校验、安全解包 |
| `deploy/infer.py`, `samples/` | 无机器人依赖的离线推理样例 |
| `deploy/sdk_bridge.py`, `geometry.py`, `guards.py`, `robot.py` | 官方 SDK 适配、坐标变换、单步限制 |
| `deploy/robot.example.json` | 待实测的配置模板；本地配置已加入 gitignore |
| `experiments/grasp/README.md`, `RESULTS.md` | 历次仿真、微调和失败情况 |

测试无需机器人：

```bash
python -m unittest discover -s deploy/tests -v
python -m pip check
```

要重新仿真，另建仿真环境并参考 `experiments/grasp/README.md`；本部署环境不安装 MuJoCo 等实验依赖。后续完整任务还需要现场遥操作数据/校正、真实视觉反馈与防滑验证，再逐步接入取量筒、倒液和体积反馈。此包交付的是现有模型的可迁移推理与真机调试入口。
