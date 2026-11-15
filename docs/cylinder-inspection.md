# 量筒拿近与图像读数：第一版动作原型

搬运阶段已经独立保存：Git 提交 `67c86c1`、标签 `carry-v1-20261115`；服务器快照在 `/data/hlou/pigment-pour-study/snapshots/carry-v1-20261115`，包含代码、实验数据与录像归档、大权重路径和 SHA256 清单。过程见 `docs/experiments/2026-11-15-carry.md`。

## 做到了什么

量筒从桌面上的自由刚体开始，右夹爪先张开、靠近、合拢，再提起、退离桌边、移到头部相机前。通过关节控制扶正量筒，用图像中的液面位置调整高度，最后读取刻度。运行期间不直接修改量筒位姿，没有 weld/等式约束；松开夹爪后量筒会落下。

当前录制示例：设置 25 mL，图像读数约 **24.9 mL**，计算误差约 **0.07 mL**；全过程约 **33.75 秒**，量筒中心距相机约 **0.65→0.38 m**，最高提起约 **11.6 cm**，最终倾斜约 **1.1°**。这些只是简化仿真中的一次完整示例，不代表真实仪器的测量精度。

本机视频位于仓库根目录 `量筒演示_拿近与读数.mp4`。左侧显示机器人动作，右侧放大头部相机中的刻度与液面；放大只用于演示，读数算法处理原始 RGB 图像。

## 成功示例之外的结果

测试了 8、15、25、35 mL，各两种起点（seed 100/101，x/y 各 ±4 mm）。

- 第一版偏保守的滑移判据下，8 组中通过 1 组。
- 改为检查双侧实际承载接触、允许有限的夹持再定位，并要求最终静止、液面与视线对齐、倾斜不超过 3°后，通过 **2/8**。通过的是 25 mL / seed 100 与 35 mL / seed 101，误差分别约 0.07、0.19 mL。
- 其余失败包括扶正时发生持续滑移、最终不够竖直、刻度色标被遮挡。失败仍被记录为失败，即使某张图能算出一个数值。
- 空白图片、遮住校准色标、旋转图片均拒绝读数。松开夹爪的负例中量筒下落约 0.74 m、双侧接触变为 0；原搬运任务的一条回归测试仍成功。
- 另试过用单个平滑凸圆柱替代分段筒壁作碰撞体，两条试跑都提前倾斜失败，未采用。对应代码和日志保留在生成目录及快照中。

所以目前结论是：**完整动作与读数链路已出现可复现成功例，抓取与扶正还不稳。** 不应将成功视频当作稳定成功率。

## 当前方法和边界

动作由程序教师生成：使用仿真中的量筒位姿规划抓取、纠正倾斜；用有界逆运动学求关节目标，平滑插值位置与姿态，再通过执行器推进物理仿真。动作不由 Wall-OSS 生成，本轮没有对 Wall 做新动作微调。

读数模块 `cylinder_reader.read_volume(rgb)` **只接收 RGB 图像**。量筒增加了明确的校准标记：紫色 20 mL、橙色 35 mL。程序定位两个标记及蓝色液面边界，在已知标尺上插值；不读取设置液量、量筒三维位置或仿真液面高度。预设液量只用于生成场景和评分。它是已知仪器的校准视觉基线，尚非任意量筒的 OCR/VLM 读数。

液面高度按 50 mL/220 mm 标尺生成，液体以刚体负载质量和可见填充几何近似，没有流动、晃动、折射或真实弯月面。真实量筒读数还需要处理相机标定、反光和弯月面，并按液面处的水平视线观察。[读数方法参考](https://www2.chem.wisc.edu/deptfiles/genchem/lab/labdocs/modules/gradcyl/gradread.htm)。

当前从量筒侧停靠位、右手空闲、量筒已装入指定液量开始，尚未串接前一阶段“拿着颜料瓶”的状态。后续必须增加放回颜料瓶或用另一只手抓量筒，并统一倾倒后的液量状态；不能用重置代替真实交接。机器人自碰撞、夹爪软垫与液体物理仍需更准确的模型和实测标定。

## 复现

在 Thor 的 `/data/hlou/pigment-pour-study/pigment-pour-sim` 中运行：

```bash
# 新动作的额外依赖，和旧搬运环境共用 MuJoCo。
../.venv-sim/bin/python -m pip install -r simulation/inspection_requirements.txt

# 单次完整试跑；默认 25 mL，无位置扰动。
MUJOCO_GL=egl ../.venv-sim/bin/python simulation/inspect_cylinder.py \
  --volume 25 --record --output simulation/generated/my-cylinder-demo

# 只用保存的相机原图重新读数，不传入液量或三维状态。
../.venv-sim/bin/python simulation/cylinder_reader.py \
  simulation/generated/my-cylinder-demo/final_face.png \
  --output simulation/generated/my-cylinder-demo/reread.json

# 组合为中文标注视频。
../.venv-sim/bin/python simulation/make_cylinder_video.py \
  --episode simulation/generated/my-cylinder-demo \
  --output simulation/generated/my-cylinder-demo.mp4 --speed 2

# 八组扰动、拒绝读数负例、松爪测试及搬运回归。
MUJOCO_GL=egl ../.venv-sim/bin/python simulation/validate_cylinder.py \
  --output simulation/generated/my-cylinder-validation
```

输出目录必须不存在。当前验证会因尚有失败案例返回非零退出码，这是对原型局限的记录。

单次输出包含 `summary.json`、`trajectory.npz`、`final_physics.npz`、相机原图、读数标注图，以及录制时的两路 MP4。状态、动作和视频按动作前观测对齐。`joint_command` 是 **6 个右臂关节目标 + 夹爪半开度 + 底盘前进/转向速度**，共 9 维；目前不能直接交给原先要求 23 维绝对末端动作的 Wall 数据导出器。

服务器主要记录：

- 成功视频示例：`simulation/generated/cylinder-demo-v1`。
- 第一轮/第二轮扰动测试：`simulation/generated/cylinder-validation-v1`、`cylinder-validation-v2`。
- 抓取、IK、扶正失败尝试：`simulation/generated/cylinder-ik-*`、`cylinder-motion-*`、`cylinder-read-*`。
- 测试结果摘要：`docs/experiments/2026-11-15-cylinder-results.json`。

依赖实测为 MuJoCo 3.3.7、NumPy 2.5.4、SciPy 1.18.1、OpenCV headless 5.0.0.93。逆运动学使用 [SciPy 有界最小二乘](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)，其结果仍需通过真实执行器与接触仿真验证，不能将 IK 可达等同于任务成功。

## 下一轮重点

先改善薄量筒的夹持接触、闭合力与扶正时的握持再定位，再扩充动作示范。读数端应改进遮挡检测、无色标的数字/刻度识别、弯月面分割和真实相机数据。动作稳定后，再把关节命令转换为正确的末端动作标签，接入已有 Wall 微调流程。
