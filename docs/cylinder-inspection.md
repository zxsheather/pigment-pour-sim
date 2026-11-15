# 量筒拿近与图像读数：稳定夹持版

搬运阶段已经独立保存：Git 提交 `67c86c1`、标签 `carry-v1-20261115`；服务器快照在 `/data/hlou/pigment-pour-study/snapshots/carry-v1-20261115`，包含代码、实验数据与录像归档、大权重路径和 SHA256 清单。过程见 `docs/experiments/2026-11-15-carry.md`。

## 本轮最终结果

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

## 稳定夹持改动（2026-11-16）

薄圆筒在平面指垫上容易滚动，扶正时的相对运动进一步变成轴向滑移。本轮将右手两侧的平面指垫改成 V 形槽：每个高度两片 ±30° 斜面，上下两层，合计八个接触几何。两侧、多点接触限制前后滚动，上下两层限制倾覆；仍由接触力和摩擦承载量筒。

抓取中心从筒底以上 215 mm 上移到 225 mm，使 40 mL 液面与手指下缘拉开距离；20/35 mL 校准标记仍位于夹爪下方。上下两层指垫仍通过实际接触承载，顶层局部超出筒口但不作挂钩或焊接。夹爪单执行器力限仍为 10 N；默认摩擦、量筒质量、标尺、运动速度和原测试起点均未改变。`--grip-profile flat` 保留平面指垫；历史几何消融采用上移前的 215 mm 抓取高度。

视觉流程每次读数失败后都允许小角度转腕寻找刻度，包括高度调整后的观察；原先只在第一张图失败时搜索。无有效色标或液面仍拒绝输出读数。对齐控制只接收 RGB 读数中的像素偏差，三次稳定读数后才验收。

第一轮扩展测试得到 19/20，失败的是 40 mL / seed 219：握持稳定，但黑色刻度遮住液面中央几列，读成 39.07 mL，误差 0.93 mL。最终读数器检查液体连通区域内更多列，寻找至少三列相邻像素支持的最高边界；对三列取中值抑制单像素噪声，边界整体高差超过 8 px 时仍拒绝读数。它使用 RGB 中实际可见的边界，不根据设定液量补值。

只修改像素边界后，完整动作在同一案例仍有 0.56 mL 偏差；增加可见性验收及转腕后仍有约 0.54 mL 偏差，说明还存在夹爪对高液面的物理遮挡。最终把抓取点上移 10 mm，同时保留“对齐后至少一半可见宽度支持同一边界”的验收。可见性不足时，通过实际关节运动转腕寻找新视角，再重新做高度对齐；粗略对齐阶段允许使用部分边缘，最终读数必须通过可见性验收。被拒绝的观察保存在 `view_rejected_*.png` 和 `view_search_history`，失败记录均保留。

验收与原 2/8 基线相同，并增加可见性验收：双侧实际承载接触；连续失去双侧接触不超过 0.2 s；夹持点相对滑移不超过 25 mm；运输倾斜不超过 0.12 rad；无超过 1 mm 的禁入碰撞；最终倾斜 ≤3°、线速度 ≤5 mm/s、角速度 ≤0.03 rad/s；液面距离画面水平中心 ≤2 px，连续三次读数极差 ≤0.4 mL，评分误差 ≤0.5 mL。没有通过放宽这些阈值取得改善。

新增参数 `--friction-scale`、`--mass-scale`、`--yaw-jitter` 用于接触和读数扰动测试；保存双指法向接触力、最大运输倾斜、接触缺失时长，以及完整最终控制量，便于重放和排查。

消融检查保留了平面指垫，只使用新的视觉搜索逻辑：8 mL / seed 100 仍在扶正时滑移失败（25.43 mm），而原先刻度遮挡失败的 35 mL / seed 100 经一次转腕搜索后通过（读数 34.81 mL，最大滑移 8.37 mm）。这支持“接触几何改善夹持，重新搜索恢复被遮挡刻度”的机制解释；不是所有改善都来自读数程序。

## 当前方法和边界

动作由程序教师生成：使用仿真中的量筒位姿规划抓取、纠正倾斜；用有界逆运动学求关节目标，平滑插值位置与姿态，再通过执行器推进物理仿真。动作不由 Wall-OSS 生成，本轮没有对 Wall 做新动作微调。

读数模块 `cylinder_reader.read_volume(rgb)` **只接收 RGB 图像**。量筒增加了明确的校准标记：紫色 20 mL、橙色 35 mL。程序定位两个标记及蓝色液面边界，在已知标尺上插值；不读取设置液量、量筒三维位置或仿真液面高度。预设液量只用于生成场景和评分。它是已知仪器的校准视觉基线，尚非任意量筒的 OCR/VLM 读数。

液面高度按 50 mL/220 mm 标尺生成，液体以刚体负载质量和可见填充几何近似，没有流动、晃动、折射或真实弯月面。真实量筒读数还需要处理相机标定、反光和弯月面，并按液面处的水平视线观察。[读数方法参考](https://www2.chem.wisc.edu/deptfiles/genchem/lab/labdocs/modules/gradcyl/gradread.htm)。

当前从量筒侧停靠位、右手空闲、量筒已装入指定液量开始，尚未串接前一阶段“拿着颜料瓶”的状态。后续必须增加放回颜料瓶或用另一只手抓量筒，并统一倾倒后的液量状态；不能用重置代替真实交接。V 形指垫是本轮新增的仿真硬件几何，不能将结果解释成原厂平面夹爪的软件控制能力。机器人自碰撞、软垫变形、接触参数与液体物理仍需实测标定。

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

单次输出包含 `summary.json`、`trajectory.npz`、`final_physics.npz`、相机原图、读数标注图，以及录制时的两路 MP4。状态、动作和视频按动作前观测对齐。`joint_command` 是 **6 个右臂关节目标 + 夹爪半开度 + 底盘前进/转向速度**，共 9 维；目前不能直接交给原先要求 23 维绝对末端动作的 Wall 数据导出器。

服务器主要记录：

- 成功视频示例：`simulation/generated/cylinder-demo-v1`。
- 第一轮/第二轮扰动测试：`simulation/generated/cylinder-validation-v1`、`cylinder-validation-v2`。
- 抓取、IK、扶正失败尝试：`simulation/generated/cylinder-ik-*`、`cylinder-motion-*`、`cylinder-read-*`。
- 首版结果摘要：`docs/experiments/2026-11-15-cylinder-results.json`。
- 本轮最终测试：`cylinder-stable-original-v4`、`cylinder-stable-expanded-v4`、`cylinder-stable-fresh-v4`。
- 中间失败与代码版本：`cylinder-stable-*-v1/v2/v3`、`cylinder-optimization-sources`。
- 本轮完整结果：`docs/experiments/2026-11-16-cylinder-stability-results.json`。

依赖实测为 MuJoCo 3.3.7、NumPy 2.5.4、SciPy 1.18.1、OpenCV headless 5.0.0.93。逆运动学使用 [SciPy 有界最小二乘](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)，其结果仍需通过真实执行器与接触仿真验证，不能将 IK 可达等同于任务成功。

## 下一轮重点

先根据真机夹爪尺寸制作或校准 V 形软垫，实测接触摩擦、闭合力、夹持滑移与相机参数，再扩充动作示范。读数端应改进遮挡检测、无色标的数字/刻度识别、弯月面分割和真实相机数据。动作稳定后，再把关节命令转换为正确的末端动作标签，接入已有 Wall 微调流程。
