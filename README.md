# 取瓶量液 · 场景重建

根据场地照片重建量子一号 Pro、两张拼接桌、四色颜料瓶、两个托盘和量筒，提供可离线打开的 3D 查看页与 MuJoCo 场景生成脚本。

直接打开 [场景重建.html](场景重建.html)，即可旋转、缩放、切换视角、选择高亮瓶子和调整布局。页面自带机器人模型、参考照片与脚本，打开时无需联网。

## 布局

- 每张桌的短边沿机械臂前伸方向，两张桌沿此方向前后排列，长边相接。
- 四色瓶位于近端桌，量筒和另一只托盘位于另一张桌的侧边。
- 虚线及停靠轮廓用于展示绕桌角、转向和靠近量筒的位置关系。
- 单桌暂按约 0.60 × 1.20 m、桌高 0.75 m 估计，尺寸待实测校准。

这是静态场景重建。器材比例、机器人姿态与路径用于核对几何布局。

## 文件

```text
场景重建.html                   离线查看页
场地照片/任务场地.jpg             原始参考照片
simulation/world_model/scene_config.json    场景几何参数
simulation/world_model/build_scene.py       从官方机器人资产生成 MuJoCo 场景
simulation/export_scene.py      导出机器人 GLB、静态视图与来源记录
simulation/reconstruction/     当前模型和静态视图
simulation/viewer/             查看页源码与打包脚本
scripts/bootstrap_sources.py   获取固定版本的机器人资产
sources.json                   上游仓库与版本
docs/scene.md                 场景说明
licenses/                      上游许可证
```

## 重新生成

从仓库根目录运行，先获取固定版本的机器人资产：

```bash
python3 scripts/bootstrap_sources.py
```

建立独立 Python 环境，生成场景及静态视图：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m simulation.world_model.build_scene
.venv/bin/python -m simulation.export_scene
```

无显示窗口的 Linux 环境可通过 `MUJOCO_GL=egl` 选择离屏渲染后端。

重新打包查看页：

```bash
npm ci --prefix simulation/viewer --no-audit --no-fund
python3 simulation/viewer/build_viewer.py
```

输出仍为根目录的 `场景重建.html`。页面内调整仅影响当前查看会话；将导出的配置替换 `simulation/world_model/scene_config.json` 并重新生成后，才会更新保存的场景。

机器人资产与查看页依赖的来源、版本及许可见 [THIRD_PARTY.md](THIRD_PARTY.md)。
