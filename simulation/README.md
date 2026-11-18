# 场景生成与查看页

`world_model/scene_config.json` 保存桌面、托盘、瓶子、量筒、机器人位置和路径示意。所有尺寸均为照片参考估计值。

- `world_model/build_scene.py`：导入固定版本的官方机器人 URDF，添加桌子和器材，生成 `generated/scene.xml`。
- `export_scene.py`：设定展示姿态，导出 `reconstruction/quanta_x1.glb`、三张静态视图和 `manifest.json`。
- `viewer/app.js`：浏览器中的三维展示、颜色高亮和布局调整。
- `viewer/template.html`：查看页布局。
- `viewer/build_viewer.py`：将脚本、照片和机器人模型打包为根目录的离线 HTML。

安装和生成命令见 [项目 README](../README.md)。`generated/`、`third_party/`、Python 环境和前端依赖缓存不纳入 Git；当前导出的模型、视图和离线页面已包含在仓库中。

## 底盘四向移动演示

生成 `generated/scene.xml` 后，在仓库根目录运行：

```bash
LIBGL_ALWAYS_SOFTWARE=1 MUJOCO_GL=egl .venv/bin/python -m simulation.demos.demo_base_motion
```

程序通过 MuJoCo 读取底盘关节的位置和速度，根据误差施加力和转矩，使底盘沿世界坐标的 −X、+Y、+X、−Y 各移动 0.20 m。终端输出每段的目标位置和实测位置，动画保存到 `simulation/generated/base_motion.gif`。此演示使用简化的平面底盘关节，并在运行时关闭机器人与地面的接触；桌子、托盘和瓶子仍参与碰撞检测。
