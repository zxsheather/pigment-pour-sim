# 场景生成与查看页

`scene_config.json` 保存桌面、托盘、瓶子、量筒、机器人位置和路径示意。所有尺寸均为照片参考估计值。

- `build_scene.py`：导入固定版本的官方机器人 URDF，添加桌子和器材，生成 `generated/scene.xml`。
- `export_scene.py`：设定展示姿态，导出 `reconstruction/quanta_x1.glb`、三张静态视图和 `manifest.json`。
- `viewer/app.js`：浏览器中的三维展示、颜色高亮和布局调整。
- `viewer/template.html`：查看页布局。
- `viewer/build_viewer.py`：将脚本、照片和机器人模型打包为根目录的离线 HTML。

安装和生成命令见 [项目 README](../README.md)。`generated/`、`third_party/`、Python 环境和前端依赖缓存不纳入 Git；当前导出的模型、视图和离线页面已包含在仓库中。
