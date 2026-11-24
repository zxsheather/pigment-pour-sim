# 来源与许可

| 内容 | 来源 | 版本 / 许可 |
| --- | --- | --- |
| 量子一号机器人 URDF / STL 及其导出的 GLB | [X-Square-Robot/sdk_robot](https://github.com/X-Square-Robot/sdk_robot) | `63610199bbd70eafd1fc087a015ec5c74a0c906c`；见 `licenses/sdk_robot.LICENSE` |
| 查看页中的 Three.js | [mrdoob/three.js](https://github.com/mrdoob/three.js) | 0.170.0；MIT，见 `licenses/three.LICENSE` |
| 场地参考照片 | 课程场地材料 | 保留原始来源，不另行声明开放许可 |
| 部署时获取的 Wall-X 推理源码 | [X-Square-Robot/wall-x](https://github.com/X-Square-Robot/wall-x/tree/ce2a1ae31d9ee1028b97ca40e3f66034cf8c0e5f) | 固定提交；Apache-2.0，见 `licenses/wall-x.LICENSE` |
| Wall-OSS-0.5 基础模型及微调覆盖权重 | [X-Square-Robot/wall-oss-0.5](https://huggingface.co/X-Square-Robot/wall-oss-0.5/tree/b7dc23a82dbf70a65859c9fdd596703234597028) | 权重来源、归属与许可状态见 `deploy/MODEL_CARD.md`；不将源码许可直接套用于权重 |

导出机器人时会简化过密网格，并只在生成文件中对部分传感器框架的奇异惯性进行数值正则化。上游 URDF / STL 保持原样；这些处理用于场景生成，不代表真实动力学校准。
