# 来源与许可

| 内容 | 来源 | 版本 / 许可 |
| --- | --- | --- |
| 量子一号机器人 URDF / STL 及其导出的 GLB | [X-Square-Robot/sdk_robot](https://github.com/X-Square-Robot/sdk_robot) | `63610199bbd70eafd1fc087a015ec5c74a0c906c`；见 `licenses/sdk_robot.LICENSE` |
| 查看页中的 Three.js | [mrdoob/three.js](https://github.com/mrdoob/three.js) | 0.170.0；MIT，见 `licenses/three.LICENSE` |
| 场地参考照片 | 课程场地材料 | 保留原始来源，不另行声明开放许可 |

导出机器人时会简化过密网格，并只在生成文件中对部分传感器框架的奇异惯性进行数值正则化。上游 URDF / STL 保持原样；这些处理用于场景生成，不代表真实动力学校准。
