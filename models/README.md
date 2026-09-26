# 模型与示例来源

所有下载项的固定版本 URL、SHA256、文件大小见 `registry.json`。运行 `python scripts/fetch_models.py` 校验或重新下载。权重、数据集不进入 Git。

- YuNet 检测及五点定位：[OpenCV Zoo 官方模型](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)，MIT，许可证在 `licenses/YuNet-LICENSE.txt`。
- SFace 特征：[OpenCV Zoo 官方模型](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface)，Apache-2.0，许可证在 `licenses/SFace-LICENSE.txt`。本项目使用官方预训练模型，不能把它称为本项目训练的 ArcFace。
- 3DDFA_V2：[官方仓库](https://github.com/cleardusk/3DDFA_V2)，固定提交 `1b6c67601abffc1e9f248b291708aef0e43b55ae`，源代码 MIT；`vision3d/mobilenet_v1.py` 保留原作者说明，许可证在 `licenses/3DDFA-V2-LICENSE.txt`。BFM 衍生资产可能受原始模型使用条款约束，本地科研学习使用，未在本仓库重新分发权重/基模型。不要把代码 MIT 许可证当成所有训练数据或模型的许可。下载命令 `python scripts/fetch_models.py --with-3d`。
- `assets/astronaut.png` 来自 scikit-image 0.24.0；[官方说明](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut) 标明 NASA 公有领域照片。人物为宇航员 Eileen Collins。`sample.jpg` 为格式转换，不是生成脸。

3D 重建命令：`python -m vision3d.reconstruct`。流程是 YuNet 框 → 120×120 BGR → MobileNet 预测 62 参数 → 学习到的形状和表情系数作用于 3DMM → OBJ、原图投影、三角度 PNG 和原生 WebGL 浏览器交互查看器。输出位于 `reports/3d/`。没有使用固定模板假冒重建；但它仍是单图统计估计，不含真实尺度或完整后脑，不等于精确扫描。未提供真值网格，因此没有声称三维误差准确率。

LFW 为独立下载的本地基准，来源是 scikit-learn 官方 `_lfw.py` 指向的 Figshare 镜像：13,233 张、5,749 人；6000 对。`data/lfw/lfw.tgz` SHA256 为 `055f7d9c632d7370e6fb4afc7468d40f970c34a80d4c6f50ffec63f5a8d536c0`；`pairs.txt` 为 `ea42330c62c92989f9d7c03237ed5d591365e89b3e649747777b70e692dc1592`。图片不随 GitHub 源码发布。
