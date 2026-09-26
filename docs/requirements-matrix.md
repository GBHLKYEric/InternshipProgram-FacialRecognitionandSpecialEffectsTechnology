# 原 PDF 要求与验收矩阵

状态口径：`verified` 表示已有本机证据支持所写的这一小项；`planned` 表示代码/文档准备中或尚未运行；`blocked` 表示依赖明确缺失，旁列解除条件。**一行中的理论学习通过不代表训练、准确率或部署通过。**本表禁止用模型作者的数字、合成图训练损失、预训练模型推理或模板网格冒充原要求的真实训练结果。

本版已纳入2026-09-26的本机运行证据；未通过的部分保持未验收。理论说明见 [教程](tutorial-zh.md)，官方依据见 [资料与许可](sources.md)。`verified（部分）`只验证同一格中明确列出的子项，不代表整行PDF任务全部通过。

| 编号 | PDF 要求 | 交付/验收证据要求 | 当前状态与差距 |
|---|---|---|---|
| 1.1 | Anaconda、Python、PyTorch、OpenCV、MMDetection | 安装版本、解释器路径、import 和算子测试 | verified（部分）：项目venv、PyTorch/OpenCV/ORT，另有独立Docker MMDetection3.3/MMCV2.1 CPU NMS+训练通过；`reports/environment.json`、mmdet-environment-freeze.txt。Anaconda未装，venv不是Anaconda。 |
| 1.2 | Git 基本操作、GitHub 仓库与首次提交 | 远程公共仓库 URL、提交 SHA、实际文件清单 | planned；本地 Git commit 不是远程开源成功。 |
| 1.3 | Docker 镜像与环境 | Dockerfile、build 日志、容器内 Hello World | verified：WSL Docker29.1.3实际构建运行，容器Python3.12.14；Dockerfile/hello_world.py；MMDetection研究镜像另外Python3.10。 |
| 1.4 | Jupyter 实验 | notebook 文件、执行后的输出与无错误记录 | verified：`notebooks/01-first-vision-lab.ipynb` 4个代码单元实际执行。 |
| 2.1 | 检测、对齐、识别、验证基础 | 教程第 1、5、7 课及练习 | verified：教程与答案已交付；学习效果仍需学生自己完成练习。 |
| 2.2 | 下载探索 CelebA、LFW，可视化 | 实际文件统计、分布与本地样本图 | verified（部分）：LFW合法维护镜像已本地下载，6000对真实运行；CelebA图像与完整分析尚未完成，已有旧CSV不代表图像已取得。 |
| 2.3 | MMDetection 人脸模型检测 | 框架推理日志、权重来源、检测图 | verified（部分）：WIDER小样本自训RetinaNet推理已跑，`reports/wider_pilot_prediction.jpg`显示top10低分候选，模型未收敛不能称有效检测器；YuNet属于独立应用基线。 |
| 2.4 | 关键点模型定位 | 输入、坐标输出、可视化 | verified：YuNet五点真实推理，`reports/effect-none.jpg`；不代表68点训练。 |
| 3.1 | MTCNN、RetinaFace 原理 | 教程对照与原论文链接 | verified：教程第5课；RetinaNet配置不能叫RetinaFace实现。 |
| 3.2 | MMDetection 在 WIDER FACE 训练 | COCO 转换、配置、训练日志、权重、数据清单 | verified（部分）：真实WIDER32张/242框，MMDetection RetinaNet随机初始化1epoch16step训练，36.11秒；完整数据正式训练尚未完成。 |
| 3.3 | WIDER 验证集检测指标 | Precision、Recall、明确协议的 AP | verified（部分）：16张/153框独立验证，COCO bboxAP=0、AP50=0，`reports/wider_pilot.json`；非WIDER官方easy/medium/hard指标，未收敛。 |
| 4.1 | HRNet、SAN 原理 | 教程关键点章节 | verified：教程第5课。 |
| 4.2 | 在 300-W 或 COFW 训练 | 数据协议、训练日志、权重、独立评估 NME | blocked：需合法数据与本机训练；预训练五点不能替代。 |
| 4.3 | 仿射变换对齐 | 对齐实现、关键点、前后图 | verified（部分）：SFace alignCrop在完整LFW推理中实际运行；68点仿射实现位于research/landmarks.py，单独前后图须依真实训练权重补充。 |
| 5.1 | ResNet、ArcFace 原理 | 教程数学示例与实现对照 | verified：教程第4/7课与research/recognition.py；梯度smoke通过。 |
| 5.2 | MS-Celeb-1M 或其子集训练 ResNet50+ArcFace | 数据来源授权、ResNet50 配置、训练曲线与 checkpoint | blocked：未获得可核验 MS-Celeb-1M 训练包；合法自建数据属于明确替代实验。 |
| 5.3 | LFW 验证，自训模型 >98.5% | 6,000 pairs 十折、阈值仅从训练折选择、均值/标准差、失败计数 | verified（部分）：预训练SFace首轮严格单脸72.25%；第二轮最大主体98.9333%±0.38873个百分点、零失败，两个JSON均保留；后者达预训练基线数值目标，自训ArcFace尚无正式结果。 |
| 6.1 | 量化、剪枝、蒸馏 | 教程原理、收益与误区 | verified：教程第10课，不能把置零自动视为加速。 |
| 6.2 | 训练好识别模型的动态量化 | FP32/INT8 文件、相同输入精度、速度、大小 | verified（部分）：随机ResNet50动态Linear量化管线和计时已跑；`reports/optimization-smoke/comparison.json`；训练模型精度未验，INT8本次变慢。 |
| 6.3 | ONNX 转换与推理 | ONNX checker、ORT 推理、数值差、输入输出规格 | verified（部分）：随机ResNet50导出+checker+ORT数值检查通过，3DDFA真实预训练回归器也已导出运行；自训识别checkpoint导出待5.2。 |
| 6.4 可选 | ByteNN 模拟加速 | 清楚标为模拟的实现与比较 | blocked：无可核验 SDK；如用 ORT，标为推理模拟。 |
| 7.1 | GAN、StarGAN、AttGAN 原理 | 教程生成章节与练习 | verified：教程第12课与训练实现；真实CelebA训练另列7.2。 |
| 7.2 | CelebA 训练 StarGAN，发色/年龄/性别属性 | 数据属性清单、训练日志、checkpoint、编辑结果 | blocked：需 CelebA 与真实训练；染色/磨皮图像处理不算 GAN。 |
| 7.3 | FID、IS 质量评估 | 特征器、输入规模、预处理、统计分数 | blocked：依赖有意义生成结果与独立真实集；随机网络 FID 不可报告为标准 FID。 |
| 8.1 | 3DMM、NeRF 原理 | 教程三维章节 | verified：教程第14课。 |
| 8.2 | PRNet 或 3DDFA_v2 单图三维重建 | 实际权重推理、输入图、对应 OBJ/PLY | verified：真实3DDFA_V2，38365顶点/76073面，`reports/3d/face.obj`与reconstruction.json；无3D真值精度。 |
| 8.3 | PyTorch3D 或 OpenGL 渲染多视图 | 重建网格、渲染过程、至少三个视角 | verified（部分）：matplotlib三个视角已输出；WebGL（OpenGL ES）查看器已生成，浏览器实际交互验证单独记录；未安装PyTorch3D。 |
| 9.1 | 关键点特效原理 | 教程坐标、透明度与跟踪章节 | verified：教程第11课与app.py。 |
| 9.2 | 实时关键点、动态贴纸 | 真实视频/摄像头路径、贴纸随脸移动、帧时 | verified（部分）：360帧仿射动画逐帧检测/贴纸，演示18秒；真人摄像头/移动端实际跟踪未验证。 |
| 9.3 | 磨皮、美白、口红 | 可调节效果、结果视频、无脸行为 | verified：效果图片与视频，测试无脸行为；五点口红区域为近似，张嘴/大姿态有局限。 |

## 阶段与综合交付

| PDF 综合要求 | 需要的证据 | 状态 |
|---|---|---|
| 环境报告、Hello World、OpenCV 灰度程序 | 版本报告、可运行脚本与输出 | verified：环境JSON/freeze、Docker Hello与已执行Notebook |
| 数据分析报告与基线 | 数量、类别、失败数量、数据拆分和指标 | verified（部分）：LFW全协议与失败；CelebA完整探索未完成 |
| 难例挖掘/难例采样、大规模类别问题 | 原理说明及训练代码，不能仅出现术语 | verified：P-K/batch-hard实现与梯度检查；真实训练效果待评估 |
| 三类特效演示程序 | 明确类型与输出；三种贴纸不等于属性编辑+三维+动态三大类全实现 | verified（部分）：几何贴纸、美颜美妆和三维已有输出；GAN属性编辑正式训练未完成 |
| 统一界面、实时视频 | 本地网页、输入/效果/结果一体化，Chrome 验证 | verified（部分）：本地服务/HTTP与界面交付；Chrome自动化入口不可用，摄像头实测未通过 |
| CPU/移动端实时性 | 设备、分辨率、均值/P95、端到端 FPS；移动端单独实测 | verified（部分）：检测+特效中位17.935ms、P95 62.103ms；无移动端/完整端到端保证 |
| 项目演示视频 | 可播放视频、实际处理过程与限制说明 | verified：reports/demo.mp4，360帧18秒；仿射动画非真人录像 |
| 项目总结报告、汇报 PPT | 本文档目录及实际 PPTX | verified（部分）：Markdown、教程/报告/问题PDF与PPTX已生成；最终视觉检查另记录 |
| StyleGAN3、DECA 必读、CNN/度量/GAN/扩散/3DGS/可微渲染 | 教程覆盖、原论文链接、练习与答案 | verified：18课完整教程与sources.md；不称所有模型都已训练 |
| BytePS、火山引擎、内部高效工具 | 真实部署凭据或明确模拟/未体验说明 | blocked：BytePS CUDA/NCCL；云 API 需服务开通和凭据；内部 SDK 未提供。 |
| 全部源代码 GitHub 开源 | 公共仓库可读取、源码齐全、许可证、无私密数据 | planned |
| 面向高三零基础教程 | 从 Python/图像/向量到实验、每课练习答案 | verified：tutorial-zh.md，18课、术语、公式、答案和复现入口 |
| 执行问题及解决方法 | 实际错误、根因、修改、复测与剩余限制 | verified：issues-and-fixes.md，包含版本兼容、网络/环境与旧代码审计 |
| 全部源码汇编与使用位置（新增要求） | 源码快照、模块/调用入口/PDF编号/验证命令 | verified：code-map.md、完整Markdown/HTML源码汇编及reports/code-inventory.json；最终发布前再按文件清单核对 |

## 不能改变口径的四条规则

1. 测试训练代码能反向传播，只证明“训练程序能执行”；必须有训练数据、训练轮次与验证才能评价模型。
2. 预训练模型取得的性能归属于指定预训练模型与本次处理协议，不能归给本项目从零训练。
3. 可旋转的固定人脸模板展示三维坐标概念，不证明从输入图片恢复了该人的三维几何。
4. 未完成项必须保留；开源发布、视频或漂亮界面不能抵消未通过的科研验收项。
