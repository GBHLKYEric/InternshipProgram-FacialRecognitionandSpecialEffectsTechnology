# 原 PDF 要求与验收矩阵

状态口径：`verified` 表示已有本机证据支持所写的这一小项；`planned` 表示代码/文档准备中或尚未运行；`blocked` 表示依赖明确缺失，旁列解除条件。**一行中的理论学习通过不代表训练、准确率或部署通过。**本表禁止用模型作者的数字、合成图训练损失、预训练模型推理或模板网格冒充原要求的真实训练结果。

本版已纳入2026-09-26—27的本机运行证据；未通过的部分保持未验收。理论说明见 [教程](tutorial-zh.md)，官方依据见 [资料与许可](sources.md)。`verified（部分）`只验证同一格中明确列出的子项，不代表整行PDF任务全部通过。

| 编号 | PDF 要求 | 交付/验收证据要求 | 当前状态与差距 |
|---|---|---|---|
| 1.1 | Anaconda、Python、PyTorch、OpenCV、MMDetection | 安装版本、解释器路径、import 和算子测试 | verified：Anaconda 2026.07-1、conda 26.5.3 与独立项目 Python 3.12.14 已安装；CPU autograd、YuNet、Tk/ImageTk、真实 Jupyter 内核及 pip check 通过，见 [Anaconda 报告](../reports/anaconda-environment.json) 与[环境教程第 7 节](continued-experiments.md#7-把四类运行环境真正弄明白anaconda-安装启动与排错)。原主 venv 保留；独立 XPU 算子验证通过；独立 Docker MMDetection 3.3/MMCV 2.1 CPU NMS 与训练通过，版本见 reports/mmdet-environment-freeze.txt。 |
| 1.2 | Git 基本操作、GitHub 仓库与首次提交 | 远程公共仓库 URL、提交 SHA、实际文件清单 | verified：公开仓库GBHLKYEric/InternshipProgram-FacialRecognitionandSpecialEffectsTechnology；沿用用户原仓库并保留原文件。最终提交与文件核验见交付说明。 |
| 1.3 | Docker 镜像与环境 | Dockerfile、build 日志、容器内 Hello World | verified：WSL Docker29.1.3，Hello及核心lab镜像均实际构建运行；lab容器faces1/组合特效/cosine1、宿主健康检查200；MMDetection研究镜像另外Python3.10。 |
| 1.4 | Jupyter 实验 | notebook 文件、执行后的输出与无错误记录 | verified：`notebooks/01-first-vision-lab.ipynb` 4个代码单元实际执行。 |
| 2.1 | 检测、对齐、识别、验证基础 | 教程第 1、5、7 课及练习 | verified：教程与答案已交付；学习效果仍需学生自己完成练习。 |
| 2.2 | 下载探索 CelebA、LFW，可视化 | 实际文件统计、分布与本地样本图 | verified：LFW实际13233图/5749身份及分布SVG；CelebA完整202599图官方StarGAN配套ZIP已下载并校验，图像名与40属性表对应，按真实清单统计属性；实际训练/评估所需图像按名单解压，不把配置池大小写成已访问样本数。 |
| 2.3 | MMDetection 人脸模型检测 | 框架推理日志、权重来源、检测图 | verified（部分）：WIDER小样本自训RetinaNet推理已跑，`reports/wider_pilot_prediction.jpg`显示top10低分候选，模型未收敛不能称有效检测器；YuNet属于独立应用基线。 |
| 2.4 | 关键点模型定位 | 输入、坐标输出、可视化 | verified：YuNet五点真实推理，`reports/effect-none.jpg`；不代表68点训练。 |
| 3.1 | MTCNN、RetinaFace 原理 | 教程对照与原论文链接 | verified：教程第5课；RetinaNet配置不能叫RetinaFace实现。 |
| 3.2 | MMDetection 在 WIDER FACE 训练 | COCO 转换、配置、训练日志、权重、数据清单 | verified（部分）：完整12880图/156994个COCO有效框与兼容Docker已准备；完整1epoch迁移训练进行中。早期32图随机初始化pilot保留；只有最终完成记录才能验收全量训练。 |
| 3.3 | WIDER 验证集检测指标 | Precision、Recall、明确协议的 AP | verified（预训练完整基线）：YuNet全部3226张val，官方难度协议Python实现，AP easy0.8844221/medium0.8656839/hard0.7504021，与OpenCV Zoo参考实现绝对差均0；不是自训RetinaNet结果。其全量训练模型评估仍进行中。 |
| 4.1 | HRNet、SAN 原理 | 教程关键点章节 | verified：教程第5课。 |
| 4.2 | 在 300-W 或 COFW 训练 | 数据协议、训练日志、权重、独立评估 NME | blocked：300-W需要真实登记信息；COFW官方彩色/灰度下载中断，灰度仅293760/178022021字节且MD5不符，普通续传失败。未使用残缺包、未产生真实NME；见[下载证据](../reports/cofw-access.json)。预训练五点不能替代。 |
| 4.3 | 仿射变换对齐 | 对齐实现、关键点、前后图 | verified（部分）：SFace alignCrop在完整LFW推理中实际运行；68点仿射实现位于research/landmarks.py，单独前后图须依真实训练权重补充。 |
| 5.1 | ResNet、ArcFace 原理 | 教程数学示例与实现对照 | verified：教程第4/7课与research/recognition.py；梯度smoke通过。 |
| 5.2 | MS-Celeb-1M 或其子集训练 ResNet50+ArcFace | 数据来源授权、ResNet50 配置、训练曲线与 checkpoint | blocked（指定数据）：未获得可核验MS-Celeb-1M训练包；已另做真实替代pilot，LFW非pairs的52身份/133图、身份交集0，3epoch的loss40.1119→36.12599，margin分类训练准确率仍0；`runs/arcface-pilot/`有日志/曲线/权重，不把替代数据算作原数据要求通过。 |
| 5.3 | LFW 验证，自训模型 >98.5% | 6,000 pairs 十折、阈值仅从训练折选择、均值/标准差、失败计数 | verified（评估流程），数值目标未通过：真实短训ArcFace在完整6000对上FP32为52.6667%±0.97468个百分点、INT8为53.0167%±1.60632个百分点，零失败；预训练SFace第二轮98.9333%属于另一模型，不能代替自训结果。 |
| 6.1 | 量化、剪枝、蒸馏 | 教程原理、收益与误区 | verified：教程第10课，不能把置零自动视为加速。 |
| 6.2 | 训练好识别模型的动态量化 | FP32/INT8 文件、相同输入精度、速度、大小 | verified（部分）：真实短训checkpoint已量化并做完整LFW；batch8中位104.477→105.454ms，大小95,398,463→94,612,859字节；`reports/arcface-pilot/comparison.json`。训练尚未收敛，量化只覆盖Linear，本次没有加速。 |
| 6.3 | ONNX 转换与推理 | ONNX checker、ORT 推理、数值差、输入输出规格 | verified：真实短训checkpoint导出opset17动态batch，checker/ORT通过，最大特征绝对误差8.79×10^-7，batch8中位155.836ms；3DDFA真实回归器也已导出运行。ONNX全LFW精度未单独计算，不能由数值误差直接认定。 |
| 6.4 可选 | ByteNN 模拟加速 | 清楚标为模拟的实现与比较 | verified（按原文模拟范围）：同一真实ONNX checkpoint，关闭/开启ORT图优化，batch1中位34.3693→25.2712ms，batch8为253.4583→175.5968ms；数值一致性通过。没有使用内部ByteNN SDK或测移动端。 |
| 7.1 | GAN、StarGAN、AttGAN 原理 | 教程生成章节与练习 | verified：教程第12课与训练实现；真实CelebA训练另列7.2。 |
| 7.2 | CelebA 训练 StarGAN，发色/年龄/性别属性 | 数据属性清单、训练日志、checkpoint、编辑结果 | verified（有限规模微调）：真实CelebA、作者预训练G/D，最终继承1200D/240G更新；512→4096配置池，2400次抽样并非去重数。三段协议、Adam恢复、seed43、中断至少150次丢弃更新均记录在[实验汇总](../reports/stargan-experiment/report.json)与[教程第8节](continued-experiments.md#8-用真实-celeba-数据完成-stargan-微调评估与桌面部署)。真实编辑和原生桌面接入通过；不称从零全量训练或论文复现。 |
| 7.3 | FID、IS 质量评估 | 特征器、输入规模、预处理、统计分数 | verified（限定协议）：torch-fidelity0.4.0/InceptionV3，前后各768源图+768独立真实参考、相同XPU生成/torch构建与CPU指标。FID41.272788→40.980400；IS3.088072±0.253799→3.098037±0.170242。图片级无交集，身份重叠未审计；小样本变化不证明显著提升。旧CPU前模型报告另外保留。 |
| 8.1 | 3DMM、NeRF 原理 | 教程三维章节 | verified：教程第14课。 |
| 8.2 | PRNet 或 3DDFA_v2 单图三维重建 | 实际权重推理、输入图、对应 OBJ/PLY | verified：真实3DDFA_V2，38365顶点/76073面，`reports/3d/face.obj`与reconstruction.json；无3D真值精度。 |
| 8.3 | PyTorch3D 或 OpenGL 渲染多视图 | 重建网格、渲染过程、至少三个视角 | verified（原生OpenGL）：pyglet2.1.16/OpenGL4.6/24位深度，GPU查询实际76073面；0°、+35°、−35°三个视角已保存，归零像素一致、子进程正常退出，见reports/3d-native/report.json。最终桌面按钮接入另见desktop-integration.json；未安装PyTorch3D。 |
| 9.1 | 关键点特效原理 | 教程坐标、透明度与跟踪章节 | verified：教程第11课与app.py。 |
| 9.2 | 实时关键点、动态贴纸 | 真实视频/摄像头路径、贴纸随脸移动、帧时 | verified（本机功能）：原生Tk+OpenCV直接摄像头640×480，两轮20秒149/316帧；停止/重开/退出释放通过。首轮7.4277FPS全部检出脸，第二轮15.7790FPS但仅49帧检出脸；不能把差异归因优化或称流畅度达标。 |
| 9.3 | 磨皮、美白、口红 | 可调节效果、结果视频、无脸行为 | verified：效果图片与视频，测试无脸行为；五点口红区域为近似，张嘴/大姿态有局限。 |

## 阶段与综合交付

| PDF 综合要求 | 需要的证据 | 状态 |
|---|---|---|
| 环境报告、Hello World、OpenCV 灰度程序 | 版本报告、可运行脚本与输出 | verified：环境JSON/freeze、Docker Hello与已执行Notebook |
| 数据分析报告与基线 | 数量、类别、失败数量、数据拆分和指标 | verified：LFW全量分布/评估、CelebA完整图像清单和40属性统计、完整WIDER数据/预训练验证；不同模型结果独立记录 |
| 难例挖掘/难例采样、大规模类别问题 | 原理说明及训练代码，不能仅出现术语 | verified：P-K/batch-hard实现、梯度检查与真实133图训练；性能收益未做独立消融，不能归因给难例策略 |
| 三类特效演示程序 | 明确属性编辑、三维与动态三大类 | verified（本机演示）：真实StarGAN有限微调与ONNX编辑、3DDFA三维/原生GL、关键点动态贴纸/美颜已集成；不代表所有模型从零训练或身份保持已验证。 |
| 统一界面、实时视频 | 原生桌面窗口、实际摄像头、模块集成 | verified（本机功能）：desktop.py直接调用摄像头、六效果、双图验证、真实StarGAN ONNX属性编辑与完整网格原生OpenGL；最新集成报告desktop-integration.json通过。摄像头两轮独立记录；网页仅辅助版本。 |
| CPU/移动端实时性 | 分辨率、设备、计时边界与FPS；移动端单独实测 | verified（本机测量）：保留两轮原生报告及有脸/无脸分层，首轮7.4277FPS/P95 159.3555ms，第二轮15.7790FPS/P95 168.6882ms；并行训练且输入不同，不称优化收益。CPU流畅度目标未通过，真实移动端未验证。 |
| 项目演示视频 | 可播放视频、实际处理过程与限制说明 | verified：最终原生程序43.2秒/216帧H.264演示，真实Tk控件与GL帧缓冲，只用公开样图；另保留18秒仿射特效视频。播放FPS不是算法吞吐。 |
| 项目总结报告、汇报 PPT | 本文档目录及实际 PPTX | verified（部分）：Markdown、教程/报告/问题PDF与PPTX已生成；最终视觉检查另记录 |
| StyleGAN3、DECA 必读、CNN/度量/GAN/扩散/3DGS/可微渲染 | 教程覆盖、原论文链接、练习与答案 | verified：18课完整教程与sources.md；不称所有模型都已训练 |
| BytePS、火山引擎、内部高效工具 | 原文要求的模拟、官方体验及准确范围 | verified（模拟/样例体验）：BytePS概念两进程加权SGD与全批参考最大误差4.47e-8；ByteNN概念由ORT模拟；火山引擎官方人像融合样例交互已体验。未使用内部SDK或部署鉴权云API |
| 全部源代码 GitHub 开源 | 公共仓库可读取、源码齐全、许可证、无私密数据 | verified（已有公开基线）：仓库已公开且沿用用户重命名后的地址；本次最终桌面源码与新增实验须按最终交付清单同步发布 |
| 面向高三零基础教程 | 专业概念、直观解释、运行步骤、练习答案 | verified：18课主教程、继续实验教程、原生桌面使用与代码教程，配完整源码汇编和文件用途 |
| 执行问题及解决方法 | 实际错误、根因、修改、复测与剩余限制 | verified：issues-and-fixes.md，包含版本兼容、网络/环境与旧代码审计 |
| 全部源码汇编与使用位置（新增要求） | 源码快照、模块/调用入口/PDF编号/验证命令 | verified：code-map.md、完整Markdown/HTML源码汇编及reports/code-inventory.json；最终发布前再按文件清单核对 |

桌面源码副本位于用户桌面的“智能视觉AI项目_完整源代码”文件夹；复制后按逐文件SHA256核对，最终清单见交付文件。

## 不能改变口径的四条规则

1. 测试训练代码能反向传播，只证明“训练程序能执行”；必须有训练数据、训练轮次与验证才能评价模型。
2. 预训练模型取得的性能归属于指定预训练模型与本次处理协议，不能归给本项目从零训练。
3. 可旋转的固定人脸模板展示三维坐标概念，不证明从输入图片恢复了该人的三维几何。
4. 未完成项必须保留；开源发布、视频或漂亮界面不能抵消未通过的科研验收项。
