# 官方资料、论文与许可核验

核验日期：2026-09-26。资料中的基准数字是作者的实验结果，不是本项目测量值。所有可变的软件版本以本机环境报告为准；网页使用滚动的 `main`/`stable` 时，复现实验应另记录提交号或安装版本。

## 原需求中需要校正的事实

- **OpenMMLab 不应写成“字节跳动开源项目”。**其官方介绍及教程列出的机构包括香港中文大学、商汤等；本项目应写“使用 OpenMMLab 开源视觉框架”。没有证据支持原 PDF 对字节内部使用范围的陈述。[OpenMMLab 官方教程](https://openmmlab.com/community/cvpr2021-tutorial)
- **BytePS 是实际存在的开源分布式训练框架。**官方仓库于 2025-12-08 归档，其说明要求 CUDA/NCCL，明确不支持纯 CPU 训练。一个单进程梯度平均示例只能叫原理模拟。[BytePS 官方仓库](https://github.com/bytedance/byteps)
- **ByteNN（模拟）不可写成已经安装了字节内部引擎。**本次未获得可核验的官方公开 SDK、授权或运行接口；ONNX Runtime 可以演示跨平台推理原理，但不等于 ByteNN。[字节跳动官方开源组织](https://github.com/bytedance)
- **MMClassification 的后续项目为 MMPreTrain。**学习旧文档时应留意工程名称与版本，而不要把新旧配置混装。[MMPreTrain 官方仓库](https://github.com/open-mmlab/mmpretrain)

## 环境与工程

| 资料 | 本项目用来核验的内容 |
|---|---|
| [Python 3.12 Tkinter](https://docs.python.org/3.12/library/tkinter.html) | 原生窗口、事件循环与线程模型；本机实际Tk8.6.12。Tk控件由主线程更新，后台线程通过队列提交结果。 |
| [OpenCV VideoCapture](https://docs.opencv.org/4.13.0/d8/dfe/classcv_1_1VideoCapture.html) | 摄像头打开、读取和释放的API语义；文档4.13，实际本机使用OpenCV5并实测DirectShow路径，不能混写版本。 |
| [Pillow ImageTk](https://pillow.readthedocs.io/en/stable/reference/ImageTk.html) | 将图像转换为Tk可显示的PhotoImage；实际Pillow12.3.0。 |
| [ONNX Runtime图优化](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html) | ORT图优化等级与后端限制；ByteNN概念模拟的公开实现依据。 |
| [PyTorch Intel GPU支持](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/get_start_xpu.md) | XPU官方软件栈与支持设备；实际兼容性由本机算子和训练反向传播检查补充。 |
| [Anaconda静默安装](https://www.anaconda.com/docs/getting-started/advanced-install/silent-mode) | 当前用户安装、PATH和默认Python选项；最终使用的NoRegistry/NoShortcuts由下载安装器帮助输出另核验。 |
| [MMDetection 3.3.0 版本断言](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/mmdet/__init__.py) | `2.0.0rc4 <= mmcv < 2.2.0`，`0.7.1 <= mmengine < 1.0.0`。并不是任意最新版本都兼容。 |
| [MMDetection 安装](https://github.com/open-mmlab/mmdetection/blob/main/docs/en/get_started.md) | Python、PyTorch、MMCV、MMEngine 的安装顺序及推理流程。 |
| [MMCV 安装](https://github.com/open-mmlab/mmcv/blob/main/docs/en/get_started/installation.md) | wheel 必须匹配操作系统、Python、PyTorch、CUDA；下载 `.tar.gz` 表示可能进入源码编译。`mmcv-lite` 不提供全部编译算子。 |
| [PyTorch 量化](https://docs.pytorch.org/docs/stable/quantization.html) | 动态/静态/QAT 的区别；旧 eager API 与新量化工具的迁移，需与安装版本对应。 |
| [PyTorch ONNX 导出](https://docs.pytorch.org/docs/stable/onnx.html) | 导出器、算子集与动态形状；转换后必须运行数值一致性检查。 |
| [OpenCV YuNet 模型说明](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | 轻量人脸检测与五点定位；模型目录的单独许可证。 |
| [OpenCV SFace 模型说明](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) | 特征提取、对齐与相似度；模型说明中的 Apache-2.0 声明。 |
| [SFace 权重溯源待澄清 issue](https://github.com/opencv/opencv_zoo/issues/313) | 2026-09-26 核验时仍有关于准确权重训练数据来源的未答问题；本项目不额外保证一切商业用途。 |
| [InsightFace 官方许可](https://github.com/deepinsight/insightface) | 代码 MIT 与公开预训练模型的非商业研究许可分别处理；不把“开源代码”解释为“权重无限制”。 |
| [PyTorch3D 安装](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) | PyTorch/CUDA/C++ 扩展匹配，不能假设任意 Windows wheel 都存在。 |

## 数据集

| 数据集 | 官方/维护方来源 | 获取与使用边界 |
|---|---|---|
| CelebA | [香港中文大学项目页](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)；[torchvision 下载器](https://github.com/pytorch/vision/blob/main/torchvision/datasets/celeba.py) | 项目页列出 202,599 张、10,177 个身份、40 属性、5 关键点。非商业研究，禁止再分发图像与数据；身份标注说明有研究申请条件。只在本地存储合法获取的数据，开源仓库发布加载器。 |
| LFW | [原站](http://vis-www.cs.umass.edu/lfw/)；[scikit-learn 官方下载器与 SHA256](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/datasets/_lfw.py) | 原站本次网页工具不可达；维护方下载器提供 Figshare 原图与 pairs 镜像。13,233 图、5,749 身份；评估必须说明原图/对齐版、检测失败与十折协议。 |
| MS-Celeb-1M | [微软研究历史项目页](https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-million-celebrities-real-world/)；[原始论文](https://arxiv.org/abs/1607.08221) | 历史页保留 2016 年竞赛说明，不是本次有效下载/授权凭证。本项目没有获得可核验的授权训练包，不从未知镜像补齐。若用合法自建身份数据，必须在报告写成替代数据实验。 |
| WIDER FACE | [作者论文](https://arxiv.org/abs/1511.06523)；[作者项目页](http://shuoyang1213.me/WIDERFACE/)；[torchvision 官方下载器](https://github.com/pytorch/vision/blob/main/torchvision/datasets/widerface.py) | 原站本次网页工具不可达；维护方下载器保留 Google Drive 入口。需训练图、标注与独立验证图；不能把几张演示图的 AP 写成 WIDER 全集 mAP。 |
| 300-W | [Imperial College iBUG 官方页](https://ibug.doc.ic.ac.uk/resources/300-W/) | 研究用途、禁止商业训练；68 点，原标注从 1 开始计数；评估 NME 使用外眼角距离。官方页中的 600 张 challenge test 与常见训练集不是同一个拆分，必须记清。 |
| COFW | [当前官方CaltechDATA记录](https://data.caltech.edu/records/bc0bf-nc666)；[DOI 10.22002/D1.20099](https://doi.org/10.22002/D1.20099)；[Perona Lab数据目录](https://www.vision.caltech.edu/datasets/) | 旧作者页面已不可达，以当前记录为准。官方灰度包COFW.zip MD5为e092eeab9d0790674f86047880410e5a，彩色包为8b21d126c4e1fb307cb463578eef0511。本次普通下载中断/校验失败，未用于训练，见reports/cofw-access.json。原始29点与300-W的68点不可混用。 |

后续实际获取：WIDER采用作者页面明确链接的 [CUHK-CSE维护镜像](https://huggingface.co/datasets/CUHK-CSE/wider_face)，按固定提交和ZIP Range下载32训练图/16验证图，数据在本地保留，镜像标示CC BY-NC-ND-4.0；统计与来源清单随实验记录。300-W训练相关入口为 [官方facial point annotations页面](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/)，下载表单需要姓名、邮箱和机构，未代填虚构身份。不要把可直接下载的600张challenge测试图当成正式训练集。

CelebA原Google Drive下载出现配额限制后，本次采用[StarGAN官方项目download.sh](https://github.com/yunjey/stargan/blob/master/download.sh)明确链接的[配套图片包](https://www.dropbox.com/s/d1kjpkqklf0uw77/celeba.zip?dl=1)和[128×128五属性预训练权重包](https://www.dropbox.com/s/7e966qq0nlxwte4/celeba-128x128-5attrs.zip?dl=1)。这是算法作者项目的配套下载源，不是CelebA数据作者官方镜像。完整ZIP未有作者公布哈希，记录本次观察SHA256和ZIP CRC；内部原属性表MD5与torchvision公布值75e246fa4810816ffd6ee81facbd244c一致。完整202599张图片文件名与属性表对应。原图、生成图和权重仅保留本地，未因代码开源而更改数据许可。

StarGAN实验协议依据[官方data_loader.py](https://github.com/yunjey/stargan/blob/master/data_loader.py)的seed1234打乱、前1999张测试划分，以及[官方solver.py](https://github.com/yunjey/stargan/blob/master/solver.py)中的BCE按batch归一化、patch梯度惩罚和测试运行方式。复原当前作者代码不等于独立审计公开权重的历史训练数据。指标使用[torch-fidelity官方实现](https://github.com/toshas/torch-fidelity)与其[Inception兼容特征器](https://github.com/toshas/torch-fidelity/blob/master/torch_fidelity/feature_extractor_inceptionv3.py)所列权重，实际安装0.4.0。前后主比较均XPU生成、同torch2.14+xpu构建、CPU计算FID/IS；旧CPU前模型报告保留为历史。特征权重SHA256、输入规模与参数见reports/stargan-experiment/report.json。

LFW 镜像校验值来自 scikit-learn 维护代码：

```text
lfw.tgz: https://ndownloader.figshare.com/files/5976018
SHA256: 055f7d9c632d7370e6fb4afc7468d40f970c34a80d4c6f50ffec63f5a8d536c0
pairs.txt: https://ndownloader.figshare.com/files/5976006
SHA256: ea42330c62c92989f9d7c03237ed5d591365e89b3e649747777b70e692dc1592
```

## 必读论文与扩展阅读

1. [RetinaFace: Single-stage Dense Face Localisation in the Wild](https://arxiv.org/abs/1905.00641)：检测、五点与多任务监督；读“输出是什么、多个任务如何共同训练”。
2. [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)：把类别分离转换到归一化特征的角度空间；注意论文发表年份与预印本年份不同。
3. [StyleGAN3 / Alias-Free GANs](https://arxiv.org/abs/2106.12423)；[NVIDIA 官方实现](https://github.com/NVlabs/stylegan3)：减少纹理黏在图像坐标上的现象；使用旧 StyleGAN2 权重并不会自动得到 StyleGAN3 效果。
4. [DECA](https://arxiv.org/abs/2012.04012)：从野外图像估计可动画化的详细三维人脸。
5. [StarGAN 官方实现](https://github.com/yunjey/stargan)：多个属性域共享生成器与判别器。
6. [PRNet 官方实现](https://github.com/yfeng95/PRNet)：位置图回归的三维重建。[3DDFA_V2 官方实现](https://github.com/cleardusk/3DDFA_V2)：三维稠密对齐。模型、基底数据与代码许可需要分别看。
7. [DDPM](https://arxiv.org/abs/2006.11239)：逐步加噪与学习去噪。[NeRF](https://arxiv.org/abs/2003.08934)：从视图学习空间密度和颜色。[3D Gaussian Splatting](https://arxiv.org/abs/2308.04079)：用可优化三维高斯表示和渲染场景。
8. [FID 的原始论文 TTUR](https://arxiv.org/abs/1706.08500)：比较特征分布；图像少、特征器不同或预处理不同的分数不可直接比较。
9. [HRNet](https://arxiv.org/abs/1908.07919)：保留高分辨率与跨尺度融合。[SAN](https://arxiv.org/abs/1803.04108)：用风格聚合减少输入风格变化对关键点检测的影响。[AttGAN](https://arxiv.org/abs/1711.10678)：属性约束与保留非目标内容的生成研究。
10. [Khronos WebGL 官方说明](https://www.khronos.org/webgl/)：WebGL基于OpenGL ES，使用浏览器图形接口；生成查看器HTML不等于已在目标GPU/浏览器渲染验收。

## 引用与发布规则

本项目原创源码以仓库 LICENSE 为准。第三方依赖、下载权重、样本数据、模型基底和生成结果不自动继承该许可。保留第三方原始许可证与来源；GitHub 只上传允许发布的代码、文档、配置和演示素材，不上传数据集、人脸库、访问令牌、未核清许可权重或用户摄像头照片。下载链接可变，下载失败应写真实错误，不填写预设成功结果。
