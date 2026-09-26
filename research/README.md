# 真实研究流水线与边界

在项目根目录运行，Windows 使用 `.venv\Scripts\python.exe` 替代下列 `python`。本目录不会下载受限数据、用随机图片伪装数据集，或在没有评估时输出 98.5%。`smoke` 只验证张量、梯度和文件格式；不能作为训练成果。

## 已验证入口

```powershell
python -m unittest discover -s tests -p test_research.py -v
python -m research.smoke --end-to-end --output reports/research-smoke.json
python -m research.optimize --synthetic-smoke --output reports/optimization-smoke --threads 2 --repeats 5
python -m research.lfw --backend sface --root data/lfw/lfw --pairs data/lfw/pairs.txt --models models --on-failure count-incorrect --threads 2 --output reports/lfw-sface.json
python -m research.lfw --backend sface --root data/lfw/lfw --pairs data/lfw/pairs.txt --models models --face-policy largest --on-failure count-incorrect --threads 2 --output reports/lfw-sface-largest.json
```

两个评估命令均使用全部官方 6000 对，不跳过失败样本。十折依官方顺序，每次用另外九折选择识别阈值，再在未参与选阈值的一折计分。失败对始终算错误；TAR/FAR 中缺特征不会被接受。它们评估的是使用外部训练数据的预训练 SFace，不是本项目从零训练的 ArcFace 模型。

第一次把应用中的「恰好一张脸」规则直接用于 LFW，得到 **72.25%**。排查发现 1175 张图片含背景人物，全部失败都属于多脸而非无脸。LFW 指定的是照片中的主体，因此第二次采用固定几何规则：选最大人脸框，面积相同选中心最近的人脸。检测阈值仍为 0.8，规则不读取身份或配对标签。第二次得到 **98.9333% ± 0.3887 个百分点**（十折均值与标准差），6000 对全部计入、7701 张唯一图像全部成功。TAR 均值 97.7667%，训练折校准目标 FAR 为 0.001，测试 FAR 均值同为 0.001。原始两个 JSON 报告均保留。第二次是排查后的预处理实验，独立最终认证还需要新测试集；SFace 训练身份与 LFW 是否重叠未独立审计。应用仍保留严格单脸规则，避免用户输入照片目标不明确。

余弦相似度为 $s=\frac{f_1\cdot f_2}{\lVert f_1\rVert\lVert f_2\rVert}$；特征已归一化时就是点积。以 $s\ge t$ 判为同一人。Accuracy 是全部配对判对的比例；TAR 是同人配对被接受的比例；FAR 是不同人配对被误接受的比例。每折仅 300 个负样本，FAR 的单折分辨率是 $1/300$，所以低 FAR 估计不应过度解读。

## ResNet50、ArcFace 与困难样本

先准备合法取得且按身份分文件夹的**对齐人脸**：`data/identities/Alice/*.jpg`、`data/identities/Bob/*.jpg`。不要拿 LFW 测试图像训练。默认每批 P=8 个身份、每身份 K=4 张不同照片。少量数据需明确减小 P/K，仍要求每个身份至少 K 张。

```powershell
python -m research.recognition --data data/identities --output runs/arcface --epochs 20 --p 8 --k 4 --device cpu
python -m research.lfw --root data/lfw-aligned --pairs data/lfw/pairs.txt --checkpoint runs/arcface/last.pt --output reports/lfw-arcface.json
python -m research.optimize --checkpoint runs/arcface/last.pt --images data/representative-aligned --output runs/optimized --lfw-root data/lfw-aligned --pairs data/lfw/pairs.txt
```

输入 RGB，112×112，归一化到 [-1,1]；特征输出 L2 归一化。损失为 ArcFace 分类交叉熵加 0.1 倍 batch-hard triplet。ResNet50 从随机权重开始。`history.csv` 的 train_accuracy 是带 margin 的训练分类准确率，**不是** LFW 验证准确率。`last.pt` 含模型、ArcFace 分类头、优化器和 epoch；当前最小 CLI 不提供自动断点续训。

ArcFace 将正确类别的 logit 从 $s\cos\theta_y$ 改为 $s\cos(\theta_y+m)$，其他类别仍为 $s\cos\theta_j$；这里 $m=0.5$ 弧度、缩放 $s=64$。代码对接近 $\pi$ 的角度使用单调回退，避免角度越大反而得分越高。交叉熵 $-\log\frac{e^{z_y}}{\sum_j e^{z_j}}$ 要求正确类别分数超过其他类别。Triplet 损失是 $\max(0,d(a,p_{hard})-d(a,n_{hard})+0.2)$：在当前批次中选最远的同人和最近的异人，推动同人靠近、异人远离。这里 $d$ 是归一化特征间欧氏距离。

当前额外执行一个真实小样本试验：从 LFW 中选出**完全未出现在官方 pairs 的身份**，共 52 个身份、133 张照片，身份交集为零；它不是 MS-Celeb-1M 训练验收。训练图与测试图均使用相同的 YuNet 最大主体选择和五点对齐，保存 112×112 JPEG，质量参数 100。`reports/lfw-pilot-data.json` 保存身份列表、原图与对齐图 SHA256、划分规则和零交集证据。训练结果弱也必须保留。

```powershell
python -m research.prepare_lfw_pilot
python -m research.recognition --data data/lfw-pilot-train --output runs/arcface-pilot --epochs 3 --p 4 --k 2 --embedding-dim 128 --lr 0.01 --triplet-weight 0.1 --device cpu --threads 2
python -m research.lfw --root data/lfw-aligned --pairs data/lfw/pairs.txt --checkpoint runs/arcface-pilot/last.pt --batch-size 32 --threads 2 --output reports/lfw-arcface-pilot.json
python -m research.optimize --checkpoint runs/arcface-pilot/last.pt --images data/lfw-pilot-train --output runs/arcface-pilot/optimized --samples 8 --threads 2 --repeats 10
```

`history.csv` 记录每个 epoch 的损失、训练分类准确率、抽样次数；P-K 按批独立抽样，`samples` 是采样次数，不等于去重图片数。自动生成 `training-curves.svg`，曲线点来自 CSV 的实际数值。`config.json` 保存全部命令参数。

动态量化只覆盖最后的 Linear，卷积保持 FP32，所以文件不一定显著缩小、速度也可能更慢。`comparison.json` 分开记录文件大小、延时、特征数值误差、可选真实 LFW 精度。仅数值误差不能证明识别精度不变。ONNX 使用 opset 17、动态 batch，导出后经 checker 和 ONNX Runtime 比对。当前 PyTorch 的旧动态量化和 TorchScript ONNX 导出均有弃用警告，但已实际运行；升级时优先重跑验证，再考虑 torchao/新导出器。

## 300W 关键点与仿射对齐

采用可训练的 ResNet18 坐标回归基线，**不是 HRNet/SAN**。先将 300W 官方划分分别放在 train/val；`.pts` 的 68 个点按原始一基坐标转零基。裁剪框由标注点加 20% 边界构成。训练和验证不得混用同一张图。

```powershell
python -m research.landmarks prepare --root data/300w/train --output data/300w/train.json
python -m research.landmarks prepare --root data/300w/val --output data/300w/val.json
python -m research.landmarks train --root data/300w/train --train data/300w/train.json --val-root data/300w/val --val data/300w/val.json --output runs/landmarks
python -m research.landmarks infer --checkpoint runs/landmarks/last.pt --image assets/face.jpg --box 30 20 230 240 --output runs/landmark-example
```

NME = 每个点的平均像素误差 / 眼睛外眼角距离（68 点索引 36 与 45），以原图像素计算，避免非等比缩放扭曲分母。它与使用瞳距/包围盒归一化的数字不可直接比较。推理输出 `landmarks.png`、`aligned.png` 与坐标 JSON；`--box` 应使用与训练相近的人脸裁剪。对齐从 68 点提取双眼中心、鼻尖、两个嘴角，估计相似仿射矩阵。

## WIDER FACE 与 MMDetection

为验证本机确实能训练，提供可复现的真实小子集下载器：

```powershell
python -m research.fetch_wider_subset --train 32 --val 16 --seed 42 --output data --report reports/wider-subset.json
```

从 CUHK-CSE 发布的 WIDER FACE 镜像，按固定种子在**原始 train/val 各自内部**抽样，取得 32 张训练图（242 个有效框）和 16 张验证图（153 个有效框）。HTTP Range 只取 ZIP 中选中的图片，总图像传输约 15 MB，避免下载 1.8 GB 两个完整压缩包；官方标注 ZIP 另约 3.6 MB，校验完整 SHA256。图片逐项由 ZIP CRC32 校验，并记录独立 SHA256。报告明确标注：完整图片 ZIP 的 SHA256 是服务器公布值，未因只下载局部而声称验证了整个 ZIP。数据遵循原有 CC BY-NC-ND-4.0，不随项目代码许可证改变，不上传到源代码仓库。

子集工具已经输出 `data/wider/train.json`、`val.json`，以下转换命令供完整数据使用：

```powershell
python -m research.data --images data/WIDER_train/images --annotations data/wider_face_train_bbx_gt.txt --output data/wider/train.json
python -m research.data --images data/WIDER_val/images --annotations data/wider_face_val_bbx_gt.txt --output data/wider/val.json
```

转换器检查路径、重复文件、框长度、图像边界；排除标记 invalid 的框，并保留零人脸图像。`research/configs/wider_retinanet.py` 是 MMDetection 3.x RetinaNet ResNet50-FPN 单类配置，**RetinaNet 不等于 RetinaFace**。

MMDetection 依赖带编译算子的 MMCV，单独环境安装，避免与 Windows 主演示环境混装。本项目独立 Docker 实验使用 PyTorch 2.1、MMCV 2.1、MMDetection 3.3，具体锁定版本与实际训练记录见 `reports/wider_pilot.json` 和 `scripts/mmdet_pilot.py`。安装必须验证 `import mmcv.ops`，不能仅凭 pip 成功宣布就绪。

在已安装且验证的 MMDetection checkout 中，用其标准 `tools/train.py <本项目绝对路径>/research/configs/wider_retinanet.py` 训练，用 `tools/test.py <同一配置> <checkpoint>` 评估。当前项目配置依赖 `mmdet::` 包配置解析，工作目录必须为本项目根目录才能正确读取 `data/`。这里报告的是 COCO bbox AP，不能冒充 WIDER 官方 Easy/Medium/Hard AP；后者还需要 WIDER 官方评估工具和子集标注。

## StarGAN 属性编辑

提供实际的条件生成器、判别器、Wasserstein 损失、梯度惩罚、属性分类和循环重建训练循环。CelebA 属性文件与官方分区文件均必需，训练只读分区 0，拒绝缺图与缺标注。默认属性为黑发、金发、棕发、Male、Young；这些是数据集标签，不代表系统能够可靠判断个人性别或年龄。

令 $G(x,c)$ 为把图片 $x$ 编辑成属性 $c$ 的生成器，$D_{src}$ 为判别器的真实度分数，$D_{cls}$ 为属性分类器。判别器最小化 $E[D_{src}(G(x,c))]-E[D_{src}(x)]+L_{cls,real}+10L_{GP}$；生成器最小化 $-E[D_{src}(G(x,c))]+L_{cls,fake}+10\lVert G(G(x,c),c_{original})-x\rVert_1$。其中 $L_{GP}=E[(\lVert\nabla_{\hat x}D_{src}(\hat x)\rVert_2-1)^2]$，$\hat x$ 是真假图像之间的随机插值。分类损失为多标签二元交叉熵；目标标签从真实批次打乱得到，保持合法属性组合。

```powershell
python -m research.stargan train --root data/celeba/img_align_celeba --labels data/celeba/list_attr_celeba.txt --partition data/celeba/list_eval_partition.txt --output runs/stargan --device cpu
python -m research.stargan generate --checkpoint runs/stargan/last.pt --image assets/face.jpg --targets 0,1,0,0,1 --output runs/stargan/blond.png
python -m research.stargan metrics --real data/celeba-heldout --generated runs/generated-heldout --output reports/stargan-quality.json
```

默认 100000 次判别器更新，每五步更新一次生成器。CPU 完整训练非常慢；应先验证小批次，再为正式实验准备 GPU 和充足时间。真实 CelebA、正式权重、生成效果和 FID/IS 目前未获得训练证据。`latest_grid.png` 只是当前训练状态的可视化，早期结果可能是噪声。`metrics` 选用 `torch-fidelity`（需单独安装并获得 Inception 权重），不是自造 FID；要记录图片数量、选择方法、裁剪、分辨率和随机种子。至少十张只是运行条件，可信比较需远多于十张；IS 使用 ImageNet 类别，在人脸任务有局限。

## 参考依据

- [PyTorch 版本配对与官方安装索引](https://docs.pytorch.org/get-started/previous-versions/)
- [MMDetection 自定义数据集](https://mmdetection.readthedocs.io/en/latest/advanced_guides/customize_dataset.html)
- [ArcFace 论文](https://arxiv.org/abs/1801.07698)
- [StarGAN 官方源码](https://github.com/yunjey/stargan)
- [LFW 论文与数据集主页](https://vis-www.cs.umass.edu/lfw/)
- [torch-fidelity 官方项目](https://github.com/toshas/torch-fidelity)
