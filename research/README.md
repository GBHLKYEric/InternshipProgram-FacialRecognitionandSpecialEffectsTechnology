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
python -m research.optimize --checkpoint runs/arcface-pilot/last.pt --images data/lfw-pilot-train --output runs/arcface-pilot/optimized --samples 8 --threads 2 --repeats 10 --lfw-root data/lfw-aligned --pairs data/lfw/pairs.txt
```

`history.csv` 记录每个 epoch 的损失、训练分类准确率、抽样次数；P-K 按批独立抽样，`samples` 是采样次数，不等于去重图片数。自动生成 `training-curves.svg`，曲线点来自 CSV 的实际数值。`config.json` 保存全部命令参数。

本机真实 pilot 的三轮损失为 40.111934、36.924611、36.125987，带 margin 的训练分类准确率均为 0。完整 LFW 的 FP32 准确率为 52.6667%（十折标准差 0.97468 个百分点），Linear 动态 INT8 为 53.0167%（标准差 1.60632 个百分点），均无失败配对。该微小差异不能证明量化提高了准确率，模型也没有达到 98.5%。批量 8、2 线程、10 次计时的中位数分别为 FP32 104.477 ms、INT8 105.454 ms、ORT 155.836 ms；当前设置下两条优化路线都没有加速。完整数值、文件大小、模型与配对哈希见 `reports/arcface-pilot/comparison.json`，真实训练曲线与配置也保存在该目录。

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

本机尚未获得这部分真实训练数据。300-W 原始下载需要填写实际姓名、联系邮箱和机构，不能捏造登记。COFW 的[当前官方 CaltechDATA 页面](https://data.caltech.edu/records/bc0bf-nc666)公开可读，但实际彩色下载在 1086044 字节处中断，灰度下载只取得 293760 / 178022021 字节；校验失败后保留 `.part`，没有解压或用于训练。普通断点续传也遇到 HTTP 500 或 SSL EOF；Chrome 普通下载曾在官方存储跳转处显示 `ERR_BLOCKED_BY_CLIENT`，没有调整安全设置。详细尝试见 `reports/cofw-access.json`。因此不能给出真实 COFW/300-W NME；现有 68 点代码也不能直接宣称支持 COFW 原始 29 点标注。恢复官方正常下载后还需要确认 MATLAB 字段、点序、坐标原点与评估分母，再完成独立训练/测试。

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

提供实际的条件生成器、判别器、Wasserstein 损失、梯度惩罚、属性分类和循环重建训练循环。属性表与分区表均必需，训练只读分区 0，拒绝缺图与缺标注。分区可以来自 CelebA 官方文件，也可以使用有完整来源记录的研究划分；本次采用 StarGAN 作者代码的划分，不能称作 CelebA 官方分区。默认属性依次为黑发、金发、棕发、Male、Young；这些是数据集标签，不代表系统能够可靠判断个人性别或年龄。

令 $G(x,c)$ 为把图片 $x$ 编辑成属性 $c$ 的生成器，$D_{src}$ 为判别器的真实度分数，$D_{cls}$ 为属性分类器。判别器最小化 $E[D_{src}(G(x,c))]-E[D_{src}(x)]+L_{cls,real}+10L_{GP}$；生成器最小化 $-E[D_{src}(G(x,c))]+L_{cls,fake}+10\lVert G(G(x,c),c_{original})-x\rVert_1$。其中 $L_{GP}=E[(\lVert\nabla_{\hat x}D_{src}(\hat x)\rVert_2-1)^2]$，$\hat x$ 是真假图像之间的随机插值。分类损失为多标签二元交叉熵；目标标签从真实批次打乱得到，保持合法属性组合。

```powershell
python -m research.stargan train --root data/celeba/img_align_celeba --labels data/celeba/list_attr_celeba.txt --partition data/celeba/list_eval_partition.txt --output runs/stargan --device cpu
python -m research.stargan generate --checkpoint runs/stargan/last.pt --image assets/face.jpg --targets 0,1,0,0,1 --output runs/stargan/blond.png
python -m research.stargan metrics --real data/celeba-heldout --generated runs/generated-heldout --output reports/stargan-quality.json
```

默认 100000 次判别器更新，每五步更新一次生成器。一个 step 表示一次判别器参数更新，不是遍历完整数据集的一个 epoch。当前分类项按官方实现对属性求和、仅按 batch 大小平均；梯度惩罚先对判别器全部 patch 输出求和再求输入梯度，不能提前平均 patch 改变梯度尺度。`latest_grid.png` 是训练状态可视化，单独看它不能证明编辑质量。

日志中的 D/G loss 可以为负：Wasserstein 判别器输出的是不受 [0,1] 限制的实数分数，不是概率；其真假分数差也不是普通分类准确率。GAN 中两个网络的目标同时变化，因此不能把某一条 loss 单调下降当成唯一成功标准。重建项非负，仍只能说明来回编辑后有多接近输入，不能单独保证目标属性真的改变。

本机已经取得 StarGAN 官方项目配套下载源中的完整 CelebA 包和作者的 200000-step G/D 权重。原 CelebA Google Drive 链接出现公开下载配额限制，没有规避限制。配套 ZIP 包没有作者公布的完整压缩包哈希，因此记录本次观察到的 SHA256/MD5 与逐文件 CRC；其内部原始属性表的 MD5 与 torchvision 发布的 CelebA 原始 MD5 一致。完整 ZIP 中 202599 张图片与属性表文件名集合一致，40 个属性的阳性计数见 `reports/celeba-expanded-data.json`。来源应称“StarGAN 官方项目配套下载源”，不能称 CelebA 数据作者官方镜像。所有原图、生成图和权重仅供本地非商业研究，不能随开源代码再分发。

准备命令从原始属性顺序出发，使用 `random.Random(1234)` 打乱，前 1999 张为作者代码中的测试部分，剩余为训练部分。取训练池前 4096 张；测试部分前 768 张作生成源，随后 768 张作真实参考，三组图片文件名无交集，逐图记录 SHA256。这是按图片划分，不保证人物身份无交集；复原作者代码也不能独立证明其公开检查点的历史训练数据。

```powershell
python -m research.fetch_official_data stargan-official --file celeba.zip --output data/stargan-official --report reports/celeba-stargan-author-download.json
python -m research.fetch_official_data stargan-official --file celeba-128x128-5attrs.zip --output data/stargan-official --extract --report reports/stargan-pretrained-download.json
python -m research.stargan_pilot prepare --train 4096 --evaluation 768 --output data/celeba-expanded --report reports/celeba-expanded-data.json
```

`prepare` 直接从完整 ZIP 按需提取，不必解压全部 20 万个小文件。作者权重严格载入匹配的网络结构：G 的宽度为 64、残差块为 6，D 为 6 层下采样，图片为 128×128，属性顺序固定。实际训练使用独立 XPU 环境，Intel Arc 130T 上已验证 WGAN-GP 所需的二阶自动微分；CPU 主环境与 XPU 环境不可互相覆盖安装。

本次微调先在 512 张池上运行 200 D / 40 G 更新，再扩展到 4096 张池继续训练。先导阶段采用分类 BCE 全元素平均、GP 对平均 patch 求梯度，属于与官方权重不同的教学损失。扩展阶段已修正为官方归一化。扩展运行曾被用户打断：日志记录到 650 步，最后完整检查点只含其中 500 步，后面的日志更新不计入最终模型。从该检查点恢复模型和两个 Adam 优化器，以 seed 43 重新开始数据顺序，继续 500 D / 100 G 更新；不是恢复原数据迭代器的精确重放。所有原始日志保留，不能为了报告累计步数而删除中断差异。

```powershell
# 以下 python 应指向已验证的 XPU 环境；本机为 ../../work/xpu-env/Scripts/python.exe
python -m research.stargan_pilot prepare --train 512 --evaluation 128 --output data/celeba-pilot --report reports/celeba-pilot-data.json
python -m research.stargan train --root data/stargan-official/celeba/images --labels data/celeba-pilot/attributes.txt --partition data/celeba-pilot/partition.txt --output runs/stargan-pilot --device xpu --threads 2 --size 128 --width 64 --blocks 6 --batch-size 2 --steps 200 --n-critic 5 --lr .00001 --log-interval 10 --save-interval 100 --init-generator data/stargan-official/200000-G.ckpt --init-discriminator data/stargan-official/200000-D.ckpt --loss-protocol legacy
# 复现保存下来的前500步即可；原命令计划1000步，但650步后中断。
python -m research.stargan train --root data/stargan-official/celeba/images --labels data/celeba-expanded/attributes.txt --partition data/celeba-expanded/partition.txt --output runs/stargan-expanded --device xpu --threads 2 --size 128 --width 64 --blocks 6 --batch-size 2 --steps 500 --n-critic 5 --lr .00001 --log-interval 50 --save-interval 250 --continue-checkpoint runs/stargan-pilot/last.pt --loss-protocol official
python -m research.stargan train --root data/stargan-official/celeba/images --labels data/celeba-expanded/attributes.txt --partition data/celeba-expanded/partition.txt --output runs/stargan-resumed --device xpu --threads 2 --size 128 --width 64 --blocks 6 --batch-size 2 --steps 500 --n-critic 5 --lr .00001 --seed 43 --log-interval 50 --save-interval 100 --continue-checkpoint runs/stargan-expanded/last.pt
python -m research.stargan_pilot evaluate --checkpoint data/stargan-official/200000-G.ckpt --manifest data/celeba-expanded/manifest.json --attributes data/celeba-expanded/attributes.txt --output runs/stargan-before-xpu --report reports/stargan-before.json --device xpu --threads 2
python -m research.stargan_pilot evaluate --checkpoint runs/stargan-resumed/last.pt --manifest data/celeba-expanded/manifest.json --attributes data/celeba-expanded/attributes.txt --output runs/stargan-after --report reports/stargan-after.json --device xpu --threads 2
```

评估前逐图重新核对清单哈希，并拒绝训练/生成源/真实参考图重叠。前后两个模型使用同一生成源、同一真实参考和同一目标标签：第 i 张生成图的目标属性等于第 i 张真实参考的属性，保持目标标签分布一致。图像先 RGB、中心裁剪 178×178，再双线性缩放到 128×128、像素变换到 [-1,1]。推理中的 InstanceNorm 使用每张图自身统计量，与官方 `Solver.test` 的行为一致；直接 `eval()` 使用旧权重的 running statistics 曾产生色偏与棋盘纹理，现已禁用这些运行缓冲区，并有回归测试。该修复同时用于前后模型，不能把错误归一化的结果作为微调前基线。

`metrics` 使用 torch-fidelity 0.4.0 的 InceptionV3 兼容实现，FID 使用 2048 维特征，IS 使用分类 logits，10 个划分、随机种子 2020。FID 比较两组特征的均值与协方差，越低通常表示分布越接近；IS 同时奖励单图分类明确、全体类别多样，但其 ImageNet 类别对人脸编辑任务并不充分。768 张的协方差估计仍很不稳定，不能与论文大样本 FID 直接横比。只有同协议前后数值与图像共同支持时才能讨论变化，不能仅凭损失下降称质量提高。预训练模型的既有能力也不能记作本次从零训练成果。

`research.stargan_deploy` 将最终 G 导出为固定 FP32 ONNX：`images=[1,3,128,128]`、`attributes=[1,5]`，输出 `edited=[1,3,128,128]`。导出先做 ONNX checker，再用真实图片比对 PyTorch/ORT，报告最大绝对误差、线程数、延时、模型大小和检查点哈希，模型路径为 `runs/stargan-deploy/generator.onnx`，验证依据为 `reports/stargan-deployment.json`。这证明部署数值相符，不证明属性编辑总是成功，也不证明身份保持。

```powershell
python -m research.stargan_deploy --checkpoint runs/stargan-resumed/last.pt --image data/stargan-official/celeba/images/028136.jpg --threads 2 --repeats 5
python -m research.stargan_pilot summarize
```

本机实际导出的模型为 33736691 字节，真实图最大绝对误差为 1.96695×10⁻⁶，平均绝对误差为 2.02974×10⁻⁷。CPU 两线程、五次 ORT 测量中位数 281.333 ms；当时有其他实验同时运行，这不是独占硬件基准。导出器的 `instance_norm train=True` 警告表示该算子使用输入自身统计量，并不表示导出时在更新模型参数；相同归一化模式已经过真实图数值对照。

完整的前后实测已完成，`reports/stargan-before.json` 与 `reports/stargan-after.json` 分别记录检查点 SHA256、样本数、预处理、特征模型、设备与指标配置。汇总及保留/丢弃日志、实际曲线见 `reports/stargan-experiment/`。

| 同一 768 源图 + 768 独立参考图协议 | 作者预训练模型 | 本次微调后模型 |
|---|---:|---:|
| FID，较低通常更接近参考分布 | 41.272788 | 40.980400 |
| IS 均值 | 3.088072 | 3.098037 |
| IS 十个划分的标准差 | 0.253799 | 0.170242 |

FID 下降 0.292388，IS 均值增加 0.009965。这只是该固定小样本协议的观察变化，没有统计显著性检验，不能宣称普遍质量提升。IS 的划分标准差也不是模型性能差的置信区间。真实图网格中可见编辑能力和部分生成伪影；其大部分既有能力来自作者预训练。最后 500 步续训实际耗时 372.452 秒，XPU 峰值已分配内存 1143694848 字节；这个耗时含数据处理、G 更新与保存检查点，不能解释为单独 D 前向延时。最终模型实际继承 1200 D / 240 G 更新，原作者检查点本来的更新不计入本次训练量。

主表中的前后模型均使用 XPU 生成、torch 2.14.0+xpu 构建、CPU 指标设备，控制硬件与软件变量。首次前模型曾使用 CPU 生成和 CPU torch 构建，得到 FID 41.272913、IS 3.088127；原报告保存在 `reports/stargan-before-cpu.json`，仅作为历史记录。汇总器会检查生成设备、指标设备、torch 构建和指标参数一致后才生成主比较。

## 参考依据

- [PyTorch 版本配对与官方安装索引](https://docs.pytorch.org/get-started/previous-versions/)
- [MMDetection 自定义数据集](https://mmdetection.readthedocs.io/en/latest/advanced_guides/customize_dataset.html)
- [ArcFace 论文](https://arxiv.org/abs/1801.07698)
- [StarGAN 官方源码](https://github.com/yunjey/stargan)
- [LFW 论文与数据集主页](https://vis-www.cs.umass.edu/lfw/)
- [torch-fidelity 官方项目](https://github.com/toshas/torch-fidelity)
