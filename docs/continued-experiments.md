# 继续实验：从环境可用到可核验的实际结果

本章连接主教程与本轮新实验。所有命令在项目根目录执行；`python`须替换为对应环境的解释器。代码正文完整收录于[源码汇编](code-compendium.md)，文件用途见[代码地图](code-map.md)。本文的硬件结果来自这台电脑，不能直接套用到另一台电脑或手机。

最终前端已经按用户要求改为原生桌面程序，直接调用本机摄像头。其运行方法、线程设计和实测见[本地桌面教程](desktop-guide.md)。下节保留较早的网页摄像头对照实验，避免把两种实现及不同后台负载的数字混在一起。

## 1. 辅助网页的摄像头对照实验

打开本地应用，允许浏览器使用摄像头，选择“组合特效”，展开“摄像头性能实验”，点击“测量20秒”。保持页面可见并让脸处于画面中。程序对每一帧依次执行：画入Canvas、编码JPEG、发送HTTP请求、检测和合成特效、返回图像、浏览器解码。完成后可保存不含图像的JSON报告；停止按钮释放摄像头。

2026-09-26的Chrome实测使用640×480、设备报告30 FPS输入设置，20.0695秒完成250帧，实际处理吞吐量12.4567 FPS。采集到结果解码耗时中位83.1 ms、P95 93.2 ms，250帧均在页面可见时记录；217帧检出人脸。原始逐帧数值在`reports/webcam-chrome.json`，独立复核和曲线由下列命令生成：

```powershell
python scripts/analyze_webcam.py --report reports/webcam-chrome.json
```

这里有四个容易混淆的量。摄像头设置的30 FPS表示设备提供画面的配置；页面实际处理的12.46 FPS是完成帧数除以测量秒数；单帧延迟描述一帧经历所记录环节的时长；“检测到脸的帧比例”只说明检测器返回非空结果，没有逐帧人工真值，不能叫检测准确率。33帧没有脸也应保留，因为删除它们会改变观测记录。无脸时省去部分特效合成，耗时可能明显较短。

本次计时尚未测量传感器曝光到画面采集之间的等待，也未测量屏幕真正发光的时刻。因此它不是严格的物理“传感器到屏幕”延迟。处理吞吐量包含帧调度，通常不能用“1000除以延迟中位数”替代。后台训练、功耗和温度没有做实验室级控制，记录代表这一次实际运行。

练习：若模型内部耗时10 ms，但每秒只能显示15个结果，能否声称系统达到100 FPS？答案：不能。1000/10只描述该模型调用的理论处理能力，编码、数据传输、页面调度和显示还会消耗时间。应报告完整系统的完成帧数和所测阶段。

## 2. BytePS概念模拟：两个人如何一起更新同一个模型

把一批8个样本分给两个工作进程：甲有3个，乙有5个。参数服务器先把同一份模型参数发给两人；每人根据自己的样本计算平均梯度。梯度可以直观理解为“参数往哪个方向调整，损失会增加多快”。服务器收集梯度后，按样本数加权，再进行一次SGD更新：

```text
g = (3/8) × g_甲 + (5/8) × g_乙
θ_next = θ_current − learning_rate × g
```

不能简单取两人的平均，否则3个样本与5个样本会获得相同总权重，偏离整个8样本批次的平均损失。代码使用Python标准库`multiprocessing`的`spawn`启动两个真实进程，通过`Pipe`传递参数和梯度。每步都与单进程、相同初始参数、同一完整批次的SGD独立计算比较。

```powershell
python -m research.ecosystem_simulation parameter-server
```

实际运行5次更新后，最大参数绝对差为4.47×10⁻⁸，小于设定容差10⁻⁶。报告记录分片大小、损失、参数误差和消息有效载荷估计；`reports/parameter-server-simulation.json`保留全部5步。输入是固定随机种子的8×4教学向量，模型是线性分类器，目的在于验证同步聚合的数学与进程通信，不提供人脸识别准确率。

该等价性有明确条件：两边使用同一初始参数与同一批次，损失按样本求平均，更新规则相同。本例没有BatchNorm、随机Dropout和不同步的模型状态。把同样结论直接套到含局部BatchNorm统计的深层网络可能不成立，需要同步统计或重新设计比较。真实多机环境还涉及网络、故障恢复、异步执行与通信压缩，本例没有复现这些机制，也没有声称两个进程一定比一个更快。

[BytePS官方项目](https://github.com/bytedance/byteps)的公开安装说明依赖CUDA和NCCL。此处按原PDF“模拟”范围交付原理实验，源码和报告均明确`actual_byteps_sdk_used=false`。

练习：甲有2个样本，平均梯度为6；乙有6个样本，平均梯度为2，整个批次平均梯度是多少？答案：`(2/8)×6+(6/8)×2=3`。直接平均得到4，会高估甲的贡献。

## 3. ByteNN概念模拟：让同一张计算图以不同方式执行

神经网络除了可以改变权重，也可以改变执行安排。图优化会尝试消除冗余、合并支持的算子、选择适合后端的算子实现与布局。它不等于重新训练，也不等于全部权重变成INT8。[ONNX Runtime图优化说明](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html)

本实验加载已真实训练的ArcFace pilot所导出的同一个ONNX模型，用同一组8张对齐图像，比较`ORT_DISABLE_ALL`与`ORT_ENABLE_ALL`。两组都限定CPUExecutionProvider、算子内2线程、算子间1线程；每组预热3次、重复15次。源码在`research/ecosystem_simulation.py`的`inference_engine`函数：

```powershell
python -m research.ecosystem_simulation inference-engine --model runs/arcface-pilot/optimized/embedding.onnx --images data/lfw-pilot-train --repeats 15
```

| 批量 | 关闭图优化的中位耗时 | 开启图优化的中位耗时 | 本次中位数比值 |
|---|---:|---:|---:|
| 1 | 34.3693 ms | 25.2712 ms | 1.3600 |
| 8 | 253.4583 ms | 175.5968 ms | 1.4434 |

输出特征最大绝对差分别为5.01×10⁻⁷与5.36×10⁻⁷，通过`rtol=10⁻³、atol=10⁻⁵`数值一致性检查。报告为`reports/inference-engine-simulation.json`。比值大于1表示这次开启图优化更快，但不保证换模型、设备或并发负载后仍有相同收益。两组按顺序运行且存在其他实验进程，时间并非严格随机交错、独占设备的性能研究。

这项结果属于ONNX Runtime的桌面CPU执行，不属于内部ByteNN SDK，也没有验证移动端。它比较的是同一个ONNX模型的两种执行配置，不能与先前PyTorch/INT8实验不同负载下的绝对时间拼接成加速结论。特征数值接近也不等于已经测过完整LFW的任务准确率。

练习：优化后特征最大差只有10⁻⁶，能直接写“识别准确率完全不变”吗？答案：不能。若某个配对原分数恰好在阈值附近，小变化可能改变判定；应在固定评估协议上重新计算准确率、TAR和FAR。

## 4. Intel Arc不是CUDA设备，但可能是可用的训练设备

这台电脑有Intel Arc 130T集成GPU。主环境安装的`torch 2.14.0+cpu`无法使用XPU，说明该软件构建没有相应后端，不能据此断定硬件无法训练。独立`work/xpu-env`安装官方`torch 2.14.0+xpu`及`torchvision 0.29.0+xpu`后，已实际通过张量传入GPU、反向传播、NMS，以及项目StarGAN的WGAN-GP二阶梯度和Adam更新检查。CPU主环境与旧MMDetection容器各自保留。

GPU运算通常异步。只在Python函数调用前后看时间，可能只测到提交任务的时间。计时前后调用`torch.xpu.synchronize()`，等待GPU完成，再计算差值。`scripts/verify_xpu.py`与`reports/xpu-environment.json`记录实际形状、预热次数和每次耗时。

完整生成器的微测使用宽度64、6个残差块、batch2、128×128、FP32，且模型与输入事先放到设备。3次同步前向计时中位数为XPU23.6644 ms、CPU350.6754 ms；它只说明这个形状的一次微测，不代表训练全过程或所有模型都有14.8倍加速。集成GPU报告的约16 GiB可用内存是共享内存视角，不能称为独立16 GB显存显卡。

OpenCL、XPU、CUDA和ONNX Runtime的ExecutionProvider是不同接口。OpenCV的UMat能够在OpenCL上完成小运算，并不能证明OpenCV DNN也在GPU上运行；本机OpenCV 5请求DNN OpenCL target时出现不支持提示，因此没有把那次回退推理记为GPU加速。MMCV原生算子也不能因新PyTorch的XPU成功而自动迁移。

练习：`torch.cuda.is_available()`返回False，是否所有GPU路线都不可用？答案：不是。它只检查CUDA后端；Intel设备需要合适的XPU构建，并对实际模型算子逐项验证。

## 5. 火山引擎：官方样例体验与部署学习

本次在Chrome打开[官方人像融合体验页](https://www.volcengine.com/experience/effect-fusion)，激活内置“方案一”的融合控件，观察两张官方示例输入与融合输出。截图保存为`reports/volcengine-experience.png`，文字证据为同名JSON。没有上传用户照片、创建账户或开通计费服务。结果图可能为预先计算，因此只将其归为官方样例交互体验，未宣称完成实时云API推理。

把学习应用部署为云服务时，须确定输入图像格式、尺寸与数量限制，处理授权和用途约束，设置鉴权和请求签名，管理超时/重试/限流，再记录请求成功率、延迟分位数和成本。密钥由服务端管理，不能写在公开网页JavaScript或Git仓库。一个网页展示出合成图，并不能证明你已获得可部署SDK、API额度或服务等级承诺。

对比本地应用，云端还增加网络传输与服务侧排队；移动端则增加机型、功耗、温度、摄像头和算子支持的差异。浏览器缩成手机宽度仅验证排版，真实手机的速度必须在真实设备上记录。尚未完成的设备或服务验收继续保留在[需求矩阵](requirements-matrix.md)。

练习：为什么网页能展示融合效果，却不能据此把“云API部署”写成完成？答案：展示可能来自静态预计算资源；API部署还要有实际请求、授权配置、响应、错误处理与可复现日志，两种证据证明的事情不同。

## 6. 完整WIDER验证与检测指标

### 6.1 数据完整性先于训练

这次已完整取得 WIDER 官方训练集 12,880 张和验证集 3,226 张，来源为作者页面链接的 CUHK-CSE 镜像，固定版本 `db171f1b7fedf4d3453e81297ff02f9915356d19`。训练 ZIP 为 1,465,602,149 字节，验证 ZIP 为 362,752,168 字节，标注 ZIP 为 3,591,642 字节。整包 SHA256 与每个 ZIP 成员的 CRC 均检查；路径穿越和符号链接也在解压前拒绝。完整哈希保存在 `reports/wider-full-data.json`。

```powershell
python -m research.wider_download
```

文件存在并不代表下载完整。一次被截断的 ZIP 可能仍有文件名与非零大小；完整哈希、ZIP 目录和逐项 CRC 各自检查不同层面的完整性。下载器先把不完整内容保留为 `.incomplete`，完成核验后才作为正式数据使用。Windows 长文件名通过局部扩展路径处理，没有截短原始文件名或修改全局注册表。

转换到 COCO 标注后，训练集有 156,994 个有效框、2,426 个排除框；验证集有 39,112 个有效框、596 个排除框。原始训练 TXT 声明 159,420 个真实框，而相应 MAT 里有 159,424 个条目：差额来自四张零脸图片各有一个 `[0,0,0,0]` 占位条目。四张图片应当保留，但占位条目不能当成人脸。

### 6.2 从一个阈值走到整条曲线

假设一张图有两个真实人脸，算法画出三个框，其中两个正确、一个误检，那么这个简单例子的 Precision 是 `2/3`，Recall 是 `2/2`。如果提高分数阈值，算法通常会少保留一些框，误检和漏检都会发生变化。AP 用整条 Precision–Recall 曲线评价排序质量，不能当作某个阈值下的“正确率”。

WIDER 还按 easy、medium、hard 提供不同难度的目标与忽略标记。评估必须使用原始难度 MAT，不能拿经过 COCO 转换的框数替代。重复框的匹配、坐标的面积约定和被忽略目标的处理都会改变 AP，即使它们看起来只是几行代码。

本项目的 `research/wider_eval.py` 对应官方 MATLAB 协议实现了 Python 计算：IoU≥0.5、包含端点的面积约定、难度 ignore、重复匹配、全局分数归一化、1,000 个阈值扫描与 precision envelope 积分。实际运行的是 Python 版本，没有宣称运行了 MATLAB 或获得官方榜单认证。

```powershell
python -m research.wider_eval --predict-yunet --resume --threads 2
python -m research.wider_reference
```

完整预训练 YuNet 基线使用全部 3,226 张验证图、原始分辨率、单尺度、不翻转；置信度截断 0.3，NMS 0.45，top_k 5,000。31 张零检测图保留为空，没有插入虚假框，也没有删除难例。总计 52,544 个预测框。

| 难度 | 官方目标脸数量 | 本次 AP |
|---|---:|---:|
| easy | 7,211 | 0.8844221 |
| medium | 13,319 | 0.8656839 |
| hard | 31,958 | 0.7504021 |

同一份预测另交给固定源码哈希的 OpenCV Zoo 参考评估函数，三项 AP 的绝对差均为 0。这里的意义是两个实现对同一输入得出一致结果，提高了计算过程的可信度；它不能证明数据标注绝无错误，也不能把预训练 YuNet 的成绩归给本项目训练的 RetinaNet。

原始推理总时间 225.0547 秒；单图推理中位 50.9081 ms、P95 110.8007 ms，CPU 两线程。这些图的尺寸不同于原生摄像头的 640×480，且没有窗口显示和特效，因此不能直接拿它与桌面 FPS 比快慢。

### 6.3 从短训试验到完整迁移训练

早期 32 张训练图的随机初始化 RetinaNet 只用于验证 MMDetection 的训练、保存、读取和评估路径。其 mAP=0 需要保留，不能称“检测器已经学会”。后续完整实验载入 MMDetection 官方 COCO 预训练 RetinaNet R50/FPN，替换最后的 80 类分类层为人脸单类，冻结 ResNet50，训练 FPN 和检测头。

完整训练采用原始 train/val 拆分，最大边 320、batch2，一轮 6,440 次更新。它是一轮迁移训练基线，不是原论文的完整训练计划。若验证 AP 很低，也必须记录，而不是只展示损失下降。训练状态和最终结果分别见 `reports/wider-mmdet-full-progress.json` 与训练完成后的报告。

首次尝试学习率 0.0025、没有 warmup，在第 100 次更新附近出现 NaN。原始框数据检查未发现非有限坐标或非正尺寸，但这不足以证明原因只有学习率。修正尝试采用学习率 0.0005、250 步线性 warmup、梯度范数 10 裁剪、逐步有限性检查，每 100 步保存 checkpoint，并在独立目录保留失败历史。

```bash
docker run --rm --name face-vision-wider-full --cpus 4 \
  -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 \
  -v "$PWD:/project" face-vision-mmdet:3.3 \
  python scripts/mmdet_full.py --threads 4 --epochs 1 --batch-size 2 \
  --size 320 --lr 0.0005 --output runs/wider-mmdet-full-stable
```

上面命令在 WSL 的项目目录执行。中断后可以使用同一配置与实际保存的 `iter_*.pth` 恢复。标准按 epoch 恢复可能重新读取当前轮已见的前缀；本项目的恢复逻辑按已保存批次数，在解码图片前跳过这些批次。已有独立测试证明样本索引不重放，但没有保存全部随机增强状态，不能承诺重启后的随机增强逐位相同。

训练完成并导出全部验证图预测后，再执行官方难度协议和独立交叉检查：

```powershell
python -m research.wider_eval --prediction-dir runs/wider-mmdet-full-stable/wider_predictions --output reports/wider-mmdet-full-official.json
python -m research.wider_reference --prediction-dir runs/wider-mmdet-full-stable/wider_predictions --project-report reports/wider-mmdet-full-official.json --output reports/wider-mmdet-full-crosscheck.json
```

练习：预训练 YuNet 的 hard AP 为 0.7504，而一轮 RetinaNet 的 COCO AP 是另一个数，能直接宣布哪个模型更好吗？答案：不能。需要相同数据、框匹配协议、输入尺寸和预测设置；WIDER hard AP 与 COCO 的多 IoU AP 也不是同一个指标。应在一致协议下比较，并保留两种模型各自的来源。

## 7. 把四类运行环境真正弄明白：Anaconda 安装、启动与排错

本节记录 2026-09-26 在这台电脑上完成的安装。Anaconda 已经实际安装，项目 conda 环境也通过了计算、检测、Notebook 和图形组件验证。请先使用现成环境学习，不必再次安装。完整机器记录见 [Anaconda 验证报告](../reports/anaconda-environment.json)，精确 Python 包版本见 [conda 项目环境清单](../reports/anaconda-environment-freeze.txt)。

### 7.1 先认识“解释器、包、环境”

`desktop.py` 是源代码；`python.exe` 是读懂并执行源代码的解释器；PyTorch、OpenCV、Pillow 是代码调用的包。一个运行环境把解释器、包和相应的本地库组织在一起。相同的源代码换一个解释器执行，可能得到不同版本的包，甚至提示找不到包。

`venv` 是 Python 自带的隔离工具，主要隔离项目使用的 Python 包。`conda` 可以管理独立解释器和包，也可以管理许多非 Python 的二进制依赖。Anaconda Distribution 是包含 conda、Python 和一批常用软件的发行版。因此，“已经创建 `.venv`”能证明有一个隔离环境，不能证明安装了 Anaconda。最初验收把 Anaconda 标为未完成是正确的；实际安装并验证后，才能更新该项。

本项目有四类运行环境。它们共享源代码和按需读取的模型文件，安装的包相互隔离。Anaconda 自带的 `base` 另外承担管理工具所在基础环境的角色；不要把它与项目环境混称为同一个 Python。

| 环境 | 实际关键版本 | 用途和分开的理由 |
|---|---|---|
| 主项目 `.venv` | Python 3.12.14；torch 2.14.0+cpu；torchvision 0.29.0+cpu；OpenCV 5.0.0；Pillow 12.3.0；Tk 8.6.12 | 原生桌面程序的主要运行环境，已完成本机摄像头和界面实测。它在 Anaconda 安装前就已建立，后续保留，避免改变已通过验收的运行条件。 |
| Anaconda 管理的项目 conda 环境 | Python 3.12.14；torch 2.14.0+cpu；torchvision 0.29.0+cpu；OpenCV 5.0.0；NumPy 2.5.3；Pillow 12.3.0；JupyterLab 4.6.4；Tk 8.6.15；torch-fidelity 0.4.0 | 满足真实 Anaconda 环境要求，也用于学习 conda、Notebook 与环境复现。基础发行版是 Anaconda 2026.07-1，conda 26.5.3，`base` Python 3.14.6；项目另选 3.12.14，不需要把 `base` 降级。 |
| 独立 `work/xpu-env` venv | Python 3.12.14；torch 2.14.0+xpu；torchvision 0.29.0+xpu | Intel Arc 130T 上的 PyTorch 训练与兼容性实验。CPU 与 XPU 构建使用同一个包名 `torch`，在同一环境反复安装会替换彼此，因此单独建环境。 |
| WSL 内的 MMDetection Docker 环境 | Python 3.10.21；torch 2.1.0+cpu；torchvision 0.16.0+cpu；NumPy 1.26.4；MMCV 2.1.0；MMEngine 0.10.7；MMDetection 3.3.0；OpenCV 4.11.0 | 保留 OpenMMLab 已验证的旧版本组合及 Linux 二进制算子。镜像名为 `face-vision-mmdet:3.3`。不能把新环境的 torch 升级直接套在旧 MMCV 编译算子上。 |

OpenCV 的 Python 导入名是 `cv2`；此处 `cv2.__version__` 为 `5.0.0`，其安装分发包 `opencv-python` 的版本为 `5.0.0.93`。这两个显示不同不代表装了两份相互冲突的 OpenCV。类似地，`+cpu` 与 `+xpu` 是 PyTorch 构建标记，选包时不能省略其含义。

### 7.2 实际安装在哪里：区分物理目录与目录联接

下面是这台电脑的真实路径，不能直接假定另一台电脑也有同名目录。

| 对象 | 实际位置 |
|---|---|
| 项目根目录 | `C:\Users\0\Documents\Codex\2026-09-26\plugin-computer-use-openai-bundled-github\outputs\face-vision-lab` |
| 主项目解释器 | 项目根目录下的 `.venv\Scripts\python.exe` |
| Anaconda **物理安装目录** | `C:\Users\0\Documents\Codex\work\fv-ana-260926` |
| 项目 conda 环境的**物理目录** | `C:\Users\0\Documents\Codex\work\fv-env-260926`，解释器是此目录下的 `python.exe` |
| Anaconda 目录联接 | 本线程工作区的 `work\anaconda` → 上面的 `fv-ana-260926` |
| conda 项目环境目录联接 | 本线程工作区的 `work\conda-face-vision` → 上面的 `fv-env-260926` |
| XPU 解释器 | 本线程工作区的 `work\xpu-env\Scripts\python.exe` |

目录联接（Junction）可以理解为文件系统中的“另一条入口”，它没有复制一份安装。最终方案是**长入口指向真实短目录**，因为安装工具可能解析入口背后的真实路径。反过来创建一个短入口，让它指向仍然很长的真实目录，本次已经试过，解包仍然失败。不要移动、复制后直接运行一个已安装环境来代替重新创建；环境中可能记录原来的绝对路径。

### 7.3 安装做了什么，哪些证据说明已经成功

安装器取自 [Anaconda 官方归档](https://repo.anaconda.com/archive/)，文件名为 `Anaconda3-2026.07-1-Windows-x86_64.exe`。实测 SHA256 与官方值一致：

```text
b545f4bd8ab3bf32d99002a0779a887668ebfe479ee32ecbf060375670d5ee09
```

数字签名验证状态为 `Valid`，签名组织为 Anaconda, Inc.。SHA256 用于检查下载内容是否与公布的文件相同；签名用于验证软件发布者，两者解决不同问题。本次在用户明确接受安装器与官方 `main` 仓库条款后执行安装，范围为个人学习，没有注册付费计划。其他使用场景应重新阅读 [官方条款](https://www.anaconda.com/legal/terms/terms-of-service)，不能把本次授权或适用范围自动转移给另一组织。

最终安装参数包括 `JustMe`、`RegisterPython=0`、`AddToPath=0`、`NoRegistry=1`、`NoShortcuts=1` 和静默模式 `/S`。`NoRegistry` 与 `NoShortcuts` 是通过这个安装器的 `/S /?` 实际确认支持后加入的。没有运行 `conda init`，没有修改持久 PATH、默认 Python、长路径注册表策略或 GPU 驱动。创建项目环境时仅使用官方 `https://repo.anaconda.com/pkgs/main` channel。

[安装脚本](../scripts/setup_anaconda.ps1)默认只下载和校验；只有显式传入 `-InstallAndConfigure -AcceptAnacondaTerms` 才进入接受条款与安装分支。这是复现入口的行为说明，本机已经完成安装，日常学习使用下一节的启动命令即可。

验收不只看“能导入包”。[验证脚本](../scripts/verify_anaconda.py)实际检查了以下内容：

1. 环境中存在 `conda-meta/history`，当前 `sys.prefix` 确实对应目标环境。
2. PyTorch 对输入 `x=2` 的 `x²` 做反向传播，得到导数 `2x=4`。
3. YuNet 读取项目的 NASA 公开示例图，检出一张脸。
4. Jupyter 启动明确指定的 conda Python 内核，在真实 cell 中计算 `torch.arange(4).sum()`，输出 `6`，同时输出正确环境路径；检查后关闭临时内核。
5. Tk 创建隐藏窗口、更新并销毁，Pillow 的 ImageTk 图片可绑定到标签控件。
6. 补齐 `torch-fidelity==0.4.0` 后实际导入，运行 `pip check` 通过；恢复安装涉及的 `jsonpointer` 包清单及文件也完整。

验证边界同样要准确：conda 验证没有打开摄像头；摄像头、六种效果和窗口关闭释放的实测来自主 `.venv`，见[本地桌面教程](desktop-guide.md)。导入 torch-fidelity 也不等于计算过 FID/IS，质量评估应查对应的真实实验报告。

### 7.4 日常启动：先固定目录，再指定解释器

以下命令适用于这台电脑的 PowerShell。`&` 是调用运算符，用来执行引号或变量中保存的程序路径。`$env:` 设置只影响当前进程及随后启动的子进程；此处不修改系统持久设置。

```powershell
$Project = 'C:\Users\0\Documents\Codex\2026-09-26\plugin-computer-use-openai-bundled-github\outputs\face-vision-lab'
Set-Location -LiteralPath $Project
$env:SystemRoot = 'C:\Windows'
$env:WINDIR = 'C:\Windows'
$env:COMSPEC = 'C:\Windows\System32\cmd.exe'
$env:PYTHONIOENCODING = 'utf-8'
```

主环境日常启动命令如下。该脚本检查并安装主环境依赖、准备模型，然后默认打开原生桌面窗口；窗口中的按钮控制摄像头。

```powershell
.\start.ps1
```

如果依赖和模型已准备好，也可直接启动并先核对解释器：

```powershell
& '.\.venv\Scripts\python.exe' -c "import sys, torch; print(sys.executable); print(torch.__version__)"
& '.\.venv\Scripts\python.exe' desktop.py
```

使用 conda 项目环境时，推荐 `conda run -p`。它按指定目录为子进程准备 Windows DLL 查找环境，不要求先修改默认 Python，也不需要永久改 PATH。

```powershell
$Conda = 'C:\Users\0\Documents\Codex\work\fv-ana-260926\Scripts\conda.exe'
$EnvPrefix = 'C:\Users\0\Documents\Codex\work\fv-env-260926'
& $Conda run --no-capture-output -p $EnvPrefix python -c "import sys, torch; print(sys.executable); print(torch.__version__)"
& $Conda run --no-capture-output -p $EnvPrefix python desktop.py
```

这里应打印 `fv-env-260926\python.exe` 和 `2.14.0+cpu`。上面的桌面启动命令供学习者使用；conda 的已记录验收范围是上一节列出的功能，并未补做一轮摄像头性能比较。打开 Notebook 时仍使用同一前缀，避免“Jupyter 来自环境 A，代码内核却来自环境 B”：

```powershell
& $Conda run --no-capture-output -p $EnvPrefix python -m jupyterlab notebooks/01-first-vision-lab.ipynb
```

Notebook 单元格中运行 `import sys; print(sys.executable)`，确认内核解释器。需要重新进行 conda 功能检查时，可以执行下列命令；它会更新本机验证报告并短暂创建隐藏 Tk 窗口与本地 Notebook 内核，不会打开摄像头。

```powershell
& $Conda run --no-capture-output -p $EnvPrefix python -m pip check
& $Conda run --no-capture-output -p $EnvPrefix python scripts/verify_anaconda.py --environment $EnvPrefix --report reports/anaconda-environment.json
```

XPU 环境的身份检查使用它自己的解释器：

```powershell
$XpuPython = 'C:\Users\0\Documents\Codex\2026-09-26\plugin-computer-use-openai-bundled-github\work\xpu-env\Scripts\python.exe'
& $XpuPython -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.xpu.is_available())"
```

本机已经验证 `2.14.0+xpu` 与 `True`。完整的算子、反向传播和同步计时入口是 [verify_xpu.py](../scripts/verify_xpu.py)，适合没有其他训练抢占设备时运行；兼容性证据见 [XPU 报告](../reports/xpu-environment.json)。第 4 节已解释计时边界，不能把这些小实验推广为所有模型的加速倍数。MMDetection 按第 6 节在 WSL 的专用 Docker 镜像里运行，不要往主 `.venv` 中补装那套旧依赖。

遇到包缺失时，先看 `sys.executable`，再对**同一个解释器**运行 `-m pip`。裸写 `pip install ...` 可能命中 PATH 中另一个环境；这比“反复安装仍无法导入”更值得先检查。项目已锁定依赖，练习时也不要随意把 torch、torchvision、MMCV 全部升级到最新。

### 7.5 实际遇到的路径故障与修复依据

Windows 的传统路径限制涉及“环境目录 + 包内子目录 + 文件名”的完整路径。安装前缀看起来不长，不代表安装后的每个文件都足够短；不同程序对扩展路径、符号别名的支持也不同。本机的 `LongPathsEnabled` 原值为 `0`，本次没有修改它。

| 真实现象 | 判断与采取的修复 | 为什么不能省略这一点 |
|---|---|---|
| 子进程缺少 `SystemRoot`、`WINDIR`、`COMSPEC`，伴随 SSL 随机源异常、DNS/CIM 报错和 `venv ensurepip` 失败 | 在任务子进程补回正确 Windows 目录变量后，SSL 随机源、SSLContext、DNS、curl 与环境创建恢复 | 不把这组错误误判成包版本冲突；没有关闭 TLS 校验，也没有改代理、防火墙。 |
| 主 `.venv` 安装 JupyterLab 时，深层前端静态资源超过传统长路径限制，pip 报文件找不到 | 当时使用同一解释器的 `\\?\C:\...\python.exe` 扩展路径完成相关安装，随后导入和依赖检查通过 | `\\?\` 是 Windows 扩展路径前缀，是有条件的兼容方法，不是任意路径故障的通用修复。 |
| Anaconda 安装器拒绝 93 字符的原目标目录，显示最大允许 54，退出码 2 | 为这份安装器规划真实短目录 | **54 是这份安装器在当前条件下报告的前缀限制**，不能写成 Windows 对所有路径统一限制 54 字符。 |
| NTFS 8.3 短名称含 `~`，被安装器判为无效字符 | 放弃该失败方案，使用正常命名的真实短目录 | 不应把失败的尝试写成学生可以照抄的成功步骤。 |
| 短 Junction 指向物理长目录，安装器能开始，但提取 `anaconda-client` 失败，外层出现 `InvalidArchiveError`、`BrokenProcessPool` | 提取器解析到真实长路径；最终把文件实际放在 `fv-ana-260926`，工作区入口反向指向它 | 只缩短显示入口不够。外层进程池异常不能直接等同于 CPU 或安装包损坏。失败日志与目录保留。 |
| 先前“删除旧联接并移动失败目录”的组合命令被自动审批拒绝，未执行 | 改为单独核验对象，在原父目录内只重命名本轮已知失败目录，保留为 `work/anaconda-failed-long-path` | 不扩大清理范围；没有递归删除不明目录，也没有把审批失败写成已执行。 |
| conda 环境已经改成真实短目录后，沿用扩展解释器路径安装 `jsonpointer`，其 `../../Scripts/jsonpointer` 触发 `[Errno 22] Invalid argument` | 改用普通的真实短 Python 路径继续 pip 安装；检查 `jsonpointer 3.1.1` 的 RECORD 和全部列举文件均完整 | 扩展路径下的 `..` 处理与普通路径不同。前面修好 Jupyter 的方法在这里不能机械照搬；实际环境不同，修复也要不同。 |
| 补全依赖期间研究清单新增 `torch-fidelity==0.4.0` | 前一次 pip 结束后串行补装，重新导入并检查依赖，更新冻结清单 | 不在同一环境并发运行多个 pip，以免包文件与元数据互相覆盖。 |

最后保留三个习惯：先记录完整错误和解释器路径；一次只改一个可解释的条件；修复后运行实际功能，再更新验收结论。“安装命令返回成功”与“模型、内核和窗口都工作”是不同层次的证据。

### 7.6 练习与参考答案

**练习 1：** 主 `.venv` 中可以 `import torch`，是否足以勾选“Anaconda 已安装”？

**答案：** 不足以。需要验证 Anaconda 发行版或 conda 工具真实存在，并验证项目 conda 环境的解释器、`conda-meta/history` 及功能。本项目在补装之前只完成了 venv，在补装并通过报告中的检查之后才满足 Anaconda 项。

**练习 2：** 安装一个包后，桌面程序仍提示 `ModuleNotFoundError`。先重新装所有依赖，还是先看解释器？请写出主环境的检查命令。

**答案：** 先执行 `& '.\.venv\Scripts\python.exe' -c "import sys; print(sys.executable)"`，确认启动与安装是否使用同一环境。安装时也使用该解释器的 `-m pip`，不要只凭终端前面的环境名称猜测。

**练习 3：** `base` 显示 Python 3.14.6，项目显示 3.12.14，是不是版本损坏？

**答案：** 不是。它们是两个不同环境；`base` 提供管理工具，项目选择经过验证的 3.12.14 运行依赖。检查各自路径和功能即可，不需要为了让数字相同而升级项目或降级基础环境。

**练习 4：** 给一个很长的物理安装目录创建短 Junction，能否保证安装器不再遇到长路径？

**答案：** 不能。某些工具解析联接后的物理路径。本次成功的是把文件放进真实短目录，再建立方便访问的长入口。不能通过缩短界面显示的路径长度推断实际文件操作一定成功。

**练习 5：** 主环境 `torch.cuda.is_available()` 为 False，是否说明 Intel Arc 无法参与计算？直接给主环境装 XPU wheel 是否合适？

**答案：** 第一个判断不成立：CUDA 是另一套计算后端，当前主环境还是 CPU 构建。独立 XPU 环境已经验证 `torch.xpu.is_available()` 为 True，以及实际张量、反向传播和部分模型算子可用。第二个做法会替换主环境的 torch 构建，改变已验收条件，因此本项目使用独立环境。

**练习 6：** conda 的 Tk、ImageTk 与 Notebook 都通过，能否声称该环境摄像头已达到某个 FPS？

**答案：** 不能。这些测试没有打开摄像头。摄像头吞吐量必须在对应环境、分辨率、效果、设备和负载条件下单独测量。引用主 `.venv` 的结果时应保留其环境与测量条件，不能移植为 conda 的实测数字。

## 8. 用真实 CelebA 数据完成 StarGAN 微调、评估与桌面部署

这一节已经完成真实数据训练和指标计算。我们使用作者公开的 StarGAN 预训练模型作为起点，在本机 Intel Arc 130T 上微调，最后把生成器接入原生桌面程序。最终模型实际继承本次 **1200 次判别器更新、240 次生成器更新**。这不是从零复现论文的完整训练，也不表示把全部 CelebA 图片训练了一遍。

### 8.1 先理解模型究竟在做什么

设一张人脸图片为 $x$，目标属性向量为 $c$。生成器 $G(x,c)$ 接收图片与指定的目标标签，输出一张编辑后的图片。判别器有两个任务：给图片一个真实度分数，预测图片的属性。训练时，生成器试图让编辑后的图片既符合目标标签，又能被判别器看作真实图片；判别器则学习区分真实图片和生成图片。

本项目的属性顺序固定为：

| 向量位置 | 数据集标签 | 例子 `[0,1,0,0,1]` 的含义 |
|---|---|---|
| 0 | Black_Hair | 不指定黑发 |
| 1 | Blond_Hair | 指定金发 |
| 2 | Brown_Hair | 不指定棕发 |
| 3 | Male | 目标标签为 0 |
| 4 | Young | 目标标签为 1 |

这些是生成条件，不是判断现实人物真实年龄或性别的结论。桌面程序让使用者选择合成方向，不把生成结果当作身份认证。发色选项采用互斥设置；模型也不保证一定达到目标或保持身份。

生成器采用宽度 64、六个残差块的网络；判别器采用六层下采样结构。输入和输出图片均为 128×128。残差块的基本思想是学习对已有特征的修改量，而不是每一层都重新构造全部信息。

### 8.2 数据从哪里来，哪些验证真正完成了

首先尝试的是 [CelebA 数据作者主页](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)指向的原始 Google Drive 下载。普通大文件确认后仍出现公开下载配额限制，没有规避配额、登录或登记要求。

随后使用 [StarGAN 官方项目 `download.sh`](https://github.com/yunjey/stargan/blob/master/download.sh)明确提供的配套下载链接。准确名称是 **“StarGAN 官方项目配套下载源”**，不能称作“CelebA 数据作者官方镜像”。原始数据仍遵守 CelebA 非商业研究条件；原图、生成图、模型权重保留本机，不随源代码再分发。

实际取得的完整图片 ZIP 为 1,417,753,833 字节，观察到的 SHA256 为 `8ef7e94f7e82121a137491406bcfb1a22be832bcffc5d0fa3f232a0fe36410b0`。配套来源没有公布完整 ZIP 的官方哈希，所以不能把这个观察值说成“已通过原 CelebA 图片包官方 MD5”。哈希只是文件指纹；是否来自正确来源，还要结合下载地址和来源说明判断。

压缩包内部原始属性表的 MD5 为 `75e246fa4810816ffd6ee81facbd244c`，与 torchvision 发布的 CelebA 原始属性文件 MD5 一致。完整 ZIP 中的 **202599 张图片文件名，与属性表文件名集合完全一致**。40 个属性的阳性计数保存在 `reports/celeba-expanded-data.json`。选入本次实验的每张图另外记录 SHA256，并由 ZIP 成员 CRC 检查读取完整性。

直接统计完整属性表，本次使用的五个标签分布如下。“阳性”在这里仅指标注值为 1，不是医学概念。

| 标签 | 标注为 1 的图片数 | 占全部 202599 张的比例 |
|---|---:|---:|
| Black_Hair | 48472 | 23.925% |
| Blond_Hair | 29983 | 14.799% |
| Brown_Hair | 41572 | 20.519% |
| Male | 84434 | 41.675% |
| Young | 156734 | 77.362% |

比例的计算就是“该标签为 1 的图片数 ÷ 全部图片数”。这些标签并不均衡，不能假定模型在每个属性上都学得同样充分。它们是多标签标注，三个发色也不要求覆盖全部图片；这些数字不代表真实世界人口分布。

完整解压曾因任务中断停止。解决方法不是假定“目录里有很多图就等于完整数据”，而是保留完整 ZIP，按实验清单直接提取所需图片。这样既验证来源和完整性，也避免反复解压二十万个小文件。

### 8.3 训练池、抽样次数、去重数量不是一回事

本次按照作者 `data_loader.py` 的代码复原划分：从原始属性表顺序开始，使用 Python 随机种子 1234 打乱，前 1999 张作为作者代码中的测试部分，其余作为训练部分。它不同于 CelebA 另行提供的官方分区表。

从训练部分取前 4096 张作为扩大后的训练池；从测试部分取前 768 张作为生成源，再取随后 768 张作为真实参考。训练图、生成源图和真实参考图的文件名三者无交集。**图片不重叠，不等于人物身份不重叠**：同一个人可能有多张不同照片。作者当前代码也不能独立证明公开检查点历史上确实采用了相同划分，所以报告保留预训练数据重叠的审计限制。

一个 batch 是一次送入网络的小批图片；本次 batch size 为 2。一个 step 是一次判别器参数更新。一个 epoch 才表示遍历一轮训练池。

| 最终模型继承的阶段 | 配置训练池 | 保留的 D 更新 | 保留的 G 更新 | 图片抽样次数 |
|---|---:|---:|---:|---:|
| 初始先导，seed 42 | 512 | 200 | 40 | 400 |
| 扩展阶段的完整检查点，seed 42 | 4096 | 500 | 100 | 1000 |
| 从检查点恢复，seed 43 | 4096 | 500 | 100 | 1000 |
| 合计 | 不能相加当作人数 | 1200 | 240 | 2400 |

为什么合计不是 2400 张去重图片？不同阶段可能再次抽到同一张图。当前没有逐次访问清单，因此总去重数记为未知，不能编造。为什么也不能说训练了 4096 张？训练池只是可供抽样的范围，每个扩展阶段只取了其中 1000 次，没有走完一个 epoch。图片数量更不能当作不同人物的数量。

### 8.4 为什么使用预训练，怎样理解三段训练

随机初始化意味着模型最开始几乎不会画脸。作者预训练检查点已经形成图像表征与编辑能力，微调是在这些权重上做小幅调整。本次机器适合完成真实的小规模实验，但没有执行论文量级的从零长训。因此下载并严格载入作者提供的 `200000-G.ckpt` 和 `200000-D.ckpt`，学习率设为 $10^{-5}$。文件名中的原始训练阶段来自作者发布信息，不能计入本机本次新增训练量。

训练目标包含四种约束：

1. **真实度差异**：生成图片的分数要接近真实图片。
2. **属性分类**：生成图片要符合目标标签。
3. **循环重建**：先编辑成目标属性，再改回原属性，应接近原图。
4. **梯度惩罚**：约束判别器对输入微小变化的敏感程度，使训练更稳定。

判别器的目标为 $E[D(G(x,c))]-E[D(x)]+L_{cls,real}+10L_{GP}$。生成器的目标为 $-E[D(G(x,c))]+L_{cls,fake}+10L_{rec}$，其中 $L_{rec}$ 是还原图与输入图的平均绝对误差。判别器输出实数分数，不是限制在 0 到 1 的概率，所以 D/G loss 可以为负。GAN 中两个网络的目标同时变化，一条损失下降不能单独证明图片变好。

代码审查发现初始先导阶段与作者损失归一化有两处差异：五属性 BCE 默认平均除以 $N\times5$，作者实现只除以 batch 大小 $N$；另外，判别器输出 2×2 patch 分数，梯度惩罚应先对四个分数求和再求输入梯度，提前平均会改变梯度尺度。这两处已在扩展阶段修正，分类、重建、梯度惩罚的权重分别为 1、10、10。原先导结果保留，不能改写成“从第一步起就完全相同”。命令中的 `--loss-protocol legacy` 专门用于复现历史先导，默认 `official` 使用修正后的公式。

扩展阶段原计划运行 1000 步，任务中断时日志至少到 650 步，但最后一个完整检查点只保存到 500 步。因此后续至少 150 次已经记录的更新丢失，不能计入最终模型；没有写日志的尾部更新数量未知。恢复时同时载入 G、D 以及两个 Adam 优化器的状态，并以 seed 43 开始新的数据顺序，再训练 500 步。这是恢复参数和优化器，不是精确恢复中断瞬间的数据迭代器。最终历史 CSV 保留丢弃行并加标记，曲线只画最终模型真正继承的更新。

恢复阶段 500 步实际耗时 372.452 秒，PyTorch 报告的 XPU 峰值已分配内存为 1,143,694,848 字节。耗时包含数据处理、平均每五步一次的 G 更新和检查点保存，不能当作单独 D 前向延时；峰值也不是整台电脑或驱动的全部显存使用量。

### 8.5 真实遇到的 InstanceNorm 推理错误

训练模型和使用模型是两种运行状态。PyTorch 的 `.train()`、`.eval()` 对某些归一化层有不同含义；它们不是“要不要计算梯度”的同义词。

作者测试代码没有把 G 切到普通 eval 状态，InstanceNorm 使用每张图自身的统计量。我们最初直接 `.eval()` 后采用旧检查点的 running statistics，真实图出现明显色偏与棋盘纹理。加载成功没有报错，仍可能存在这种语义兼容问题。

修复时先严格载入全部模型权重，再关闭 InstanceNorm 的运行均值/方差缓冲区，使模型在 eval 模式下仍使用每张图自身的统计量。回归测试确认这与作者测试所用的逐图归一化数值一致。此前错误网格保存在本地审计目录，未用作最终“微调前”基线。**前后两个模型都应用同一修复**，不能把修复推理错误获得的变化说成微调提升。

### 8.6 FID 和 IS 计算了什么，结果能说明多少

本次使用 `torch-fidelity 0.4.0` 的 InceptionV3 兼容实现及其公开权重，没有自造替代指标。FID 使用 2048 维特征；IS 使用分类 logits，划分为 10 组，随机种子 2020，batch size 为 16，PyTorch CPU 线程数为 2。初次前模型用 CPU 生成、后模型用 XPU 生成，且 torch 构建不同；虽然差异可能很小，严谨比较仍应控制这个变量。因此又将前模型改为与后模型相同的 XPU 生成、`torch 2.14.0+xpu` 构建、CPU 指标设备，重新完成对照。旧结果保存在 `reports/stargan-before-cpu.json`，不删除，也不再作为主比较。

两次评估使用相同的 768 张生成源图和 768 张独立真实参考。第 i 张生成图的目标向量等于第 i 张参考图的五个属性，保持目标属性分布一致。读入前重新检查图像 SHA256 和三组图片无交集。处理为 RGB、中心裁剪 178×178、双线性缩放到 128×128、像素归一化到 [-1,1]。生成图与真实参考都保存为相同处理后的 PNG 再计算指标。

FID 用一组特征的均值和协方差描述其分布，比较生成集合与真实集合的差异，通常越低越接近参考分布。IS 同时鼓励单张图片有明确类别、整体图片覆盖多样类别。但其分类器来自 ImageNet，人脸编辑的细致质量不一定能被这些类别描述。

| 相同协议实测 | 作者预训练模型 | 本次微调后模型 |
|---|---:|---:|
| FID | 41.272788 | 40.980400 |
| IS 均值 | 3.088072 | 3.098037 |
| IS 十个划分的标准差 | 0.253799 | 0.170242 |

FID 减少约 0.292388，IS 均值增加约 0.009965。变化很小，而且只有 768 个样本，协方差估计不稳定；这不是统计显著性证明，也不能直接与论文大样本 FID 比较。IS 的划分标准差不是两个模型性能差的置信区间。真实输出仍可见生成伪影，身份保持也没有单独量化验证。可靠结论是：训练、生成、标准指标计算都真实执行；在此固定协议上观察到上述微小变化。旧 CPU 前模型对照为 FID 41.272913、IS 3.088127，只保留为历史，不用于上表主比较。

### 8.7 所有入口与本机复现步骤

以下命令的工作目录都是项目根目录。`python` 应明确指向所需环境：本机数据准备、汇总和 ONNX 导出使用 `.venv/Scripts/python.exe`；XPU 训练以及前后两次受控评估使用 `../../work/xpu-env/Scripts/python.exe`，但这两次评估的 Inception 指标计算设备仍为 CPU。独立 XPU 环境已经验证真实张量传输、反向传播与 WGAN-GP 二阶自动微分。不要把 CPU/XPU 轮子混装到同一个环境。

PowerShell 工具环境曾缺少 Windows 运行时变量。执行命令前补回本机真实系统路径：

```powershell
$env:SystemRoot='C:\Windows'
$env:WINDIR='C:\Windows'
$env:COMSPEC='C:\Windows\System32\cmd.exe'
$env:OMP_NUM_THREADS='2'
$env:OPENBLAS_NUM_THREADS='2'
```

**数据与环境入口：**

```powershell
# 原始入口失败也会明确写报告，并以非零状态结束；不绕过配额。
python -m research.fetch_official_data celeba --report reports/celeba-download.json
python -m research.fetch_official_data stargan-official --file celeba.zip --output data/stargan-official --report reports/celeba-stargan-author-download.json
python -m research.fetch_official_data stargan-official --file celeba-128x128-5attrs.zip --output data/stargan-official --extract --report reports/stargan-pretrained-download.json
python -m research.stargan_pilot prepare --train 512 --evaluation 128 --output data/celeba-pilot --report reports/celeba-pilot-data.json
python -m research.stargan_pilot prepare --train 4096 --evaluation 768 --output data/celeba-expanded --report reports/celeba-expanded-data.json
../../work/xpu-env/Scripts/python.exe scripts/verify_xpu.py
```

**复现最终检查点所保留的三个训练阶段：**

```powershell
python -m research.stargan train --root data/stargan-official/celeba/images --labels data/celeba-pilot/attributes.txt --partition data/celeba-pilot/partition.txt --output runs/stargan-pilot --device xpu --threads 2 --size 128 --width 64 --blocks 6 --batch-size 2 --steps 200 --n-critic 5 --lr .00001 --log-interval 10 --save-interval 100 --init-generator data/stargan-official/200000-G.ckpt --init-discriminator data/stargan-official/200000-D.ckpt --loss-protocol legacy
python -m research.stargan train --root data/stargan-official/celeba/images --labels data/celeba-expanded/attributes.txt --partition data/celeba-expanded/partition.txt --output runs/stargan-expanded --device xpu --threads 2 --size 128 --width 64 --blocks 6 --batch-size 2 --steps 500 --n-critic 5 --lr .00001 --log-interval 50 --save-interval 250 --continue-checkpoint runs/stargan-pilot/last.pt --loss-protocol official
python -m research.stargan train --root data/stargan-official/celeba/images --labels data/celeba-expanded/attributes.txt --partition data/celeba-expanded/partition.txt --output runs/stargan-resumed --device xpu --threads 2 --size 128 --width 64 --blocks 6 --batch-size 2 --steps 500 --n-critic 5 --lr .00001 --seed 43 --log-interval 50 --save-interval 100 --continue-checkpoint runs/stargan-expanded/last.pt
```

第二条复现命令直接运行被保留的 500 步，不必故意再制造一次中断。原始计划参数仍在原始 config 中，CSV 的丢弃标记不能在复现时伪造。固定随机种子有助于复现抽样和初始化条件，但没有承诺不同硬件、驱动或算子实现下逐字节相同。本机长任务后来使用隐藏后台进程、stdout/stderr 日志与更频繁检查点保存，聊天中断不再需要取消训练。

**单图生成、前后评估、独立指标与汇总：**

```powershell
python -m research.stargan generate --checkpoint runs/stargan-resumed/last.pt --image data/stargan-official/celeba/images/028136.jpg --targets 0,1,0,0,1 --output runs/stargan-example.png
../../work/xpu-env/Scripts/python.exe -m research.stargan_pilot evaluate --checkpoint data/stargan-official/200000-G.ckpt --manifest data/celeba-expanded/manifest.json --attributes data/celeba-expanded/attributes.txt --output runs/stargan-before-xpu --report reports/stargan-before.json --device xpu --threads 2
../../work/xpu-env/Scripts/python.exe -m research.stargan_pilot evaluate --checkpoint runs/stargan-resumed/last.pt --manifest data/celeba-expanded/manifest.json --attributes data/celeba-expanded/attributes.txt --output runs/stargan-after --report reports/stargan-after.json --device xpu --threads 2
python -m research.stargan metrics --real runs/stargan-after/real-reference --generated runs/stargan-after/generated --output reports/stargan-metrics-recheck.json --threads 2
python -m research.stargan_pilot summarize
```

单图命令使用已经在本机实验中读取过的 CelebA 图片 `028136.jpg`。`research.stargan generate` 与本次 CelebA 评估采用相同的预处理：把 RGB 图像中心裁成 178×178，再用双线性插值缩放到 128×128，最后把像素归一化到 [-1, 1]。因此，它适合人脸已居中、按 CelebA 方式对齐的输入；把人脸偏在一侧的整张照片直接传给它，中心裁剪可能截掉人脸。

桌面程序接受普通照片或摄像头画面，使用另一种输入处理：先要求 YuNet 检测到恰好一张脸，再围绕检测框截取方形区域，边长约为框的较长边的 1.8 倍，水平中心位于框中心，竖直中心位于框顶向下 0.45 倍框高处；超出图像边界的部分用反射填充，随后缩放为 128×128 并做相同归一化。这是检测框引导的裁剪，并不是 CelebA 的几何对齐。两者复用同一个生成器，但姿态、脸部位置和背景分布可能不同，所以本章的 CelebA FID/IS 不能直接解释为任意摄像头输入的质量指标。界面展示的是输入裁剪与生成裁剪，不会自动把编辑后的人脸贴回整张原图。

`summarize` 是针对本次已记录的 200+500+500 实验的收集器，先检查阶段和检查点一致性，再输出 `reports/stargan-experiment/report.json`、原始历史 CSV、真实训练曲线 SVG 和三个阶段配置。另一个训练计划应建立自己的实验清单，不能套用这份固定历史称作自己的训练结果。

**导出、验证和启动原生桌面程序：**

```powershell
python -m research.stargan_deploy --checkpoint runs/stargan-resumed/last.pt --image data/stargan-official/celeba/images/028136.jpg --threads 2 --repeats 5 --output runs/stargan-deploy/generator.onnx --report reports/stargan-deployment.json
python -m unittest discover -s tests -p test_research.py -v
python desktop.py
```

ONNX 的输入名固定为 `images` 和 `attributes`，形状分别为 `[1,3,128,128]`、`[1,5]`，类型为 FP32；输出名为 `edited`。图片是 RGB 的 NCHW 排列：1 张图片、3 个颜色通道、128 行、128 列。输出裁剪到 [-1,1] 后，以 `(output+1)*127.5` 变回图片像素。原生程序直接加载此本地模型，可从图片或当前相机画面进行一次属性编辑；这不等于实时视频 GAN 编辑。

实际模型大小为 33736691 字节。真实图片上 PyTorch 与 ONNX Runtime 的最大绝对误差为 $1.96695\times10^{-6}$，ONNX checker 和数值一致性断言通过。CPU 两线程、五次运行中位数为 281.333 ms；测量时有其他实验同时运行，不能视为独占硬件基准。旧 ONNX 导出器给出的 InstanceNorm `train=True` 警告表示该算子使用输入统计量，不表示正在更新模型权重；真实数值对照是是否兼容的重要证据。

### 8.8 尚未取得的数据，不写成已完成训练

本轮还测试了 COFW 官方公开下载，入口为：

```powershell
python -m research.fetch_official_data cofw --file COFW.zip --extract --report reports/cofw-grayscale-resume.json
```

[当前官方 CaltechDATA 记录](https://data.caltech.edu/records/bc0bf-nc666)可读，但彩色包只收到 1086044 字节，灰度包只收到 293760 / 178022021 字节。正常续传遇到 HTTP 500 或 SSL EOF，Chrome 普通下载跳转显示 `ERR_BLOCKED_BY_CLIENT`；没有修改安全设置。残缺包未通过公布的 MD5，程序保留 `.part` 并拒绝把它当作可训练数据。具体记录为 `reports/cofw-access.json`。300-W 下载登记需要真实个人联系信息，未捏造身份。因此本项目不能报告真实 COFW/300-W 训练 NME；已有 68 点基线也不能直接当作原始 COFW 29 点模型。

### 8.9 自测题与参考答案

1. **500 个 step、batch size 为 2，是否代表 500 个 epoch？** 不是。代表 500 次 D 更新和 1000 次图片抽样；训练池 4096 张时不足一轮。每五步更新一次 G，则有 100 次 G 更新。
2. **三个阶段抽样 2400 次，可以说见过 2400 个不同人吗？** 不可以。图片可能重复，不同图片也可能属于同一个人；总去重图片数与人物数都未审计。
3. **为什么恢复必须保存 Adam 状态？** Adam 除参数外还保存梯度的一阶、二阶统计量。只恢复权重会改变后续优化轨迹。本次恢复了两个优化器，但改变了数据顺序，所以仍不能称精确续接中断点。
4. **为什么最终不是 200+650+500=1350 次更新？** 中间只把前 500 步写入完整检查点，至少后 150 步没有进入恢复模型。正确的最终继承量是 200+500+500=1200。
5. **模型成功载入，为什么仍可能生成错误图？** 参数形状正确不保证运行语义正确。此次 `.eval()` 改变了 InstanceNorm 统计量来源，需复核作者测试行为并进行真实图和数值验证。
6. **FID 从 41.2728 变成 40.9804，能宣布显著提高吗？** 不能。变化约 0.29，只有 768 样本，没有显著性分析，且指标对人脸编辑质量与身份保持的覆盖有限。
7. **ONNX 最大误差很小，能证明图片质量优秀吗？** 不能。它证明部署模型与 PyTorch 实现数值接近；两个实现可以同样产生伪影，质量需要另行评估。
8. **下载器发现 MD5 不符，应删掉验证继续训练吗？** 不应。先保留残缺文件和错误记录，核实传输长度与官方校验值，恢复正常下载；不完整数据不能产生可信实验结果。
