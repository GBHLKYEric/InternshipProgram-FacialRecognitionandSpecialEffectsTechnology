# Face Vision Lab 人脸视觉实验室

独立开源教育项目：最终前端是 Windows 原生桌面窗口，直接调用本机摄像头，实现检测、五点定位、双图验证、动态贴纸与美颜、3DDFA 单目重建，并接入本地 StarGAN 属性编辑模型。另提供真实训练/评估代码和专业中文入门教程。与字节跳动无隶属关系，不包含其内部 SDK。

## 本地启动

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/fetch_models.py
.\.venv\Scripts\python.exe desktop.py
```

Windows 也可双击 `启动桌面程序.cmd`，或运行 `powershell -File start.ps1`。默认直接打开原生桌面程序，无需浏览器。图像仅在本机处理；摄像头帧不保存。运行环境为 Python 3.12，依赖版本已固定。GAN 和三维功能需要研究依赖及对应的本地模型，具体见[桌面程序教程](docs/desktop-guide.md)。

早期网页应用保留为辅助实验：显式运行 `powershell -File start.ps1 -Web -Port 8765` 后访问 <http://127.0.0.1:8765>。网页与桌面有各自的测试报告。

研究和文档依赖分别在 `requirements-research.txt`、`requirements-docs.txt`。MMDetection 使用独立 `scripts/mmdet.Dockerfile`，避免旧版 MMCV 与主环境冲突。

## 已验证成果

- YuNet/SFace 本地应用、输入校验、六种显示模式、双图相似度和 Jupyter 实验。
- 原生Tkinter窗口直接调用DirectShow摄像头：640×480、20.0601秒149帧；六效果、同图验证、原生三维重建/旋转35°、停止释放/重开/退出释放通过。首轮同时有训练任务，7.4277FPS，中位延迟136.4935ms/P95 159.3555ms；不据此称流畅度达标。
- 完整WIDER数据12880训练/3226验证图已校验；预训练YuNet完整验证的easy/medium/hard AP为0.8844221/0.8656839/0.7504021，与固定OpenCV Zoo参考实现逐项差为0。这是完整基线，不是自训RetinaNet成绩。
- BytePS两工作进程概念模拟、ByteNN的ORT图优化概念模拟、Intel Arc XPU算子与GAN反向传播测试、火山引擎官方人像融合样例体验均有独立报告。
- SFace 完整 LFW 6,000 对 / 10 折：主体人脸策略 **98.9333%**，标准差 **0.3887 个百分点**。首轮严格单脸策略 **72.25%**，原报告保留。第二轮依据首轮错误分析改变了预处理，不能称为未经调整测试集的独立认证；结果属于预训练 SFace，不能归给自训 ResNet50。
- 3DDFA_V2 真实照片重建：38,365 顶点、76,073 三角面，OBJ 与 WebGL 查看器。没有三维真值，不报告重建精度。
- 18 秒 / 360 帧的真实逐帧特效推理视频；输入是公开照片的仿射动画，不是实拍摄像头录像。
- WIDER 32 张训练 / 16 张验证的真实 RetinaNet 训练、推理、COCO 评估闭环；1 epoch 尚未收敛，mAP=0，不是正式 WIDER benchmark。
- ResNet50+ArcFace 真实小样本训练：52个LFW非pairs身份、133图、3epoch；完整6000对的FP32准确率52.6667%、动态Linear INT8为53.0167%，均未达到98.5%。这是替代数据pilot，不是MS-Celeb-1M训练。
- 同一真实checkpoint的优化比较：batch8、CPU2线程、10次计时，FP32中位104.477ms、INT8 105.454ms、ONNX Runtime 155.836ms；本次两种转换均未加速。300-W/COFW关键点真实训练仍未完成。
- CelebA真实StarGAN微调已完成：使用官方StarGAN项目配套数据和作者预训练G/D，最终保留1200次D/240次G更新；训练池从512扩至4096，实际抽样400+1000+1000次，总去重图片数未知。不是从零全量训练；中断丢弃更新、损失修正和seed43恢复均有原始记录。
- StarGAN同协议前后各768生成图/768独立参考图，均XPU生成、同torch构建、CPU计算标准FID/IS：FID **41.272788→40.980400**，IS均值 **3.088072→3.098037**；微小变化不构成统计显著提升。最终G已导出ONNX并接入原生桌面程序，真实图PyTorch/ORT最大绝对误差1.97×10^-6。见[实验汇总](reports/stargan-experiment/report.json)与[第8节教程](docs/continued-experiments.md#8-用真实-celeba-数据完成-stargan-微调评估与桌面部署)。
- Docker Hello World、模型 SHA256 校验、环境核验与可重复训练配置。

## 文档与全部代码

- [18课专业入门教程](docs/tutorial-zh.md)
- [原生桌面程序使用与代码教程](docs/desktop-guide.md)
- [继续实验专业教程](docs/continued-experiments.md)
- [完整源码汇编及使用位置](docs/code-compendium.md)，本地浏览 [HTML](docs/code-compendium.html)
- [文件、调用入口与PDF任务映射](docs/code-map.md)
- [逐项验收矩阵](docs/requirements-matrix.md)
- [项目总结报告](docs/project-report.md)
- [实际问题、根因、处理与复测](docs/issues-and-fixes.md)
- [官方资料与许可依据](docs/sources.md)
- [研究管线命令](research/README.md)

## 复现实验

```powershell
python -m unittest discover -s tests
python desktop.py --self-test
python scripts/demo.py
python scripts/make_notebook.py
python scripts/fetch_models.py --with-3d
python -m vision3d.reconstruct --image assets/sample.jpg --output reports/3d
python scripts/build_deliverables.py
```

三维ONNX导出是 `vision3d/reconstruct.py` 内部函数，首次重建会调用，不存在独立的 `vision3d.export_onnx` 模块。完整LFW按 [官方来源与SHA256](docs/sources.md) 获取并校验：把原图压缩包解压到 `data/lfw/`，使实际图像位于 `data/lfw/lfw/身份名/文件名.jpg`，把官方pairs保存为 `data/lfw/pairs.txt`；仓库没有 `scripts/fetch_lfw.py`。本机真实训练/评估的完整命令见 [源码地图](docs/code-map.md)。所有训练数据、权重和大文件仅保留本地，开源仓库提供来源、哈希及生成代码。安装研究依赖后运行各模块 `--help` 查看精确参数。PDF/PPT生成需要 Windows 微软雅黑字体。

```bash
docker build --target hello -t face-vision-hello .
docker run --rm face-vision-hello
docker build --target lab -t face-vision-lab .
docker run --rm -p 127.0.0.1:8765:8765 face-vision-lab
```

上面命令在本机已就绪的WSL Docker环境中执行。`lab`镜像实际构建并运行通过：容器中样图检测到1张脸，组合特效与同图相似度1.0通过，宿主HTTP健康检查返回200。它包含核心YuNet/SFace图像应用和文档；构建时下载并校验模型/样图，不依赖构建目录中已有的大权重。镜像默认不包含研究训练依赖、3DDFA重建模块和已生成的三维结果；三维实验需在本地研究环境按上述命令另外生成，再由本地应用查看。

应用的Host校验要求对外端口与程序监听端口一致。如果8765已占用，可用：

```bash
docker run --rm -p 127.0.0.1:8766:8766 face-vision-lab python app.py --host 0.0.0.0 --port 8766
```

然后访问 `http://127.0.0.1:8766`。不要仅把映射改成8766:8765，否则Host端口校验会拒绝请求。MMDetection真实训练使用另一个兼容镜像：

```bash
docker build -f scripts/mmdet.Dockerfile -t face-vision-mmdet:3.3 .
docker run --rm -v "$PWD:/project" face-vision-mmdet:3.3
```

## 尚未满足的完整研究验收

300-W/COFW真值训练及NME、MS-Celeb-1M指定数据训练及自训识别准确率目标、完整WIDER自训模型的最终评测、真实手机端性能和鉴权云API部署仍未全部完成。CelebA StarGAN已经真实微调、计算标准FID/IS并部署到桌面；其有限规模实验不等于从零全量训练、论文复现或统计显著质量提升。COFW官方连接中断和300-W登记条件均保留证据，不用残缺数据或合成图充当训练成果。火山引擎官方样例体验与鉴权服务部署分开验收。详见[验收矩阵](docs/requirements-matrix.md)。

## 许可

本项目原创源码为 MIT。第三方实现保留其许可证。模型、BFM派生资产与人脸数据遵守各自条款，不能因代码MIT便推断它们可任意商用或再分发。公开仓库不包含私人人脸库、数据集、密钥或训练权重。原仓库的 `hello.py/Untitled-1.md` 已原样保留。
