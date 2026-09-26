# Face Vision Lab 人脸视觉实验室

独立开源教育项目：在普通 Windows CPU 电脑完成检测、五点定位、双图验证、动态贴纸与美颜、3DDFA 单目重建，并提供真实训练/评估代码与专业中文入门教程。与字节跳动无隶属关系，不包含其内部 SDK。

## 本地启动

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/fetch_models.py
.\.venv\Scripts\python.exe app.py
```

打开 <http://127.0.0.1:8765>。Windows 也可运行 `powershell -File start.ps1`。图像仅在本机处理；摄像头需使用者授权。运行环境为 Python 3.12，依赖版本已固定。

研究和文档依赖分别在 `requirements-research.txt`、`requirements-docs.txt`。MMDetection 使用独立 `scripts/mmdet.Dockerfile`，避免旧版 MMCV 与主环境冲突。

## 已验证成果

- YuNet/SFace 本地应用、输入校验、六种显示模式、双图相似度和 Jupyter 实验。
- SFace 完整 LFW 6,000 对 / 10 折：主体人脸策略 **98.9333%**，标准差 **0.3887 个百分点**。首轮严格单脸策略 **72.25%**，原报告保留。第二轮依据首轮错误分析改变了预处理，不能称为未经调整测试集的独立认证；结果属于预训练 SFace，不能归给自训 ResNet50。
- 3DDFA_V2 真实照片重建：38,365 顶点、76,073 三角面，OBJ 与 WebGL 查看器。没有三维真值，不报告重建精度。
- 18 秒 / 360 帧的真实逐帧特效推理视频；输入是公开照片的仿射动画，不是实拍摄像头录像。
- WIDER 32 张训练 / 16 张验证的真实 RetinaNet 训练、推理、COCO 评估闭环；1 epoch 尚未收敛，mAP=0，不是正式 WIDER benchmark。
- ResNet50+ArcFace 真实小样本训练：52个LFW非pairs身份、133图、3epoch；完整6000对的FP32准确率52.6667%、动态Linear INT8为53.0167%，均未达到98.5%。这是替代数据pilot，不是MS-Celeb-1M训练。
- 同一真实checkpoint的优化比较：batch8、CPU2线程、10次计时，FP32中位104.477ms、INT8 105.454ms、ONNX Runtime 155.836ms；本次两种转换均未加速。关键点和StarGAN有源码与计算路径测试，尚无指定数据训练成果。
- Docker Hello World、模型 SHA256 校验、环境核验与可重复训练配置。

## 文档与全部代码

- [18课专业入门教程](docs/tutorial-zh.md)
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

CelebA StarGAN 正式训练及 FID/IS、300-W/COFW 真值训练及 NME、MS-Celeb-1M 指定数据训练、完整 WIDER 正式评测、手机端性能和火山引擎服务体验仍需要相应数据授权、训练资源或账号服务。短训/随机输入代码烟测不是这些研究指标的完成证明。详见验收矩阵。

## 许可

本项目原创源码为 MIT。第三方实现保留其许可证。模型、BFM派生资产与人脸数据遵守各自条款，不能因代码MIT便推断它们可任意商用或再分发。公开仓库不包含私人人脸库、数据集、密钥或训练权重。原仓库的 `hello.py/Untitled-1.md` 已原样保留。
