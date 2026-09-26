# 源码地图：在哪里使用、对应哪项要求、怎样验证

本表配合完整 `code-compendium.md` / HTML 阅读。完整源码使用可搜索HTML与Markdown，教程、报告和问题记录另提供PDF。汇编收录的是本次交付的源码快照，不是安装到虚拟环境内的全部第三方库。第三方代码保留原许可；模型权重和数据不是源码。命令在项目根目录执行，`python` 应替换为环境报告中同一解释器，例如 `.\.venv\Scripts\python.exe`。

| 文件/模块 | 使用场景与调用入口 | PDF 要求 | 可重复检查 |
|---|---|---|---|
| `desktop.py` → `Desktop`、`Worker` | 最终原生桌面窗口；直接使用OpenCV读取摄像头并调用Vision，不启动HTTP | 9.2、9.3、统一界面、实时视频 | `python desktop.py`；`python desktop.py --self-test`，见 `reports/desktop-verification.json` |
| `vision3d/native_viewer.py` → `prepare_mesh/start_viewer` | 最终原生 OpenGL 全网格窗口；spawn 子进程独占 GL 上下文，上传全部 38365 顶点/76073 面，Pipe 接收 yaw/capture/close；旧 `render_mesh` 抽样函数已删除 | 8.2、8.3、OpenGL、系统集成 | `python -m vision3d.native_viewer --self-test --output reports/3d-native`；GPU 图元查询76073、35° uniform读回/截图变化、归零像素一致、child退出0；只用公开NASA样图 |
| `vision3d/licenses/pyglet-2.1.16-LICENSE.txt`、`reports/native-viewer-dependencies.json` | 保留 pyglet 上游完整 BSD-3-Clause 许可，记录固定版本、官方来源、wheel公布哈希、许可文件哈希和实际GL上下文；库作为依赖安装，不复制其全部实现进项目 | 第三方许可、环境复现、开源交付 | 来源为官方 pyglet v2.1.16 tag/PyPI；主venv与项目conda均导入通过、pip check通过；主venv实际OpenGL4.6，未改驱动，renderer的16GB为共享内存显示 |
| `scripts/verify_desktop.py` | 实际Tk回调、六效果、同图验证、StarGAN、按钮打开全网格GL和35°回执；可选择摄像头20秒及停止/重开/退出释放；只保存公开样图截图 | 原生桌面验收 | `desktop.py --self-test-no-camera` → `reports/desktop-integration.json`，已实测通过；`--self-test`才打开相机。首轮历史为desktop-verification-first.json/desktop-camera-first.json，第二轮为desktop-verification.json/desktop-camera.json |
| `启动桌面程序.cmd` | Windows双击入口 | 本机交付 | 调用 `start.ps1`，默认原生桌面，`-Web`才启动辅助网页 |
| `scripts/copy_source_delivery.py` | 同步源码至桌面并按SHA256逐文件检查 | 用户新增桌面源码要求 | `python scripts/copy_source_delivery.py --destination <桌面目录> --report <清单路径>` |
| `app.py` → `Vision.detect/feature/verify/process` | YuNet 检测五点、SFace 对齐和特征、同人验证、图片特效 | 2.1、2.4、4.3、9.1–9.3、系统集成 | `python -m unittest discover -s tests -p test_app.py`；`python app.py --port 8765` |
| `app.py` → `decode_image`、`Handler` | 校验上传内容；`/health`、`/process`、`/verify` HTTP 接口 | 系统集成与部署 | 无效图片/无脸/错误参数测试；访问 `http://127.0.0.1:8765/health` |
| `web/index.html` | 原生浏览器界面，图片选择、摄像头采集与结果绘制 | 实时视频、统一界面 | 本地服务启动后使用页面；摄像头/实时跟踪需设备与浏览器权限 |
| `start.ps1` | Windows 快速启动；补齐子进程环境、建venv、幂等安装依赖、下载模型、启动 | 1.1、部署 | `powershell -File start.ps1`；每次核对依赖，已有满足版本的包复用 |
| `hello_world.py`、`Dockerfile`、`.dockerignore` | 构建与容器 Hello World；保留远端已有hello.py目录 | 1.3 | `docker build --target hello -t face-vision-hello .`；`docker run --rm face-vision-hello` |
| `Dockerfile` 的 `lab` 阶段 | 核心CPU图像应用容器；构建时按哈希下载模型和素材 | 1.3、系统部署 | `docker build --target lab -t face-vision-lab .`；`docker run --rm -p 127.0.0.1:8765:8765 face-vision-lab`；不含研究训练/三维生成依赖 |
| `requirements.txt`、`requirements-research.txt`、`requirements-docs.txt` | 应用、研究、文档依赖分开安装 | 1.1、报告/PPT交付 | `python -m pip check`；`python scripts/doctor.py` |
| `scripts/fetch_models.py`、`models/registry.json` | 下载权重、按来源与 SHA256 核验 | 1.1、2.4、8.2 | `python scripts/fetch_models.py`；`python scripts/fetch_models.py --with-3d` |
| `scripts/doctor.py` | 环境、依赖、设备、模型检查并输出报告 | 环境报告 | `python scripts/doctor.py`；读 `reports/environment.json` |
| `scripts/analyze_lfw.py` | 按本地实际JPG路径统计身份频数和长尾直方图，不预设数据集统计值 | 2.2、数据分析报告 | `python scripts/analyze_lfw.py --root data/lfw/lfw --output reports/lfw-distribution`；实际13233图/5749身份，JSON与SVG |
| `scripts/demo.py` | 生成6种效果图、360帧仿射动画演示、计时 JSON | 9.2、9.3、演示视频 | `python scripts/demo.py`；读 `reports/demo-metrics.json` |
| `scripts/make_notebook.py`、`notebooks/01-first-vision-lab.ipynb` | 入门Notebook：图像、检测、相似度、简单实验 | 1.4、OpenCV交付 | 重启kernel后Run All；用nbconvert/nbclient检查全部单元格 |
| `scripts/build_deliverables.py` | 生成教程/报告/问题PDF、PPTX和源码汇编 | 报告、PPT、教程、全部源码文档 | `python scripts/build_deliverables.py`；生成后检查页数、溢出、图像和文件可打开性 |
| `scripts/render_documents.py` | 逐页渲染已生成PDF，输出PNG及拼图供人工视觉检查 | 文档/PPT交付质量 | `python scripts/render_documents.py --output runs/document-preview`；可加 `--slides-pdf` 指向已由Office转换的PPT PDF；渲染成功与版面无问题仍需分别检查 |
| `research/recognition.py` | `EmbeddingNet` ResNet50、`ArcFace`、P-K采样、batch-hard、训练循环 | 5.1、5.2、难例采样 | `python -m research.recognition --help`；`python -m research.smoke`只验证代码，不等于训练达标 |
| `research/lfw.py` | 严格读取6000对、按训练折选阈值、两种后端评估、失败记录 | 2阶段基线、5.3 | 见下文完整命令；读 `reports/lfw-sface.json` |
| `research/data.py` | WIDER原标注转COCO，检查尺寸、框和路径 | 3.2、3.3 | `python -m research.data --images data/WIDER_train/images --annotations data/wider/subset_train_bbx_gt.txt --output data/wider/train.json`；这里使用下载器实际保存的子集标注 |
| `research/fetch_wider_subset.py` | 从官方页面链接的镜像按Range下载小规模真图，校验ZIP条目 | 3.2、3.3 | `python -m research.fetch_wider_subset --help`；本次32张训练/16张验证 |
| `research/configs/wider_retinanet.py` | MMDetection RetinaNet单类基线配置 | 2.3、3.2、3.3 | 在单独MMDetection环境中用其 `tools/train.py`；RetinaNet不等于RetinaFace |
| `scripts/mmdet.Dockerfile`、`scripts/mmdet_pilot.py` | 隔离PyTorch2.1/MMCV2.1/MMDet3.3，真实WIDER子集1epoch训练/评价/推理 | 1.1、2.3、3.2、3.3 | WSL中构建镜像并挂载项目后运行，见下文；报告 `reports/wider_pilot.json` |
| `research/prepare_lfw_pilot.py` | 对齐真实LFW小样本训练集，排除官方6000对涉及的身份 | 5.2替代实验 | `python -m research.prepare_lfw_pilot --help`；`reports/lfw-pilot-data.json`记录52身份/133图、无身份重叠 |
| `research/landmarks.py` | `.pts`清单、68点回归训练、外眼角归一NME、仿射对齐、推理图 | 4.2、4.3 | `python -m research.landmarks prepare --help` / `train --help` / `infer --help` |
| `research/stargan.py` | 条件生成器、真假/属性判别器、梯度惩罚、重建、编辑与FID/IS | 7.1–7.3 | `python -m research.stargan train --help` / `generate --help` / `metrics --help` |
| `research/optimize.py` | Linear动态INT8、ONNX导出、ORT数值对比、延迟、可选LFW | 6.1–6.3 | `python -m research.optimize --synthetic-smoke --output reports/optimization-smoke`；随机权重结果仅验证管线 |
| `research/smoke.py` | 用随机张量检查各网络前后向、采样、优化器与checkpoint | 研究管线验证 | `python -m research.smoke --output reports/research-smoke.json` |
| `research/__init__.py`、`vision3d/__init__.py` | Python包入口标记，使 `python -m 包.模块` 导入结构明确 | 工程复现 | 由上述模块命令实际导入；本身不执行额外模型训练 |
| `tests/test_app.py` | 样图六种效果、图片编解码、同图比对、无脸和参数边界 | 2.4、9.3、接口可靠性 | `python -m unittest discover -s tests -p test_app.py`；需要下载应用模型与样图 |
| `tests/test_research.py` | pairs格式、数据防泄漏相关边界、NME、采样等逻辑回归检查 | 指标可信性 | `python -m unittest discover -s tests -p test_research.py` |
| `scripts/verify_local.py` | HTTP成功路径、错误输入、来源限制、页面/三维查看器可访问性 | 系统验收 | 先启动app，再 `python scripts/verify_local.py`；报告http-verification.json；HTTP可访问不等于WebGL画面已渲染 |
| `vision3d/mobilenet_v1.py` | 来源于3DDFA_V2的回归骨干，保留上游版权 | 8.2 | 被 `vision3d/reconstruct.py` 内部 `export_onnx` 函数加载；权重映射采用严格检查，不存在独立export_onnx模块 |
| `vision3d/reconstruct.py` | 图片→62参数→3DMM个体形状/表情→38365顶点→OBJ | 8.1、8.2 | `python -m vision3d.reconstruct --image assets/sample.jpg --output reports/3d` |
| `vision3d/check.py` | 验证受限pickle、有限三维坐标、镜像输入会改变回归参数 | 8.2可信性 | `python -m vision3d.check`；`reports/3d/selfcheck.json`；输入相关性不等于几何精度 |
| `vision3d/viewer-template.html` | WebGL顶点、法线、三角面交互渲染模板 | 8.3 | 已有真实渲染和旋转35°截图 `reports/3d/viewer-screenshot.png`；复现时生成viewer.html再访问本地服务 `/3d` |

## LFW 实验命令与口径

```powershell
# 预训练应用基线；保留所有pairs，处理失败的pairs计为错误。
python -m research.lfw --backend sface --root data/lfw/lfw --pairs data/lfw/pairs.txt --models models --threads 2 --on-failure count-incorrect --output reports/lfw-sface.json

# 独立保存的第二次实验：固定几何主体选择，不能覆盖首轮。
python -m research.lfw --backend sface --root data/lfw/lfw --pairs data/lfw/pairs.txt --models models --threads 2 --face-policy largest --on-failure count-incorrect --output reports/lfw-sface-largest.json

# 本次真实小样本ResNet50评估；使用与训练完全一致的对齐图像。
python -m research.lfw --backend resnet --root data/lfw-aligned --pairs data/lfw/pairs.txt --checkpoint runs/arcface-pilot/last.pt --threads 2 --output reports/lfw-arcface-pilot.json
```

实际本机数据根目录可能不同，应使用下载报告中存在的路径，不能为了让命令不报错而创建空目录。ResNet评估输入需要与训练对齐程序一致。当前SFace基线明确记录了多人/未检出失败；失败全部计错的保守正确率，不等于只在成功检测子集上计算的识别准确率。

## 研究训练的完整入口示例

本次真实ArcFace替代数据实验在主venv运行：

```powershell
python -m research.prepare_lfw_pilot
python -m research.recognition --data data/lfw-pilot-train --output runs/arcface-pilot --epochs 3 --p 4 --k 2 --embedding-dim 128 --lr 0.01 --triplet-weight 0.1 --workers 0 --threads 2 --seed 42 --device cpu
```

准备程序选取52个未出现在官方pairs中的身份、133张真实图，训练与评估身份交集为空。这里的训练集来自LFW的非pairs身份，属于小样本替代实验，不是MS-Celeb-1M训练。P-K重采样每epoch产生136个样本次数，不等于有136张不同图。输出为 `runs/arcface-pilot/config.json`、`history.csv`、`training-curves.svg`、`last.pt`；loss下降不自动证明验证准确率提高。

本次已执行的MMDetection小样本闭环，在WSL终端进入挂载的项目根目录后运行：

```bash
docker build -f scripts/mmdet.Dockerfile -t face-vision-mmdet:3.3 .
docker run --rm -v "$PWD:/project" face-vision-mmdet:3.3
```

先准备好下载器产生的 `data/wider/train.json`、`val.json` 与对应图像。该命令默认随机初始化、1epoch、4CPU线程；本次输出AP为0是实测结果，不应把命令执行成功理解为已训练出可用检测器。

以下命令描述获得合法完整数据后如何运行，不代表对应训练已经完成。运行前读取 `--help`，核对数据与本机环境。

```powershell
python -m research.landmarks prepare --root data/300w/train --output data/300w/train.json
python -m research.landmarks prepare --root data/300w/val --output data/300w/val.json
python -m research.landmarks train --root data/300w/train --train data/300w/train.json --val-root data/300w/val --val data/300w/val.json --epochs 20 --output runs/landmarks

python -m research.stargan train --root data/celeba/img_align_celeba --labels data/celeba/list_attr_celeba.txt --partition data/celeba/list_eval_partition.txt --output runs/stargan
python -m research.stargan generate --checkpoint runs/stargan/last.pt --image assets/sample.jpg --targets 0,1,0,0,1 --output runs/stargan/example.png
python -m research.stargan metrics --real data/celeba-eval --generated runs/stargan/generated --output runs/stargan-metrics.json

```

对本次真实ArcFace checkpoint进行量化/导出及完整LFW比较的入口如下；与前述尚待完整数据的训练命令分开：

```powershell
python -m research.optimize --checkpoint runs/arcface-pilot/last.pt --images data/lfw-pilot-train --lfw-root data/lfw-aligned --pairs data/lfw/pairs.txt --samples 8 --threads 2 --repeats 10 --output runs/arcface-pilot/optimized
```

`--images`这里提供8张真实对齐训练图用于计时与数值一致性；任务准确率另由 `--lfw-root` 的7701张独立身份图、6000个官方pairs决定。原始输出 `runs/arcface-pilot/optimized/comparison.json`，便于公开核查的副本为 `reports/arcface-pilot/comparison.json`。FP32为52.6667%、动态Linear INT8为53.0167%，本次均未达到98.5%；ONNX本次只评数值与延迟，没有另外计算其完整LFW准确率。

StarGAN的 `--targets` 必须与checkpoint保存的属性顺序完全一致；例中的五个值不是对输入照片真实属性的断言。FID/IS使用有意义规模的独立真实集和生成集，不能把重复同一张图片当成大样本。MMDetection的训练脚本来自其安装/克隆版本，项目配置通过 `mmdet::` 继承上游配置；必须运行在匹配的MMCV环境中。

## 如何读完整源码汇编

按本表定位功能，再在汇编中搜索原始相对文件路径。先看入口参数和返回值，然后顺着调用看预处理、模型、后处理、失败分支，最后看测试。汇编行号只适用于该快照；以仓库commit和汇编生成时间识别版本。应优先运行仓库中的原始文件；教程PDF中的片段用于学习，不代替完整源码。

## 视频兼容格式导出

`scripts/export_video.py` 调用 `imageio-ffmpeg` 包附带的 FFmpeg，把 `reports/demo.mp4` 完整转为 H.264/yuv420p，输出到项目上一层的 `项目演示视频.mp4`。执行 `python scripts/export_video.py` 后逐帧解码核对360帧、20 FPS、18秒；`reports/video-export.json` 记录格式和结果。重新编码不修改原始模型性能数据。FFmpeg版本与授权以所用二进制和上游声明为准，源码仓库不包含该二进制。
