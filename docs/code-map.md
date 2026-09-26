# 源码地图：在哪里使用、对应哪项要求、怎样验证

本表配合完整 `code-compendium.md` / HTML 阅读。完整源码使用可搜索HTML与Markdown，教程、报告和问题记录另提供PDF。汇编收录的是本次交付的源码快照，不是安装到虚拟环境内的全部第三方库。第三方代码保留原许可；模型权重和数据不是源码。命令在项目根目录执行，`python` 应替换为环境报告中同一解释器，例如 `.\.venv\Scripts\python.exe`。

| 文件/模块 | 使用场景与调用入口 | PDF 要求 | 可重复检查 |
|---|---|---|---|
| `app.py` → `Vision.detect/feature/verify/process` | YuNet 检测五点、SFace 对齐和特征、同人验证、图片特效 | 2.1、2.4、4.3、9.1–9.3、系统集成 | `python -m unittest discover -s tests -p test_app.py`；`python app.py --port 8765` |
| `app.py` → `decode_image`、`Handler` | 校验上传内容；`/health`、`/process`、`/verify` HTTP 接口 | 系统集成与部署 | 无效图片/无脸/错误参数测试；访问 `http://127.0.0.1:8765/health` |
| `web/index.html` | 原生浏览器界面，图片选择、摄像头采集与结果绘制 | 实时视频、统一界面 | 本地服务启动后使用页面；摄像头/实时跟踪需设备与浏览器权限 |
| `start.ps1` | Windows 快速启动；修复受限子进程环境、建 venv、下载模型、启动 | 1.1、部署 | `powershell -File start.ps1`；已有环境不会反复安装 |
| `hello_world.py`、`Dockerfile`、`.dockerignore` | 构建与容器 Hello World；保留远端已有hello.py目录 | 1.3 | `docker build --target hello -t face-vision-hello .`；`docker run --rm face-vision-hello` |
| `requirements.txt`、`requirements-research.txt` | 应用与研究依赖分开安装 | 1.1 | `python -m pip check`；`python scripts/doctor.py` |
| `scripts/fetch_models.py`、`models/registry.json` | 下载权重、按来源与 SHA256 核验 | 1.1、2.4、8.2 | `python scripts/fetch_models.py`；`python scripts/fetch_models.py --with-3d` |
| `scripts/doctor.py` | 环境、依赖、设备、模型检查并输出报告 | 环境报告 | `python scripts/doctor.py`；读 `reports/environment.json` |
| `scripts/demo.py` | 生成6种效果图、360帧仿射动画演示、计时 JSON | 9.2、9.3、演示视频 | `python scripts/demo.py`；读 `reports/demo-metrics.json` |
| `scripts/make_notebook.py`、`notebooks/01-first-vision-lab.ipynb` | 入门Notebook：图像、检测、相似度、简单实验 | 1.4、OpenCV交付 | 重启kernel后Run All；用nbconvert/nbclient检查全部单元格 |
| `scripts/build_deliverables.py` | 生成教程/报告/问题PDF、PPTX和源码汇编 | 报告、PPT、教程、全部源码文档 | `python scripts/build_deliverables.py`；生成后检查页数、溢出、图像和文件可打开性 |
| `research/recognition.py` | `EmbeddingNet` ResNet50、`ArcFace`、P-K采样、batch-hard、训练循环 | 5.1、5.2、难例采样 | `python -m research.recognition --help`；`python -m research.smoke`只验证代码，不等于训练达标 |
| `research/lfw.py` | 严格读取6000对、按训练折选阈值、两种后端评估、失败记录 | 2阶段基线、5.3 | 见下文完整命令；读 `reports/lfw-sface.json` |
| `research/data.py` | WIDER原标注转COCO，检查尺寸、框和路径 | 3.2、3.3 | `python -m research.data --images data/WIDER_train/images --annotations data/wider_face_split/wider_face_train_bbx_gt.txt --output data/wider/train.json` |
| `research/fetch_wider_subset.py` | 从官方页面链接的镜像按Range下载小规模真图，校验ZIP条目 | 3.2、3.3 | `python -m research.fetch_wider_subset --help`；本次32张训练/16张验证 |
| `research/configs/wider_retinanet.py` | MMDetection RetinaNet单类基线配置 | 2.3、3.2、3.3 | 在单独MMDetection环境中用其 `tools/train.py`；RetinaNet不等于RetinaFace |
| `scripts/mmdet.Dockerfile`、`scripts/mmdet_pilot.py` | 隔离PyTorch2.1/MMCV2.1/MMDet3.3，真实WIDER子集1epoch训练/评价/推理 | 1.1、2.3、3.2、3.3 | WSL中构建镜像并挂载项目后运行，见下文；报告 `reports/wider_pilot.json` |
| `research/prepare_lfw_pilot.py` | 对齐真实LFW小样本训练集，排除官方6000对涉及的身份 | 5.2替代实验 | `python -m research.prepare_lfw_pilot --help`；`reports/lfw-pilot-data.json`记录52身份/133图、无身份重叠 |
| `research/landmarks.py` | `.pts`清单、68点回归训练、外眼角归一NME、仿射对齐、推理图 | 4.2、4.3 | `python -m research.landmarks prepare --help` / `train --help` / `infer --help` |
| `research/stargan.py` | 条件生成器、真假/属性判别器、梯度惩罚、重建、编辑与FID/IS | 7.1–7.3 | `python -m research.stargan train --help` / `generate --help` / `metrics --help` |
| `research/optimize.py` | Linear动态INT8、ONNX导出、ORT数值对比、延迟、可选LFW | 6.1–6.3 | `python -m research.optimize --synthetic-smoke --output reports/optimization-smoke`；随机权重结果仅验证管线 |
| `research/smoke.py` | 用随机张量检查各网络前后向、采样、优化器与checkpoint | 研究管线验证 | `python -m research.smoke --output reports/research-smoke.json` |
| `tests/test_research.py` | pairs格式、数据防泄漏相关边界、NME、采样等逻辑回归检查 | 指标可信性 | `python -m unittest discover -s tests -p test_research.py` |
| `scripts/verify_local.py` | HTTP成功路径、错误输入、来源限制、页面/三维查看器可访问性 | 系统验收 | 先启动app，再 `python scripts/verify_local.py`；报告http-verification.json；HTTP可访问不等于WebGL画面已渲染 |
| `vision3d/mobilenet_v1.py` | 来源于3DDFA_V2的回归骨干，保留上游版权 | 8.2 | 被 `export_onnx` 加载；权重映射采用严格检查 |
| `vision3d/reconstruct.py` | 图片→62参数→3DMM个体形状/表情→38365顶点→OBJ | 8.1、8.2 | `python -m vision3d.reconstruct --image assets/sample.jpg --output reports/3d` |
| `vision3d/viewer-template.html` | WebGL顶点、法线、三角面交互渲染模板 | 8.3 | 生成 `reports/3d/viewer.html` 后在支持WebGL浏览器中旋转查看；运行证据与生成证据分开 |

## LFW 实验命令与口径

```powershell
# 预训练应用基线；保留所有pairs，处理失败的pairs计为错误。
python -m research.lfw --backend sface --root data/lfw/lfw --pairs data/lfw/pairs.txt --models models --threads 2 --on-failure count-incorrect --output reports/lfw-sface.json

# 独立保存的第二次实验：固定几何主体选择，不能覆盖首轮。
python -m research.lfw --backend sface --root data/lfw/lfw --pairs data/lfw/pairs.txt --models models --threads 2 --face-policy largest --on-failure count-incorrect --output reports/lfw-sface-largest.json

# 自训ResNet50评估；需真实训练checkpoint，不能换成随机smoke权重。
python -m research.lfw --backend resnet --root data/lfw/lfw --pairs data/lfw/pairs.txt --checkpoint runs/arcface/last.pt --output runs/lfw-arcface.json
```

实际本机数据根目录可能不同，应使用下载报告中存在的路径，不能为了让命令不报错而创建空目录。ResNet评估输入需要与训练对齐程序一致。当前SFace基线明确记录了多人/未检出失败；失败全部计错的保守正确率，不等于只在成功检测子集上计算的识别准确率。

## 研究训练的完整入口示例

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

python -m research.optimize --checkpoint runs/arcface/last.pt --images data/aligned --lfw-root data/lfw/lfw --pairs data/lfw/pairs.txt --output runs/optimized
```

StarGAN的 `--targets` 必须与checkpoint保存的属性顺序完全一致；例中的五个值不是对输入照片真实属性的断言。FID/IS使用有意义规模的独立真实集和生成集，不能把重复同一张图片当成大样本。MMDetection的训练脚本来自其安装/克隆版本，项目配置通过 `mmdet::` 继承上游配置；必须运行在匹配的MMCV环境中。

## 如何读完整源码汇编

按本表定位功能，再在汇编中搜索原始相对文件路径。先看入口参数和返回值，然后顺着调用看预处理、模型、后处理、失败分支，最后看测试。汇编行号只适用于该快照；以仓库commit和汇编生成时间识别版本。应优先运行仓库中的原始文件；教程PDF中的片段用于学习，不代替完整源码。
