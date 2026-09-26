# 执行问题、根因、处理与复核

记录日期：2026-09-26。下表区分“已修复并运行”“识别到但未完成”“设计限制”；不把未出现的问题写成本次发生的故障。报告中CPU结果、预训练模型结果、随机张量测试和真实数据训练分别标识。

| 实际遇到的问题 | 定位与根因 | 采取的处理 | 复核与剩余限制 |
|---|---|---|---|
| 受限Windows子进程中SSL/DNS、venv操作失败 | 进程环境缺少 `SystemRoot`、`WINDIR`、`COMSPEC` | 为相关子进程补齐必要变量；`start.ps1`保留同样防护，不改系统全局设置 | 隔离环境成功启动、依赖和模型下载可继续；版本见 `reports/environment.json` |
| Python解释器/依赖组合容易混用 | Codex内置Python、项目venv、WSL Docker属于不同环境 | 项目使用明确的 `.venv/Scripts/python.exe`；研究依赖单独列出；记录freeze | 环境报告与实际smoke报告对应；不能把容器Python版本写成桌面版本 |
| MMDetection和主环境最新版依赖不兼容 | 3.3.0有严格版本范围，MMCV还包含编译算子 | 单独Docker锁定Python3.10.21、torch2.1.0CPU、torchvision0.16CPU、NumPy1.26.4、mmcv2.1、mmengine0.10.7、mmdet3.3、OpenCV4.11 | NMS真实运行、WIDER小样本训练/验证/推理全链成功；不改主venv、不删除版本assert |
| Chrome浏览器自动化入口不可用 | 插件返回 `Browser is not available: chrome`；原生控制无法确定目标URL后安全停止 | 保留错误，继续本地HTTP/API与文件验证，不称Chrome交互已通过 | 接口/截图或其他浏览器证据应注明来源，摄像头真人流程需要可用控制连接 |
| 旧GitHub创建仓库接口限流 | `create_repo`入口失败 | 检索已连接GitHub，使用用户已存在公开仓库 `GBHLKYEric/face_ai_project` | 已开始写入LICENSE；完整发布以远程文件读取和提交SHA为准 |
| Docker首次拉取镜像出现CloudFront EOF | 网络传输中断 | 通过正常重试重新拉取，不修改校验或使用未知镜像 | 已真实build与容器Hello World；WSL Docker 29.1.3、容器Python 3.12.14 |
| LFW原网站不可达 | 网页请求失败，并不表示数据不存在 | 查scikit-learn官方维护下载器，使用其Figshare镜像和SHA256 | 下载后运行完整6000对评估；没有用任意转载数据替代 |
| LFW首次预训练基线低，很多图检测出多个人脸 | 应用验证要求恰好一张脸；原LFW部分图片含多脸/检测多框，导致处理失败 | 保留首轮72.25%与全部失败；另加固定最大面积、并列中心最近的几何选脸策略，再跑完整6000对 | `reports/lfw-sface-largest.json`第二次98.9333%±0.38873个百分点，7701图全部成功；它是经过首轮错误分析后的第二次实验，不是自训ArcFace或完全未触碰测试集的独立最终认证 |
| 余弦同图得分略超过1 | 浮点计算舍入导致 `1.0000000488` | 显示/接口分数裁剪到数学范围[-1,1]；核对归一化 | 同图结果只是数值一致性检查，不称准确率100% |
| 动态INT8未加速 | ResNet大部分是Conv2d，当前动态量化仅覆盖Linear；转换开销与硬件有关 | 原样记录FP32/INT8/ORT计时，不写预设加速倍数 | 随机权重smoke中48.840ms→52.653ms，实际变慢；准确率字段为null |
| ONNX导出API随PyTorch变动 | 现代导出器与旧教程默认行为不同 | 当前代码显式 `dynamo=False`、opset17，并用checker与ORT对比 | 随机权重smoke最大绝对差1.71e-7；不是训练模型精度验证 |
| 3DDFA checkpoint含辅助头和不同参数名称 | 发布checkpoint带 `module.` 前缀、`fc_param`与未使用 `fc_lm` | 根据官方结构映射到本地网络，忽略明确未使用辅助头，其他参数strict加载 | 实际生成38365顶点、76073面；matplotlib静态渲染与WebGL源码已生成；无真实三维GT，未测几何精度 |
| 三维资产包含pickle | pickle通常可执行对象反序列化 | 固定来源SHA256，限制只允许所需NumPy类型的unpickler | 仍仅加载已校验官方资产；不接受用户任意pickle |
| 数据与权重许可不同于源码许可 | CelebA/300-W限制研究用途，InsightFace权重非商业研究 | 公开加载器、配置、来源和许可证；数据及受限权重留本地 | 未取得的数据训练保持blocked，源码开源不改变第三方许可 |
| BytePS/ByteNN与PDF描述有偏差 | BytePS真实存在但已归档、依赖CUDA/NCCL；没有获得ByteNN SDK | 教程解释原理与限制，不虚构内部工具；ORT只作为公开推理路线 | CPU单机没有完成真实BytePS分布式训练；ByteNN仍待SDK |
| OpenCV模型下载只有131/133字节，无法作为ONNX加载 | GitHub raw返回Git LFS指针 | 改官方media下载路径，按指针内SHA256核验232589字节YuNet与38696353字节SFace | 真实检测、特征和全LFW评估通过；没有重装Python掩盖文件问题 |
| JupyterLab安装报No such file or directory | 前端静态资源路径超过Windows传统260字符长度 | 用同一venv解释器的扩展路径 `\\?\C:\...\python.exe` 完成安装 | JupyterLab4.6.4、nbclient/nbformat可import，pip check通过；未改全局长路径注册表 |
| 旧容器pip安装torch依赖失败，回退源码后缺flit_core | pip23.0.1对typing_extensions元数据名称规范化兼容问题 | 先升pip25.3，保留setuptools<81兼容旧OpenMMLab的pkg_resources依赖 | 固定环境装完并成功训练；记录完整freeze |
| WIDER全量包下载未完 | 文件不完整，不应当有效ZIP | 保留为 `.incomplete`；从官方页面明确链接镜像读取ZIP Range子集并校验来源/CRC | 本机32张训练/16张验证是真图；不是完整数据集 |
| WIDER随机初始化一轮后评估为空 | 候选分数约0.01，默认0.05阈值全过滤 | 评价score_thr设0保留排序候选，同seed重新训练完整评价 | 真实AP=0、AP50=0、medium-size AP=.006；不以保留低分框冒充收敛 |
| MMDet单图推理缺model.cfg | Runner创建模型未自动挂推理API所需配置 | 明确设置 `runner.model.cfg=cfg` | 训练→验证→inference_detector→报告成功；低分top10图标为未收敛 |
| slim容器提示gcc not found | 框架环境采集检查编译器，而当前用预编译wheel | 核查CPU NMS与真实训练成功，不为无用检测安装编译器 | 日志中的GPU number1为单进程显示；实际CUDA=False、CPU运行 |
| 300-W训练图下载要求登记 | 官方入口要求姓名/邮箱/机构；600张challenge test不等于训练数据 | 未编造信息，不把测试集切成训练后声称标准协议 | 300-W真实训练仍blocked；代码入口已提供 |

## 已有项目的审计与本次工程关系

用户现有公开仓库 [GBHLKYEric/ByteDance_stage_1](https://github.com/GBHLKYEric/ByteDance_stage_1) 已只读检查，未修改。它包含教学演示、配置和CelebA标注CSV。本次源码发布到 `face_ai_project`，并不覆盖原仓库。旧脚本可以帮助理解张量/API，但下述结果不能作为原PDF科研要求已完成的证据。

| 旧文件 | 源码中可核验的事实 | 本次如何避免混淆 |
|---|---|---|
| `train_arcface_gpu.py` | 输入为 `torch.randn(10000,100)`，标签随机，主体为两层Linear；不是人脸图片或ResNet50 | 新增真正ResNet50+ArcFace、ImageFolder、P-K采样、训练曲线与checkpoint；仍未把随机smoke记作真实训练 |
| `plot_training_cruve.py` | 准确率数组手写，末值98.6，被标为LFW验证曲线 | 只允许从本次history/评估JSON作图；不继承98.6%结论 |
| `lfw_eval_demo.py` | 只读 `test.jpg` 与 `test2.jpg`，固定阈值0.4；没有官方pairs与十折 | 本次完整6000对、训练折选阈值、全量失败记录 |
| `stargan_demo.py` | 只有简化生成器、随机图和MSE，没有判别器或CelebA属性训练 | 新代码有生成器、判别器、属性损失、重建损失、梯度惩罚和真实数据入口；没有数据就不报编辑质量 |
| `generate_charts.py` | CelebA百分比与LFW人数分布为写死数组 | 新分析应读取真实文件；历史数组最多作演示，不能称本机统计 |
| `face_3d_mesh.py` | 调用MediaPipe并在二维图绘制连接线，没有PRNet/3DDFA模型与OBJ导出 | 本次3DDFA真实参数回归、输入相关3DMM、OBJ与多角度图；渲染后端单独记录 |
| `export_onnx.py` | 导出完成消息称可嵌入APP，未见完整目标平台推理验证 | 本次checker+ORT数值对比；移动端性能仍不宣称已验证 |

审计是在说明证据范围，不否认演示代码的学习价值。科研报告可保留“早期原型”，但必须明确随机数据、手填数值和真实基准的差别。

## 尚待解除的外部条件

完整CelebA图像与属性训练、WIDER全量正式训练（小样本已完成）、300-W或COFW训练测试数据、可核验MS-Celeb-1M授权数据、可用的浏览器/摄像头连接、真实移动设备，以及云服务凭据。MMCV兼容环境已在独立Docker中解除。源码与教程已给出推进路径；缺少外部条件时，不填补虚构数据、不购买未知服务、不伪造项目通过。
