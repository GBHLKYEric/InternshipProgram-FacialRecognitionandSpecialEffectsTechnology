# 执行问题、根因、处理与复核

记录日期：2026-09-26—27。下表区分“已修复并运行”“识别到但未完成”“设计限制”；不把未出现的问题写成本次发生的故障。报告中CPU结果、预训练模型结果、随机张量测试和真实数据训练分别标识。

| 实际遇到的问题 | 定位与根因 | 采取的处理 | 复核与剩余限制 |
|---|---|---|---|
| 用户明确最终前端须为本机桌面程序 | 早期实现是本机HTTP网页，虽然数据未出电脑，但交互形态不符合新明确要求 | 新增Tkinter原生窗口和OpenCV VideoCapture，直接调用同一Vision算法；start.ps1默认桌面，网页改为显式-Web | 真实摄像头20秒149帧；停止释放、重新打开、关闭释放，六效果、双图验证及原生三维旋转通过；首轮7.43FPS，在并行训练负载下不称流畅达标 |
| 摄像头与重建运算可能阻塞窗口 | Tk主线程同时做读帧/推理会阻塞事件循环；共享OpenCV检测器也有可变状态 | 单后台线程独占相机与模型，命令队列和容量1结果队列连接主线程；按任务编号丢弃过期结果 | 实际相机生命周期和Tk窗口测试通过；容量1可能丢弃旧处理结果，报告使用窗口提交帧率 |
| 桌面应用与网页性能口径不同 | 原生路径省去JPEG/HTTP，但采用相机读取和Tk图像提交；测试后台负载也不同 | 分别保留desktop-camera-first.json、desktop-camera.json与webcam-chrome.json，写明计时边界 | 不能仅比较7.43与12.46两个FPS就断言框架快慢；未做同负载对照 |
| 受限Windows子进程中SSL/DNS、venv操作失败 | 进程环境缺少 `SystemRoot`、`WINDIR`、`COMSPEC` | 为相关子进程补齐必要变量；`start.ps1`保留同样防护，不改系统全局设置 | 隔离环境成功启动、依赖和模型下载可继续；版本见 `reports/environment.json` |
| Python解释器/依赖组合容易混用 | Codex内置Python、项目venv、WSL Docker属于不同环境 | 项目使用明确的 `.venv/Scripts/python.exe`；研究依赖单独列出；记录freeze | 环境报告与实际smoke报告对应；不能把容器Python版本写成桌面版本 |
| MMDetection和主环境最新版依赖不兼容 | 3.3.0有严格版本范围，MMCV还包含编译算子 | 单独Docker锁定Python3.10.21、torch2.1.0CPU、torchvision0.16CPU、NumPy1.26.4、mmcv2.1、mmengine0.10.7、mmdet3.3、OpenCV4.11 | NMS真实运行、WIDER小样本训练/验证/推理全链成功；不改主venv、不删除版本assert |
| Chrome浏览器自动化入口不可用 | 插件返回 `Browser is not available: chrome`；原生控制无法确定目标URL后安全停止 | 初次失败保留；后续重新连接实际Chrome标签后完成网页交互与20秒摄像头实验 | 最终另按用户要求实现并测试Tk原生桌面，网页与桌面证据分开 |
| 旧GitHub创建仓库接口限流 | `create_repo`入口失败 | 检索已连接GitHub，使用用户已存在公开仓库 `GBHLKYEric/InternshipProgram-FacialRecognitionandSpecialEffectsTechnology` | 已发布38db093c375e6dcd2e129d9048f7857e564cd314；最新文档的收尾提交仍需最终远程核对 |
| Docker首次拉取镜像出现CloudFront EOF | 网络传输中断 | 通过正常重试重新拉取，不修改校验或使用未知镜像 | 已真实build与容器Hello World；WSL Docker 29.1.3、容器Python 3.12.14 |
| LFW原网站不可达 | 网页请求失败，并不表示数据不存在 | 查scikit-learn官方维护下载器，使用其Figshare镜像和SHA256 | 下载后运行完整6000对评估；没有用任意转载数据替代 |
| LFW首次预训练基线低，很多图检测出多个人脸 | 应用验证要求恰好一张脸；原LFW部分图片含多脸/检测多框，导致处理失败 | 保留首轮72.25%与全部失败；另加固定最大面积、并列中心最近的几何选脸策略，再跑完整6000对 | `reports/lfw-sface-largest.json`第二次98.9333%±0.38873个百分点，7701图全部成功；它是经过首轮错误分析后的第二次实验，不是自训ArcFace或完全未触碰测试集的独立最终认证 |
| 余弦同图得分略超过1 | 浮点计算舍入导致 `1.0000000488` | 显示/接口分数裁剪到数学范围[-1,1]；核对归一化 | 同图结果只是数值一致性检查，不称准确率100% |
| 动态INT8未加速 | ResNet大部分是Conv2d，当前动态量化仅覆盖Linear；转换开销与硬件有关 | 原样记录同一真实checkpoint的FP32/INT8/ORT，不写预设加速倍数 | batch8/2线程/10次，104.477ms→105.454ms；state只省约0.82%；完整LFW52.6667%→53.0167%，不能称有统计显著改善 |
| ONNX导出API随PyTorch变动 | 现代导出器与旧教程默认行为不同 | 当前代码显式 `dynamo=False`、opset17，并用checker与ORT对比；保留弃用警告作为升级事项 | 真实checkpoint最大绝对差8.79e-7；ORT中位155.836ms，比同批FP32慢；数值接近不等于已单独验证全LFW准确率 |
| 3DDFA checkpoint含辅助头和不同参数名称 | 发布checkpoint带 `module.` 前缀、`fc_param`与未使用 `fc_lm` | 根据官方结构映射到本地网络，忽略明确未使用辅助头，其他参数strict加载 | 实际生成38365顶点、76073面；matplotlib静态渲染与WebGL源码已生成；无真实三维GT，未测几何精度 |
| 三维资产包含pickle | pickle通常可执行对象反序列化 | 固定来源SHA256，限制只允许所需NumPy类型的unpickler | 仍仅加载已校验官方资产；不接受用户任意pickle |
| 数据与权重许可不同于源码许可 | CelebA/300-W限制研究用途，InsightFace权重非商业研究 | 公开加载器、配置、来源和许可证；数据及受限权重留本地 | 未取得的数据训练保持blocked，源码开源不改变第三方许可 |
| BytePS/ByteNN与PDF描述有偏差 | BytePS真实存在但已归档、依赖CUDA/NCCL；没有获得ByteNN SDK | 按PDF明确的“模拟”范围实现两个真实工作进程的参数服务器，并以ORT图优化演示推理优化 | 加权SGD与独立全批参考误差4.47e-8；ORT数值/耗时比较通过，不宣称安装BytePS或内部ByteNN SDK |
| OpenCV模型下载只有131/133字节，无法作为ONNX加载 | GitHub raw返回Git LFS指针 | 改官方media下载路径，按指针内SHA256核验232589字节YuNet与38696353字节SFace | 真实检测、特征和全LFW评估通过；没有重装Python掩盖文件问题 |
| JupyterLab安装报No such file or directory | 前端静态资源路径超过Windows传统260字符长度 | 用同一venv解释器的扩展路径 `\\?\C:\...\python.exe` 完成安装 | JupyterLab4.6.4、nbclient/nbformat可import，pip check通过；未改全局长路径注册表 |
| 旧容器pip安装torch依赖失败，回退源码后缺flit_core | pip23.0.1对typing_extensions元数据名称规范化兼容问题 | 先升pip25.3，保留setuptools<81兼容旧OpenMMLab的pkg_resources依赖 | 固定环境装完并成功训练；记录完整freeze |
| WIDER全量包下载未完 | 文件不完整，不应当有效ZIP | 保留为 `.incomplete`；从官方页面明确链接镜像读取ZIP Range子集并校验来源/CRC | 早期32/16真图pilot保留；后续已完整下载并校验12880/3226图，完整YuNet验证与官方协议交叉核对通过 |
| WIDER随机初始化一轮后评估为空 | 候选分数约0.01，默认0.05阈值全过滤 | 评价score_thr设0保留排序候选，同seed重新训练完整评价 | 真实AP=0、AP50=0、medium-size AP=.006；不以保留低分框冒充收敛 |
| MMDet单图推理缺model.cfg | Runner创建模型未自动挂推理API所需配置 | 明确设置 `runner.model.cfg=cfg` | 训练→验证→inference_detector→报告成功；低分top10图标为未收敛 |
| slim容器提示gcc not found | 框架环境采集检查编译器，而当前用预编译wheel | 核查CPU NMS与真实训练成功，不为无用检测安装编译器 | 日志中的GPU number1为单进程显示；实际CUDA=False、CPU运行 |
| 300-W训练图下载要求登记 | 官方入口要求姓名/邮箱/机构；600张challenge test不等于训练数据 | 未编造信息，不把测试集切成训练后声称标准协议 | 300-W真实训练仍blocked；代码入口已提供 |
| MS-Celeb-1M授权训练包未取得 | 不能从未知转载包推定拥有可使用数据，也不能把评估身份混入训练 | 另建明确标识的替代pilot：LFW非pairs身份52个/133图，训练与评估身份交集0 | 真实3epoch训练和曲线已保存；不改变PDF指定数据要求的未验收状态 |
| 小样本ArcFace训练loss下降，margin分类准确率仍0 | 随机初始化大型骨干、133图、3epoch与角度间隔共同使任务困难；不能仅由日志断定某一因素贡献 | 同时保存loss/准确率，执行完整6000pairs评估，不手写好看的准确率 | loss40.111934→36.125987；FP32 LFW52.6667%/INT8 53.0167%，均未达标；数据/模型/轮次需后续受控实验改进 |
| README引用不存在的独立模块/下载器 | 把三维脚本内部函数误写成模块，数据下载命令没有纳入仓库 | 删除 `python -m vision3d.export_onnx`，改用实际reconstruct入口；LFW按官方来源/SHA256说明布局，明确没有fetch_lfw.py | 用实际文件清单与CLI逐项校准；不把“若存在”留作可执行教程命令 |
| Windows非法Origin请求触发客户端10053 | 服务在未读取请求体时立即返回403，连接关闭可能造成客户端接收失败 | 先在22MB请求体上限内读取，再按Origin规则返回403；仍保持来源限制 | `scripts/verify_local.py`实测错误路径通过，没有为了消除错误放宽跨源访问 |
| 已存在venv但缺依赖时启动失败 | 旧启动逻辑仅在新建venv后安装依赖 | `start.ps1`改为每次执行幂等依赖安装，已有符合要求的包会复用 | 初次创建和已有环境两种状态遵循同一依赖清单 |
| 新clone缺少本地素材/权重会影响Docker构建和运行 | 原工作目录存在下载文件，可能掩盖干净副本的依赖 | Docker构建阶段按来源/哈希下载，构建上下文排除本机数据及已有权重 | lab镜像已真实构建运行；容器中faces=1、组合特效、同图cosine=1通过，宿主8766健康检查200；镜像只含核心应用，三维另生成 |
| 网页教程里的相对Markdown链接返回404 | 文档链接按文件布局编写，服务没有对应路由 | 生成HTML时重写链接，服务仅允许白名单docs文件路由 | 本地教程可通过服务浏览；不暴露任意文件路径 |
| 摄像头授权异步返回可能覆盖新状态 | getUserMedia在用户切换/停止后才返回，旧结果可能重新绑定视频 | 使用generation token识别过期请求，申请期间禁用按钮并释放过期流 | 代码处理竞态；后续Chrome真实摄像头20秒250帧已测，最终原生桌面另有两轮报告 |
| WSL默认用户没有Docker socket权限 | 该用户不在所需访问上下文中 | 本次Docker命令显式以WSL root运行 | 没有放宽socket权限或修改全局用户组；Hello构建/运行证据已保存 |
| Docker映射8766:8765后浏览器得到403 | 程序监听8765，HTTP Host是外部8766，已有Host端口白名单拒绝不匹配值 | 改为 `-p 127.0.0.1:8766:8766` 并显式 `python app.py --host 0.0.0.0 --port 8766` | 宿主健康检查200；默认8765:8765即可，无须削弱Host校验 |

## 已有项目的审计与本次工程关系

最终交付检查还发现Windows生成的CRLF与远端历史文本混用，使Git将行尾CR识别为尾随空白。项目增加 `.gitattributes` 固定文本LF，并在发布时统一换行；源码汇编在统一后的源码上计算哈希。视频原始输出为OpenCV的MPEG-4 Part 2，为提高常见浏览器/播放器兼容性，另以FFmpeg导出H.264/yuv420p并逐帧解码核对360帧、20 FPS、18秒；这属于交付格式处理，不改变原始模型计时。LibreOffice转换PPT时出现Python prefix提示及终端中文乱码，实际PDF文件存在、16页可渲染；以文件读取和逐页视觉检查为准。

用户现有公开仓库 [GBHLKYEric/ByteDance_stage_1](https://github.com/GBHLKYEric/ByteDance_stage_1) 已只读检查，未修改。它包含教学演示、配置和CelebA标注CSV。本次沿用另一个公开仓库，原名 `face_ai_project`，用户后来重命名为 `InternshipProgram-FacialRecognitionandSpecialEffectsTechnology`；没有覆盖 `ByteDance_stage_1`。旧脚本可以帮助理解张量/API，但下述结果不能作为原PDF科研要求已完成的证据。

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

当前仍缺300-W/COFW可完整验证的训练测试数据、可核验MS-Celeb-1M授权训练包、真实移动设备及鉴权云服务资源。自训ArcFace的98.5%数值目标未通过。后续已经取得完整CelebA并完成有限微调、已经安装Anaconda并运行XPU实验、已经完成两轮真实原生摄像头和完整OpenGL验收；这些不再列为资源缺失。完整WIDER训练按单独进度和最终报告验收。缺少外部条件时，不填补虚构数据、不购买未知服务、不伪造项目通过。

## 续补：Anaconda、Intel XPU 与原生 OpenGL 的实际兼容问题

以下是继续执行后的结果。前面的表格与“尚待解除的外部条件”保留为早期过程记录；涉及本节项目时，以这里列出的实际验证和对应报告为准。例如，Anaconda 后来已安装，真人摄像头已完成两轮，完整网格也已经在原生 OpenGL 上显示，不能继续把这些项目写成未执行。其他数据训练是否完成，应分别核对相应实验报告。

### Anaconda 安装与 Windows 路径

Anaconda Distribution 2026.07-1 的安装器来自官方归档，SHA256 与官方记录一致，Authenticode 为 `Valid`。用户明确接受安装器与官方 `main` 包仓库的条款后才进入安装分支。安装后，基础环境是 conda 26.5.3 / Python 3.14.6，另建项目环境 Python 3.12.14。主 `.venv` 保留；它在此前已经可用，但其存在本身不能作为“安装过 Anaconda”的证据。

| 实际问题或兼容边界 | 定位与处理 | 本轮复核与保留记录 |
|---|---|---|
| 安装工具对子进程环境的依赖未被满足 | 任务子进程缺少 `SystemRoot`、`WINDIR`、`COMSPEC`，曾导致 SSL 随机源、DNS/CIM、ensurepip 失败；只在相关子进程中补齐 Windows 目录变量，安装脚本另从 Windows API 获取当前用户相关目录 | SSL、下载、环境创建恢复；未关闭 TLS 校验，未改代理、持久 PATH 或系统网络设置 |
| Anaconda 安装器拒绝原工作区目标路径 | 本机 `LongPathsEnabled=0`，该安装器显示目标前缀 93 字符、最大允许 54，退出码 2；重新规划真实短目录 | 54 是**这份安装器在当前条件下**报告的前缀上限，不是所有 Windows 文件路径的统一上限 |
| NTFS 8.3 短名称方案失败 | 生成的短名称含 `~`，被安装器作为无效字符拒绝 | 放弃该方案；教程明确记录失败，不把它写成可复现的成功步骤 |
| 短 Junction 指向长物理目录仍提取失败 | 安装器接受短入口，后续 conda 提取器却解析到长物理路径；提取 `anaconda-client` 时出现 `InvalidArchiveError` / `BrokenProcessPool` | 将软件**实际**安装到短目录，再从原工作区建立指向它的联接；不把外层进程池异常直接归因于 CPU、网络或坏包 |
| 删除旧联接并移动目录的组合操作被自动审批拒绝 | 原命令未执行；随后单独核验目标，只把本轮已知失败安装目录在同级重命名为 `work/anaconda-failed-long-path` | 保留失败文件和日志，没有递归删除不明对象，也没有触及其他项目 |
| 初始安装参数不能证明“安装器不写注册表” | `RegisterPython=0` 只说明不注册默认解释器；通过本版本安装器 `/S /?` 核实另外支持 `/NoRegistry=1 /NoShortcuts=1` 后再加上 | 最终使用当前用户安装、禁止 PATH/默认 Python/安装器注册表及快捷方式修改；未运行 `conda init`，未改长路径策略 |
| 主 `.venv` 的 Jupyter 静态资源曾超过传统长路径限制 | 当时以同一解释器的 `\\?\C:\...\python.exe` 扩展路径完成安装，导入与 pip check 通过 | 该修复适用于当时的文件操作，不能推导为所有 Python 包都应使用扩展前缀 |
| conda 项目补包阶段出现 `jsonpointer` 的 `[Errno 22] Invalid argument` | 真实环境目录已经缩短，却沿用了扩展解释器路径；wheel 中脚本位置 `../../Scripts/jsonpointer` 涉及 `..`，与扩展路径处理不兼容。改用普通的真实短解释器路径续装 | `jsonpointer 3.1.1` 的 RECORD 与列举文件完整，pip check 通过；无须关闭校验或强制重装所有依赖 |
| 安装期间项目新增研究和图形依赖 | 前一轮 pip 完成后，串行补装 `torch-fidelity==0.4.0`、随后 `pyglet==2.1.16`，避免同一环境并发安装 | 两包实际导入；项目 conda 和主 `.venv` 的依赖清单更新、pip check 通过。新增 pyglet 的验证范围是包与依赖；原生 GL 实测在主 `.venv` |
| PATH 中存在多个 Python，裸写 pip 容易装到别处 | 主环境明确用 `.venv\Scripts\python.exe -m pip`；conda 用绝对 `conda.exe` 与 `conda run -p <真实环境目录>` | 本轮再次执行 conda run 身份命令，输出 `fv-env-260926\python.exe` 与 `2.14.0+cpu`；不靠永久修改默认解释器解决 |

最终物理安装目录为 `C:\Users\0\Documents\Codex\work\fv-ana-260926`，项目 conda 环境物理目录为 `C:\Users\0\Documents\Codex\work\fv-env-260926`。本线程的 `work/anaconda`、`work/conda-face-vision` 是指向它们的目录联接，不能声称软件仍物理存储在最初的长目录。

[Anaconda 报告](../reports/anaconda-environment.json)记录了 conda 元数据、解释器路径、CPU autograd、YuNet 样图检测、真实 Jupyter 内核执行、Tk 隐藏窗口及 ImageTk 绑定。临时内核已关闭；验证没有打开摄像头。安装产生的本地 Jupyter TCP transport 提示保留在日志，不能替代检查实际内核是否执行成功。运行环境关系与完整命令见[继续实验教程第 7 节](continued-experiments.md#7-把四类运行环境真正弄明白anaconda-安装启动与排错)。

### Intel Arc：分别验证计算后端和图形后端

| 实际问题或易误读的证据 | 采取的处理 | 实际结论与边界 |
|---|---|---|
| 主环境 `torch.xpu.is_available()` 为 False | 确认它安装的是 `torch 2.14.0+cpu`，随后在独立 `work/xpu-env` 从 PyTorch 官方 XPU 索引安装 `torch 2.14.0+xpu` / `torchvision 0.29.0+xpu` | 原 CPU 环境的 False 不能推出 Intel GPU 不受支持；独立环境实际张量、反向传播和 torchvision NMS 通过 |
| 只做矩阵乘法不足以证明生成模型能训练 | 检查项目 StarGAN 所用卷积、反卷积、InstanceNorm、Adam，以及 WGAN-GP 的 `create_graph=True` 二阶反向传播 | 小型 G/D 联合训练步通过；完整生成器也通过前向。原始输入规模和同步计时保存在 [XPU 报告](../reports/xpu-environment.json)，不能推广成所有任务的固定加速倍数 |
| GPU 调用异步，普通函数返回时间会低估执行时间 | 在计时前后调用 `torch.xpu.synchronize()`，记录预热、样本次数、形状与 CPU 线程数 | 测量不含模型/数据转移，后台负载与热状态未控制，只作为限定条件的小实验 |
| OpenCV 的 OpenCL 能力不等于 DNN GPU 推理已成功 | UMat 小运算与 Arc 设备发现通过；请求 YuNet OpenCL target 时，OpenCV 5 提示新图引擎当前不支持 target 设置 | 保留提示，不把成功检出人脸或请求 target 的计时当成 GPU DNN 加速证据；当前 ORT 也没有验证 OpenVINO 后端 |
| 新 PyTorch 的 XPU 通过不代表旧 MMCV 算子可迁移 | 保留 MMDetection 专用 Docker 中 torch 2.1/MMCV 2.1 的 CPU 组合，不把它混装到 XPU 环境 | 旧编译算子的兼容性独立验收，不能只改 `device='xpu'` 就宣称迁移完成 |
| 驱动 renderer 显示 `(16GB)` 容易被误当独立显存 | 与设备的集成 GPU、共享内存属性一并记录 | 这是本机驱动字符串中的共享内存显示，不宣称存在 16GB 独立显存；没有安装新驱动 |

### 原生 OpenGL 全网格与两轮摄像头验收

早期 Tk 三维预览只取约 9,000 个面。完整重建数组虽然保留了 76,073 个三角面，抽样绘制仍会在可见表面留下孔洞。最终已删除该抽样渲染函数，桌面按钮默认调用新增的 [native_viewer.py](../vision3d/native_viewer.py)，由 pyglet 2.1.16 创建原生 OpenGL 窗口，索引绘制全部面，以深度缓冲处理遮挡。Tk 与 OpenGL 各有事件循环，采用独立 `multiprocessing spawn` 进程和 Pipe，避免把 GL 上下文跨 Tk 线程使用。初始化失败会报告错误，不把未验证的回退路径写成已验收功能。

| 实际处理 | 完成证据 | 不应推导的结论 |
|---|---|---|
| 查询官方稳定版本与最低上下文要求，使用现有驱动创建窗口 | pyglet 2.1.16；请求 OpenGL 3.3，实际 Intel Arc/OpenGL 4.6/depth24；[版本、来源和许可记录](../reports/native-viewer-dependencies.json) | 没有修改 GPU 驱动；PyTorch 的 XPU 是否启用与 OpenGL 能否画网格是两个独立问题 |
| 上传完整索引，检测真实 GPU 图元数 | `GL_PRIMITIVES_GENERATED` 实测 76,073，顶点数 38,365 | 全部三角面提交不意味着每个面从当前角度都可见，更不证明三维估计几何准确 |
| 旋转 35° 后读取着色器角度并截图，再归零 | uniform 为 0.6108652353 rad；变化像素占 43.8118%；归零后与初始图逐像素一致；close 后子进程退出码 0 | 截图像素变化是旋转渲染证据，不是重建误差或端到端 FPS 指标 |
| 只有显式 capture 命令才保存 PNG | 本轮仅使用公开 NASA 样图，未打开摄像头；[原生 GL 报告](../reports/3d-native/report.json)和截图可核对 | 不自动保存摄像头脸部或用户三维模型 |
| 保留真实模型的开口与边界 | 目视检查完整脸面连续，嘴部开口及面部模型边界仍存在 | 不把模型面具补成没有依据的完整头部扫描；这些结构与抽样遗漏面造成的孔洞不同 |
| 记录已有非致命提示 | 初始化一次出现 `Ignore caches that are heterogeneous`，未伴随 GL 错误；样图检测仍打印 OpenCV 5 的既有 target 提示 | 按上下文、真实图元数和截图判断渲染是否成功，不能把所有 warning 当成同一个故障 |

最终桌面按钮的集成检查另存 [desktop-integration.json](../reports/desktop-integration.json)，已通过六种效果、同图验证、StarGAN、本机 GL 全部面、35° 绘制回执、公开样图截图和子进程正常退出。执行入口是 `desktop.py --self-test-no-camera`，本轮没有重开摄像头，因此不会覆盖下面两轮真实相机结果。

官方元数据初查脚本还曾把带 `dev1` 的预发布版本字符串转换为整数排序，产生 `ValueError`；随后直接读取 `2.1.16` 的版本特定 PyPI JSON，保留确切稳定版本，不再依赖该排序。完整 BSD 许可证已保存在 `vision3d/licenses/pyglet-2.1.16-LICENSE.txt`。

摄像头首轮保存为 [desktop-camera-first.json](../reports/desktop-camera-first.json)：20.0601 秒、149 帧、7.4277 FPS，149 帧均检出人脸。第二轮保存为 [desktop-camera.json](../reports/desktop-camera.json)：20.0267 秒、316 帧、15.7790 FPS，但只有 49 帧检出人脸，267 帧未检出。第二轮检出脸组的延迟中位数/P95 为 161.4576/198.4104 ms；未检出组为 40.5979/61.0161 ms。这些分层已由独立分析脚本复核，详见[桌面教程](desktop-guide.md)。

两轮背景负载和画面中的检测条件不同，没有受控 A/B 实验；第二轮总体 FPS 较高不能写成算法优化已经带来提速。分层依据是检测器输出，不是逐帧人工标注；“未检出”不等于确认画面无人，检出帧比例也不是检测准确率。

### 后续集成与交付过程中发现的真实问题

| 问题 | 根因与修复 | 验证及保留记录 |
|---|---|---|
| 演示录制期间读取 PNG 报 `PIL.UnidentifiedImageError` | 父进程读 `current-opengl.png` 时，GL 子进程正在向同一路径写入。改为至多一个 capture 在途：收到 `captured` 回执后先完整读入内存缓存，再发下一次写请求；等待期间使用已完成的缓存图 | 失败报告保留在 `reports/desktop-video-attempt1.json`。修复后从头复录成功，`reports/desktop-video.json` 的 `error=null`，最终216帧、5 FPS 播放网格、43.2秒播放时长；含初始化和解码检查的墙上时间为 45.4448秒。只录公开样图，未打开摄像头 |
| 视频播放帧率容易被误读为模型运行速度 | 演示视频按 5 FPS 时间网格组织，落后时复用上一帧；播放帧数/时长不等于每秒完成新的模型推理次数 | 把录制视频作为功能展示，实际摄像头吞吐量仍引用两轮独立报告；不把视频的固定帧率当作实时性能证据 |
| 摄像头错误可能使计时按钮不恢复，或让队列中的旧帧覆盖错误状态 | 错误事件触发测量中断、清除摄像头活动状态、递增任务编号并恢复按钮；旧编号结果不再提交。断流发生在 20 秒之后也仍标记 interrupted | `tests/test_desktop.py` 使用模拟断流覆盖上述边界；本轮结合既有功能回归共 5 项测试通过。模拟失败不能替代真人相机性能测量 |
| 输入法在项目内意外生成 `%SystemDrive%/ProgramData/SogouInput/...` 两个 `.bin` 文件，早期桌面副本包含了它们 | 交付过滤原先没有排除这个异常目录。加入精确 Git/Docker 忽略规则；根据旧复制清单的 SHA256 确认桌面文件确为本轮复制后，将两份桌面副本可逆移到 `work/excluded-desktop-copy` | 原项目中的输入法文件保留未改，两个文件从未推到 GitHub；重新生成桌面交付清单。此项是源码交付过滤问题，不能把输入法缓存算作项目源码，也不因此清理其他应用的数据 |

PNG 并发读写这一故障也说明“文件路径存在”并不等于“文件已经写完”。与其用固定睡眠猜测完成时间，进程之间应交换完成回执，并在复用同一路径前结束本次读取；这与模型是否预测正确是两个层面的问题。


## 续补：真实训练、评估协议与交付过程

以下条目来自实际日志、源码修复和复跑报告。失败原因尚未被单因素验证时，只报告观察与处理，避免把猜测写成唯一根因。

| 实际问题 | 判断与处理 | 复核与结果边界 |
|---|---|---|
| StarGAN 初始推理出现棋盘与偏色 | 作者测试路径采用逐图InstanceNorm；直接`eval()`使用checkpoint中过时运行统计，改变归一化。严格加载权重后禁用运行统计缓冲，并使用逐图统计 | 修复后的作者预训练与微调模型在同一规则下生成；错误图格仅留本地历史，未进入FID/IS；回归检查覆盖两种损失归一化和InstanceNorm |
| 教学损失与作者的尺度不同 | 早期BCE默认平均与patch均值梯度惩罚，不能悄悄混作官方协议；后续改为BCE求和除batch与patch求和GP | 最终模型前200个D更新属于旧尺度，后1000属于新尺度；配置和训练曲线明确标出变化点，不把全过程伪装为一个不变配置 |
| StarGAN 训练被中断，日志步数大于checkpoint步数 | 已记录650步但最后保存500；只恢复已保存G/D和两份Adam状态，最后阶段seed43重新排列数据 | 至少150次有日志更新未继承，未记录尾部未知；最终1200D/240G，配置池4096不等于已访问4096张不同图 |
| 初次生成质量前后比较软件构建不一致 | before用CPU生成/+cpu构建，after用XPU生成/+xpu构建，混合比较有混杂变量 | 保留CPU历史报告；重新用同XPU构建生成before、同CPU指标后端评分。主比较FID41.272788→40.980400，IS3.088072→3.098037；代码加入设备/构建一致性检查 |
| IS分块标准差和小样本FID容易被误读 | IS的10分块标准差不是两模型差异的置信区间；768样本FID具有有限样本偏差 | 报告样本名单、特征器、seed、分块数、输入尺寸；不写显著改善，不把图片无交集当作身份无交集 |
| ONNX导出提示InstanceNorm的train=True | 当前有意使用逐图统计；该算子语义不表示在桌面推理时反向传播或更新权重 | checker、真实图PyTorch/ORT allclose通过，最大差1.97e-6；解释告警含义，未仅因告警忽略数值检查 |
| COFW官方包无法完整传输 | 彩色流中断，正常Range返回HTTP500；灰度仅293760/178022021字节且MD5不符，普通续传又SSL EOF；浏览器下载域显示ERR_BLOCKED_BY_CLIENT | 不关闭浏览器保护、不采用残缺包；保留每次来源/字节数/校验报告。COFW为29点，现有68点入口也不能不改结构就套用；真实NME仍未完成 |
| WIDER首次全量配置loss为NaN | lr0.0025、无warmup，在100步附近发现非有限损失；标注坐标检查有效，不足以把全部原因唯一归因学习率 | 在独立目录重跑lr0.0005、250步warmup、grad_norm10及有限性保护，100步checkpoint；保留失败日志，最终完整结果单独验收 |
| MMEngine按epoch恢复可能重放前缀 | 中途checkpoint仍处于同一epoch，默认加载器可能从该轮开头读取 | 恢复采样器在图像解码前跳过已完成batch，独立测试索引不重放；未保存全部随机增强状态，不声称逐位恢复。当前完整训练进程未因修复重启 |
| 线程微测读checkpoint的配置键失败 | 脚本初期只查meta.cfg，实际checkpoint使用meta.config | 支持两个已知键后重测；监督器finally恢复同一训练容器。4/6/8线程中位0.7504/0.7622/0.6769秒，只约1.109倍；保留4线程长期训练，微测不含整条I/O路径 |
| MMEngine data_time出现不合理负数 | 该字段受计时实现和墙钟变化影响，不能直接按它分解计算/读取占比 | 对30秒容器CPU与I/O增量做独立观察；训练进度用perf_counter，不根据负data_time得出I/O瓶颈比例 |
| 长时间合盖后训练进度不动 | System Kernel-Power ID506在14:37:21.632Z记录Reason:Lid，ID507在16:07:55.544Z记录退出新型待机，约90分33.912秒 | 同容器恢复继续，未重启；内部perf_counter与日历经过时间分开。一次性SYSTEM_REQUIRED只防闲置休眠，结束释放，尊重合盖/电源按钮，不改全局电源计划 |
| 完整训练代码存在两次验证 | 当前运行版本runner.train已含末轮验证，随后runner.val再次执行 | 如实说明training_seconds包含第一次验证、validation_seconds记录第二次；修正后续入口val_begin避免重复，不把正在运行的旧版本输出冒充新版本路径 |
| 用户重命名GitHub仓库后旧地址查询失败 | 旧repo全名不再被当前连接器接受 | 查询规范仓库地址，更新origin、教程及源码汇编；后续使用完整新名称、非强制更新main并读取远端SHA验证 |
| 更新辅助服务时停止进程操作被自动审批拒绝 | 原命令没有执行；不得用另一种写法重复同一被拒停止动作 | 保留旧服务，在新回环端口8767启动已授权的新版本并通过HTTP检查；最终默认启动原生程序。没有改变系统防护或强杀其他进程 |

最终GitHub提交、源代码压缩包、桌面副本与文档校验以交付清单记录的同一版本为准。所有示例界面截图只含公开NASA样图；摄像头报告只有计时、帧数和状态，不含图像。版本锁定文件记录本机实际组合，而不是建议新电脑同时把所有版本装入一个环境。


## 最终定向审查中的四项修复

1. 单帧相机推理异常曾落到线程最外层，使工作线程永久退出。现对每帧读取/推理局部捕获、释放摄像头并发普通error，保持线程接收后续图片与重开命令；真正初始化fatal明确提示关闭重开且拒绝继续排队。假相机触发一次推理异常后，下一张图片实际被处理，回归通过。
2. 双图验证先改camera_active、后读文件，坏图可让界面说停止而设备仍采集。现先完整读取校验两图，再停止测量与切换任务；坏图回归确认相机状态与队列均不被改变。包含此前旧帧覆盖错误的检查，主测试集共7项通过。
3. 录像按时间推进可能在模型缺失或处理缓慢时跳过结果。现每场景检查实际已显示效果、验证结果、GAN结果或GL绘制/旋转回执；完成后才计展示时长，45秒未完成则失败。最终视频216帧/43.2秒，11场景都有completed_frame，all_scenes_acknowledged=true，逐帧解码计数一致。
4. 编码器close抛错可能跳过应用清理。现finish幂等，try/finally保证app.close，用户关闭窗口也走同一收尾；正常录制已验证摄像头未打开、worker与子进程释放，编码失败分支的保证来自明确finally控制流而非虚构故障实测。
