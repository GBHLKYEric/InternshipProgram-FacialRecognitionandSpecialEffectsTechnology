# 执行问题、根因、处理与复核

记录日期：2026-09-26。下表区分“已修复并运行”“识别到但未完成”“设计限制”；不把未出现的问题写成本次发生的故障。报告中CPU结果、预训练模型结果、随机张量测试和真实数据训练分别标识。

| 实际遇到的问题 | 定位与根因 | 采取的处理 | 复核与剩余限制 |
|---|---|---|---|
| 受限Windows子进程中SSL/DNS、venv操作失败 | 进程环境缺少 `SystemRoot`、`WINDIR`、`COMSPEC` | 为相关子进程补齐必要变量；`start.ps1`保留同样防护，不改系统全局设置 | 隔离环境成功启动、依赖和模型下载可继续；版本见 `reports/environment.json` |
| Python解释器/依赖组合容易混用 | Codex内置Python、项目venv、WSL Docker属于不同环境 | 项目使用明确的 `.venv/Scripts/python.exe`；研究依赖单独列出；记录freeze | 环境报告与实际smoke报告对应；不能把容器Python版本写成桌面版本 |
| MMDetection和主环境最新版依赖不兼容 | 3.3.0有严格版本范围，MMCV还包含编译算子 | 单独Docker锁定Python3.10.21、torch2.1.0CPU、torchvision0.16CPU、NumPy1.26.4、mmcv2.1、mmengine0.10.7、mmdet3.3、OpenCV4.11 | NMS真实运行、WIDER小样本训练/验证/推理全链成功；不改主venv、不删除版本assert |
| Chrome浏览器自动化入口不可用 | 插件返回 `Browser is not available: chrome`；原生控制无法确定目标URL后安全停止 | 保留错误，继续本地HTTP/API与文件验证，不称Chrome交互已通过 | 接口/截图或其他浏览器证据应注明来源，摄像头真人流程需要可用控制连接 |
| 旧GitHub创建仓库接口限流 | `create_repo`入口失败 | 检索已连接GitHub，使用用户已存在公开仓库 `GBHLKYEric/face_ai_project` | 已发布38db093c375e6dcd2e129d9048f7857e564cd314；最新文档的收尾提交仍需最终远程核对 |
| Docker首次拉取镜像出现CloudFront EOF | 网络传输中断 | 通过正常重试重新拉取，不修改校验或使用未知镜像 | 已真实build与容器Hello World；WSL Docker 29.1.3、容器Python 3.12.14 |
| LFW原网站不可达 | 网页请求失败，并不表示数据不存在 | 查scikit-learn官方维护下载器，使用其Figshare镜像和SHA256 | 下载后运行完整6000对评估；没有用任意转载数据替代 |
| LFW首次预训练基线低，很多图检测出多个人脸 | 应用验证要求恰好一张脸；原LFW部分图片含多脸/检测多框，导致处理失败 | 保留首轮72.25%与全部失败；另加固定最大面积、并列中心最近的几何选脸策略，再跑完整6000对 | `reports/lfw-sface-largest.json`第二次98.9333%±0.38873个百分点，7701图全部成功；它是经过首轮错误分析后的第二次实验，不是自训ArcFace或完全未触碰测试集的独立最终认证 |
| 余弦同图得分略超过1 | 浮点计算舍入导致 `1.0000000488` | 显示/接口分数裁剪到数学范围[-1,1]；核对归一化 | 同图结果只是数值一致性检查，不称准确率100% |
| 动态INT8未加速 | ResNet大部分是Conv2d，当前动态量化仅覆盖Linear；转换开销与硬件有关 | 原样记录同一真实checkpoint的FP32/INT8/ORT，不写预设加速倍数 | batch8/2线程/10次，104.477ms→105.454ms；state只省约0.82%；完整LFW52.6667%→53.0167%，不能称有统计显著改善 |
| ONNX导出API随PyTorch变动 | 现代导出器与旧教程默认行为不同 | 当前代码显式 `dynamo=False`、opset17，并用checker与ORT对比；保留弃用警告作为升级事项 | 真实checkpoint最大绝对差8.79e-7；ORT中位155.836ms，比同批FP32慢；数值接近不等于已单独验证全LFW准确率 |
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
| MS-Celeb-1M授权训练包未取得 | 不能从未知转载包推定拥有可使用数据，也不能把评估身份混入训练 | 另建明确标识的替代pilot：LFW非pairs身份52个/133图，训练与评估身份交集0 | 真实3epoch训练和曲线已保存；不改变PDF指定数据要求的未验收状态 |
| 小样本ArcFace训练loss下降，margin分类准确率仍0 | 随机初始化大型骨干、133图、3epoch与角度间隔共同使任务困难；不能仅由日志断定某一因素贡献 | 同时保存loss/准确率，执行完整6000pairs评估，不手写好看的准确率 | loss40.111934→36.125987；FP32 LFW52.6667%/INT8 53.0167%，均未达标；数据/模型/轮次需后续受控实验改进 |
| README引用不存在的独立模块/下载器 | 把三维脚本内部函数误写成模块，数据下载命令没有纳入仓库 | 删除 `python -m vision3d.export_onnx`，改用实际reconstruct入口；LFW按官方来源/SHA256说明布局，明确没有fetch_lfw.py | 用实际文件清单与CLI逐项校准；不把“若存在”留作可执行教程命令 |
| Windows非法Origin请求触发客户端10053 | 服务在未读取请求体时立即返回403，连接关闭可能造成客户端接收失败 | 先在22MB请求体上限内读取，再按Origin规则返回403；仍保持来源限制 | `scripts/verify_local.py`实测错误路径通过，没有为了消除错误放宽跨源访问 |
| 已存在venv但缺依赖时启动失败 | 旧启动逻辑仅在新建venv后安装依赖 | `start.ps1`改为每次执行幂等依赖安装，已有符合要求的包会复用 | 初次创建和已有环境两种状态遵循同一依赖清单 |
| 新clone缺少本地素材/权重会影响Docker构建和运行 | 原工作目录存在下载文件，可能掩盖干净副本的依赖 | Docker构建阶段按来源/哈希下载，构建上下文排除本机数据及已有权重 | lab镜像已真实构建运行；容器中faces=1、组合特效、同图cosine=1通过，宿主8766健康检查200；镜像只含核心应用，三维另生成 |
| 网页教程里的相对Markdown链接返回404 | 文档链接按文件布局编写，服务没有对应路由 | 生成HTML时重写链接，服务仅允许白名单docs文件路由 | 本地教程可通过服务浏览；不暴露任意文件路径 |
| 摄像头授权异步返回可能覆盖新状态 | getUserMedia在用户切换/停止后才返回，旧结果可能重新绑定视频 | 使用generation token识别过期请求，申请期间禁用按钮并释放过期流 | 代码处理竞态；真人摄像头授权与连续跟踪仍未手工实测 |
| WSL默认用户没有Docker socket权限 | 该用户不在所需访问上下文中 | 本次Docker命令显式以WSL root运行 | 没有放宽socket权限或修改全局用户组；Hello构建/运行证据已保存 |
| Docker映射8766:8765后浏览器得到403 | 程序监听8765，HTTP Host是外部8766，已有Host端口白名单拒绝不匹配值 | 改为 `-p 127.0.0.1:8766:8766` 并显式 `python app.py --host 0.0.0.0 --port 8766` | 宿主健康检查200；默认8765:8765即可，无须削弱Host校验 |

## 已有项目的审计与本次工程关系

最终交付检查还发现Windows生成的CRLF与远端历史文本混用，使Git将行尾CR识别为尾随空白。项目增加 `.gitattributes` 固定文本LF，并在发布时统一换行；源码汇编在统一后的源码上计算哈希。视频原始输出为OpenCV的MPEG-4 Part 2，为提高常见浏览器/播放器兼容性，另以FFmpeg导出H.264/yuv420p并逐帧解码核对360帧、20 FPS、18秒；这属于交付格式处理，不改变原始模型计时。LibreOffice转换PPT时出现Python prefix提示及终端中文乱码，实际PDF文件存在、16页可渲染；以文件读取和逐页视觉检查为准。

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

完整CelebA图像与属性训练、WIDER全量正式训练（小样本已完成）、300-W或COFW训练测试数据、可核验MS-Celeb-1M授权数据、真人摄像头授权与连续输入、真实移动设备，以及云服务凭据。MMCV兼容环境已在独立Docker中解除；三维页面已有实际浏览器渲染证据，但原Chrome插件连接错误与真人摄像头实测分别保留。源码与教程已给出推进路径；缺少外部条件时，不填补虚构数据、不购买未知服务、不伪造项目通过。
