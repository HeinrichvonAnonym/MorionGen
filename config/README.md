# A B E H 基线实验入口

本轮把代码完整性、资产齐备、推理成功和训练成功分开记录。入口默认不会训练。完整模型尚未通过资源审核时，`demo` 会留下阻塞记录；不能用 VAE 重建或仓库自带的生成样例替代受控生成结果。

## 启动命令

从仓库根目录运行。调度与五个基线环境都位于 `${MORIONGEN_DATA_ROOT}/envs/`；默认值来自 `baselines.json`，可用同名环境变量覆盖。环境尚未全部验收，实际状态见阶段报告。

```bash
bash scripts/run_baseline.sh audit --baseline all
bash scripts/run_baseline.sh demo --baseline motionlcm --scenario reach --dry-run
bash scripts/run_baseline.sh prepare --baseline motionlcm --prepare-part environment
bash scripts/run_baseline.sh prepare --baseline kimodo --asset model --asset tmr --asset benchmark-metadata
bash scripts/run_baseline.sh prepare --baseline ardy --asset local-text
bash scripts/run_baseline.sh demo --baseline motionlcm --scenario reconstruction --benchmark
bash scripts/run_baseline.sh demo --baseline ardy --scenario online
bash scripts/run_baseline.sh demo --baseline kimodo --scenario orientation
bash scripts/run_baseline.sh demo --baseline chois --scenario carry
bash scripts/run_baseline.sh evaluate --baseline motionlcm --scenario reach --input /absolute/path/motion.npz
bash scripts/run_baseline.sh train-smoke --baseline motionlcm --scenario reach --seed 0
```

`demo` 和 `evaluate-native --profile smoke` 不指定 seed 时运行 0、1、2；`--benchmark` 分别进行冷运行、三次预热和二十次测量，保存原始延迟样本。`--variant cached` 对 MotionLCM/OmniControl 复用同进程文本 embedding；ARDY/Kimodo 则在加载运动模型前一次性编码场景内有限 prompt 并释放 Llama，不能把两者混称为同一种性能优化。FP16/运动模型 offload 尚未审核，入口明确返回不支持。ARDY 在线场景的二十次基准测量流程尚未验证，现阶段只记录逐步重规划事件，不宣称完成该基准。

## 指标与公共格式

`scenarios/` 分别定义 reach、orientation、online、carry、reach_low 和 reach_high；`reconstruction` 只检查官方 VAE 重建链路。A 用指定帧手部位置和真实旋转；B 用 chunk 边界、对齐的历史预测和更新事件；E 用躯干、上臂和肘部姿态代理；H 分开计时与资源占用。不同场景和骨架不合并为总分。

公共 NPZ 必须含 `joints[T,J,3]`、`joint_names[J]`、`fps` 和 `metadata_json`。单位是米，右手 Z-up，Y-up 转换为 `(x,-z,y)`；旋转字段是 `[T,J,3,3]` 的真实全局 SO(3) 矩阵。骨架只改名称和世界基底，不做重定向。原始数据另存 `raw/`。接触标签、物体状态、表面距离、片段边界和历史误差为可选字段，不能从缺失输入补零。

位置误差和约束成功率的分母是指定位置约束数；旋转误差为测地角，分母是具有真实旋转输出的朝向约束数。jerk 在原生 `dt=1/FPS` 上计算三阶差分，报告 XYZ 分量 RMS，分母为 `3*(T-3)*J`，包含拼接边界；它与三维向量范数 RMS 相差 `sqrt(3)`。姿态风险比例按有效帧计，ROM 当前只实现肘部几何代理。接触 F1 另存 TP、FP、FN 和 `2TP+FP+FN`；无正例时为不可计算。更新恢复时间只汇总已恢复更新，并另报恢复比例及逐次观测窗口，未恢复样本作删失记录。

阈值见 `metrics.json`，它们是工程初始约定，未经任务统计推导，不是临床阈值。完整 RULA/REBA、关节负荷、CoP 和肌肉用力未实现。物体穿透只接受真实有符号表面距离；关节点距离不能冒充几何穿透。

`evaluation.json` 和 `--official-protocol` 对接上游统计函数，接收已经由相应官方评估器导出的 embedding；当前没有自动提取官方 embedding 的完整流水线。协议必须带评估器、归一化和 embedding 文件校验值、数据集、split、骨架与 FPS。当前协议要求至少 320 个真实参考和生成样本、32 项检索 batch、300 个 diversity 配对，这是本轮接口约定，不是通用的 FID 最低样本定理。三个 demo 不计算分布质量。不同评估器的结果不得直接横比。

## 日志与资源门槛

每次运行在 `develop_logs/test_runs/` 新建 UTC+8 时间戳与 UUID 目录，保存配置、口径快照、源文件校验值、Git SHA/修改状态、资产 receipt、命令、环境、stdout/stderr、资源采样、阶段事件、指标、NPZ、PNG/MP4。预检拒绝也有 manifest 和空日志文件，重跑不会覆盖。失败、超时、资源终止、缺资源、不支持和资源审核阻塞是不同状态。

`${MORIONGEN_DATA_ROOT}/envs/motiogen-tools/bin/python python/scripts/summarize_runs.py` 按相同实现/config/scenario/variant 分组汇总超时率与资源终止率；分母是实际启动的 worker 次数，预检拒绝另外计数，没有启动时比率记 `null`，不形成总分排名。

GPU 工作使用文件锁串行；默认 batch=1、worker=0。预算为 `min(20 GiB, GPU free−3 GiB)` 和 `min(12 GiB, MemAvailable−6 GiB)`，磁盘保留 80 GiB，worker 默认超时 900 秒。监测子进程树 RSS、全机余量和 GPU；设备总显存采样与 PyTorch allocator 峰值分别报告。当前未启用硬内存隔离，watchdog 可能存在采样间隙。

`inference_evidence.json` 必须先绑定资源估计依据、工作内存上界和源文件/权重 SHA。资产下载完成后仍要通过该门槛，不能把“文件存在”当作推理成功。训练另外需要 `training_evidence.json`；不能因为推理通过就启动反向传播。本轮不运行训练。

## 下载与本地原始 Llama

2026-10-06 用户已明确改为使用当前代理下载全部资源，不使用镜像。`downloads.json` 使用官方 Hugging Face、Google Drive 和 OpenAI CLIP 来源，允许官方 HTTPS 存储重定向，并继承当前代理；不记录代理凭据或临时签名 URL。统一启动：`bash scripts/download_baseline_assets.sh --jobs 3`。逐文件进度保存为资产旁的 `*.download.json`，各基线仍保存独立 prepare 日志。

官方 HF 元数据最大 2 MiB，用于获取 revision 和 LFS SHA-256；模型文件必须校验官方大小和 SHA-256。Google Drive 首先使用可续传传输，确认页不兼容时回退到外部 `motiogen-tools` 环境中的 cookie-aware `gdown` 公共流程；完成后记录本地 SHA-256 并校验 ZIP 解压。未公开上游哈希时明确区分本地完整性记录与上游身份校验。下载保持 80 GiB 磁盘余量。本地原始 Llama 8B 只接受用户提供的绝对路径；SMPL-X 只读复用配置的许可文件。环境安装使用官方 PyPI、PyTorch 与 Anaconda 源并继承当前代理。

`language_models.json` 指向用户提供的原始 HF checkpoint，排除 `fine_tuned_hf`、`merged`、`work_dirs`，并验证 Meta-Llama-3-8B-Instruct 的配置指纹；Llama 3.1 会明确拒绝。源权重只读，`original_base` 中使用 symlink；只复制并修补 `_name_or_path` 保持官方指令格式。receipt 的本地 SHA 不能证明与未核对的上游原始 shard 完全一致。两组官方 LLM2Vec adapter 仍需齐备；不替换编码器。编码阶段保持 bfloat16，限定 11 GiB GPU 与 10 GiB CPU dispatch，场景 prompt 编码后释放 8B 权重；该路径仍需用真实 Llama 3.0 完成首次资源验收。

外部 Python 源在 `python/third_party/`，C++ 源在 `third_party/`，版本见 `dependency_sources.json`。NVIDIA staging 将官方 C++ 编译并行从 `-j4` 改为 `-j1` 并保存补丁依据；baseline 工作树保持原样。成功安装环境才写 `environments/locks/`，部分环境冻结清单保存在运行日志，不能当成功锁文件。

## 当前最先推进的实验

先完成五个环境和官方资产校验；用户提供 Llama 绝对路径后注册只读 receipt 并下载两组 LLM2Vec adapter。随后按 `evaluate-native --profile smoke` 逐个运行真实 batch=1 推理；`full` 只做完整协议前检。CHOIS 使用专用单样本入口，不创建 optimizer、GradScaler、训练集或 EMA 副本。
