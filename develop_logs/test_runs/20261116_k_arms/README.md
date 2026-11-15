# K2 手臂开发实验索引

最终方法及入口：`utils/motiogen/evalkit/k/arms.py`、`fitting.py`、`python/scripts/evalkit_k_arm_audit.py`。生产配置为 `config/evalkit_k_v2.json`；汇总及审计在 `config/evalkit_k_arms_20261116/`，阶段报告在 `develop_logs/notes/20261116_evalkit_k_arm_alignment.md`。

数据盘输出根：`/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k_arms_20261116`。

- `development`：12 条开发记录，每条三格式；按 LAB 人员及两个工业子集的 ID 哈希选择，中间五秒。
- `development_first`：10 位 LAB 参与者各自按文件名排序的首条登记记录，起始五秒；补查直立标定覆盖。
- 两组中的 `k1` 对照由数据盘 `frozen_k1` 独立检出（`acb48ac`）执行。`k2_adaptive` 是最终开发版本，完整核对当前实现哈希。其他 `k2*` 文件夹是被替代的开发迭代；`k2_stable` 部分运行主动终止，仅作历史保留。
- `regression_k2_adaptive`：八份实际输出及下游接口回归；含一次 11 帧下肢肌肉求解。实际样例沿用此前开发回归资料。
- `arm_metric_cache`：旧版本的全 100 Hz 源几何统计缓存，键包含输入、模型、坐标及几何实现哈希；原冻结输出只读。

归因实验在修改 K 之前执行，读取提交 `acb48ac`/`2edc628` 的 K1 实现：

- `diagnose.py`：分解原始/滤波坐标、关节中心/注册标记角度、关节范围及源骨长波动。
- `ablate.py`：仅扩展肩范围、仅保留手臂关节中心、两者联合；每项使用独立进程，最终有效结果位于 `ablation_v2`。最初未隔离进程状态的试跑中止并被替代。
- `directions_multistart.py`：固定身体后的骨方向优化探索；未进入生产分支，证据保存在 `direction_experiment.json`。其模型输入来自上述 `ablation_v2/both`。

上述归因脚本在 K1 检出的目录作为 cwd 运行，使用现有 `/home/heinrich/.local/share/opensim-api/bin/python`。当前实现的复现使用正式 audit/report 脚本；`verify.py` 检查最终 74 份 K2 拟合、八份缓存、38 项测试日志、原冻结文件字节不变性及磁盘占用。完整日志保留在本目录，较大的模型与逐帧数据留在数据盘。

本轮没有读取并重评分 Andy/电视最终留出动作；保留其已使用历史。新统计仅来自开发记录及既有真实基线开发样例。
