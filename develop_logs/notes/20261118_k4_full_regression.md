# K4：656 路固定开发回归

已评分 10/656；通过 10/656；执行失败 0；待完成 646。

本轮使用同一冻结实现与候选配置，以原动作 R 验收。拟合与选择只读取 D；降级损失、拟合偏差和端到端误差分别保存。逐窗口通过同时要求原生 FK、实际导出、原动作时钟、有效覆盖和时序检查通过。

| 格式 | 通过／82 | 已评分 | 执行失败 | 待完成 |
|---|---:|---:|---:|---:|
| ardy | 1/82 | 1 | 0 | 81 |
| chois | 3/82 | 3 | 0 | 79 |
| closd | 1/82 | 1 | 0 | 81 |
| dart | 0/82 | 0 | 0 | 82 |
| kimodo | 5/82 | 5 | 0 | 77 |
| motionlcm | 0/82 | 0 | 0 | 82 |
| omnicontrol | 0/82 | 0 | 0 | 82 |
| primal | 0/82 | 0 | 0 | 82 |

当前结果用于确定冻结版本的适用范围；默认配置保持原值。失败窗口及逐指标原因全部保留。最终留出新增使用为 0。

[离线逐路报告](/home/heinrich/MorionGen/docs/20261118_k4_full_regression.html) · [完整结果 JSON](/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k4_20261117/full_regression_20261118_v1/results_656.json) · [运行协议](/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k4_20261117/full_regression_20261118_v1/protocol.json)
