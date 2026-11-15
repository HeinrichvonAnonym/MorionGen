# K v1：运动读取、姿态与可测性

内部统一为米、右手 Y-up，保留输入秒时钟。Andy 原始 Z-up 通过正交旋转转换；头胸段统一采用解剖右侧、上方、后方三轴，前屈为该坐标下负 X 旋转。`load(source, method)` 返回 `Motion`，`evaluate(motion)` 返回头、胸、颈以及位置支持的躯干、膝、肘、大腿和颈头连线指标。完整三维旋转与一条骨方向分别表达；颈头连线不能提供头部的独立轴向旋转。

八个格式入口和帧率在 `config/evalkit_k_v1.json`；模型中立几何及历史来源哈希在 `config/evalkit_k_calibration/`。读取器按实际字段识别可用旋转：CLoSD 旧记录缺少刚体四元数、DART 旧记录缺少参数时保留位置指标。PRIMAL 原生 Y-up 与公共 Z-up 文件分别处理。

## 输出状态

- `supported_native_frame`：具有完整段坐标，或明确标注的几何段坐标。
- `native_frame_only`：头胸原生旋转可读，解剖段标定仍待提供。
- `model_convention`：通过模型中立几何的骨方向一致性检查；误差适用范围仍需独立资料验证。
- `supported`：调用者提供了明确的段标定矩阵；该状态说明接口条件成立，精度由相应验证报告描述。
- `unreliable`：颈部指标不可可靠评价；保留其他有效部位。

`anatomical_valid` 表示当前声明的标定假设下可计算，须同时读取 `precision_scope`。它与外部解剖精度验收是两项证据。模板节点由 `k_provenance.json` 标记为 prior，CLoSD 的物理刚体名称也映射到该掩码。位置型 D1/D2 忽略额外旋转字段。若世界坐标的浮点间隔超过 0.1 mm，位置帧被标记为未知；原生旋转独立保留。这是数值精度保护条件。

## 入口

在仓库根目录、现有 OpenSim 环境运行：

```bash
/home/heinrich/.local/share/opensim-api/bin/python python/scripts/evalkit_k.py \
  --source /absolute/path/motion.npz --method dart --out /data/path/k_result --fit
```

`k.npz` 保存原始时间、关节位置、段旋转和逐指标掩码；`k.json` 保存来源、配置和版本；`fit_coordinates.npz` 保存拟合坐标及有效掩码。拟合使用均匀时钟，输出从零开始，原始时钟保留在 K 文件。非均匀输入应先显式重采样。短缺测可加 `--predict-short-gaps`，输出额外的预测数组与预测掩码，观测有效性保持原状；默认只允许 0.1 秒、双向离线插值。

`fit.py` 是兼容入口，实现在 `k/fitting.py`。输入内容、模型、配置和实现哈希共同控制缓存。缺少合格直立标定时保留诊断拟合，并将标定有效性设为 false。`position_fit_valid` 与 `calibration_valid` 分开保存。FBLS 下游遇到该条件返回 `unreliable_kinematics`。

## 验证层次

`evalkit_k_validate.py` 读取已登记名单；`evalkit_k_summarize.py` 执行按人/原记录汇总及 25 个开发折；`evalkit_k_gap_validate.py` 在只接受头胸段观测的接口上，按登记 seed 遮蔽头、胸、头胸共同缺测。预测端接收遮蔽后的矩阵及时间，不接收原始标记、未遮蔽姿态参数或特征。

光学标记的四点框架与冗余模板比较属于重建一致性证据。Andy/Han 的八格式对照把实测头胸放在明确的模板身体上，身体其余部位为先验。CoLLaboratE 用发布 BVH 的完整层级与通道，提供全身表示对照。每条记录主参考检查覆盖全程；八格式转换选取首个至少 1 秒的完整连续窗，最多 10 秒。短记录和无完整窗单列。PRIMAL 的前两帧只作历史，不进入评分。

工程档位为 5°/10°/20°。报告同时列记录数、人数、覆盖、中位数、P95 和有符号旋转向量偏差。多个记录的 P95 均值或中位数在字段名中明确；均不冒称合并帧的 P95。Han 多段先汇总至原记录。

`evalkit_k_regression.py` 对八份真实输出运行全身拟合及外力/工效接口，对一个短窗运行肌肉求解。`evalkit_k_visualize.py` 和 `evalkit_k_gold_overlay.py` 输出开发叠加图。最终留出必须给出匹配的 `--freeze`；名单始终来自原登记文件，失败和无效记录仍进入分母。
