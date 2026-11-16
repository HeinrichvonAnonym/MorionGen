# Evalkit K：运动读取、姿态、可测性与版本

内部统一为米、右手 Y-up，保留输入秒时钟。Andy 的 Z-up 导入约定通过正交旋转转换；头胸标记框架由项目按右侧、上方、后方三轴构建，前屈在此约定下为负 X 旋转，人体解剖对应另记证据。`load(source, method)` 返回 `Motion`，`evaluate(motion)` 返回头、胸、颈以及位置支持的躯干、膝、肘、大腿和颈头连线指标。完整三维旋转与一条骨方向分别表达；颈头连线描述单轴方向，头部独立轴向旋转需要额外观测。

默认配置为 `config/evalkit_k_v2.json`；K1 的八格式定义保留在 `config/evalkit_k_v1.json`，K3 由 `config/evalkit_k_v3.json` 显式选择。模型中立几何及历史来源哈希在 `config/evalkit_k_calibration/`。读取器按实际字段识别可用旋转：CLoSD 旧记录缺少刚体四元数、DART 旧记录缺少参数时保留位置指标。PRIMAL 原生 Y-up 与公共 Z-up 文件分别处理。

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

## K2 手臂开发版本

当前入口读取 `config/evalkit_k_v2.json`。K1 的实现保存在提交 `acb48ac`，原配置、冻结清单和最终留出使用记录保持不变。K2 只使用开发资料；运行 K1 的最终验证应检出原提交及原配置。

MotionLCM、OmniControl、Kimodo 比较关节中心/注册标记及肩屈曲原范围/扩展范围的候选组合，按源几何保真度选择；其他格式保持原手臂拟合策略。该范围用于覆盖运动学表示，肌肉模型在新增范围内的力学准确性需另行验证。`arms.py` 同时检查最终滤波坐标的两条骨方向、肘角及肩肘腕位置。`work/arm_quality.npz` 分别保存角度、位置掩码；10° 与 0.1 m 是明确配置的工程保真门，前者沿用误差档，后者沿用 IK 位置门。它们描述对输入的保真，解剖精度仍由外部参考限定。

`fit()` 的 `arm_valid` 再与全身标定状态相交；CLI 的 `fit_coordinates.npz`、evalkit 的 `frames.npz` 保存左右手臂掩码。工效评分在失效侧把对应手臂角度保留为未知，并计算评分区间。位置型头部状态沿用 K1。

FBLS/HYOID 的资源位置统一由 `motiogen.model_resources` 读取 `config/biomechanics_models.json`。迁移数据盘时可设置 `MORIONGEN_DATA_ROOT`，既有 upper_body 与 evalkit 共用同一路径。

扩展肩屈曲范围时，先在原 ±90° 范围内求首个观测帧，再用该解初始化扩展范围的 IK，以稳定直臂/外展姿态的欧拉角分支。校准和正式 IK 都执行该步骤；用于足部标定的交付模型默认姿态保持原值。标定候选先经过绝对姿态条件筛选，再在合格池内补足最低时长。

三个目标格式同时计算关节中心原范围、关节中心扩展范围、注册标记原范围候选，以最终滤波坐标中角度和位置共同合格的帧比例择优，再以手臂位置 RMSE 区分。候选失败单独保存，有效候选可继续；全部失败时返回拟合不可用。`receipt.arm_selection` 给出规则、候选分数及所选模型，`arm_candidates/` 保留各候选路径。该选择只使用当前输入几何，全身标定有效性仍独立判断。

原范围候选保留原 IK 初始化；仅扩展范围候选使用首帧连续初始化。注册标记候选用于保留某些直臂姿态的稳定解，同样接受最终源几何检验。全部手臂统计基于 100 Hz 最终坐标，绘图可另行抽样。

## K3：显式约定、两种骨长视图与全身方向拟合

`load(..., config=...)`、`evaluate(..., config=...)` 与 `k.fitting.fit(..., config=...)` 使用独立配置。CLI 加 `--config config/evalkit_k_v3.json`；统一 evalkit 的 `evaluate()` 传 `k_config`，批量 manifest 每行可指定 `k_config`。无显式配置的既有入口继续使用 K2。CLI 把已按 `--method` 解码的 `Motion` 传给拟合器，通用旧文件名也能保留所选格式。

| 文件 | 职责 |
| --- | --- |
| `motion.py`、`calibration.py` | 原生字段、确定性轴转换、模型资产及估计来源 |
| `reference_bvh.py` | CoLLaboratE 参考的平移通道语义；独立于拟合器 |
| `bones.py` | 公共评价骨架；输入骨长投影与评分骨长归一 |
| `fitting3.py`、`refinement.py` | 拟合视图、全身方向优化、逐部位支持和缓存 |
| `scoring.py` | 原始参考下的 D−R、F−D、F−R；投影 P−D 与 F−P |

CoLLaboratE 的动画平移通道直接提供对应轴的局部位置；未动画的轴采用 OFFSET。K3 显式使用 `absolute_channels` 和双精度 FK。共享 BVH 读取器保留原 `offset_plus_channel` 默认行为，使旧结果可在原检出复现。米制比例及 Y-up 的依据、共用胸段静止基底等项目假设记录在 K3 的来源证据表中。

`Motion.positions` 保留原始输入。拟合前投影仅使用当前输入逐段长度中位数；头胸旋转继续取原始字段。参考骨长只进入评分器调用的 `bones.normalize`，拟合器的接口不接收参考。结果保留 `source`、`projected`、`fitted` 三份几何；独立的评分产物再保存 R、D、P、F 的统一骨长视图。公共根定义为双髋中点，另报去根平移的位置差异。

方向优化以输入骨段方向为主，保留较弱的位置、连续性和初值项，20 Hz 求解后插值到 100 Hz；最终坐标经 OpenSim FK 评分。距下关节等模型既有锁定继续保留。轴向转动和解剖关节角的解释仍取决于可观测性及段定义。

`region_angle_valid` 表示对输入方向的保真；`region_registered_valid` 还要求合格直立标定；`region_position_valid` 检查配准位置。原始参考精度由验证中的 F−R 单独确定。下游力学支持取配准角度与位置条件的交集，下肢肌肉主结果再与残差条件相交，强制全帧结果单列。

CHOIS 的 body wrist 与额外 hand point 分开保留。`Motion.legacy_aliases()` 同时用于拟合节点名和 `legacy_prior_joints()` 的观测标记；额外手点为先验时，已观测的 body wrist 继续支持手臂几何。对应 body wrist 本身为先验时，该侧手臂支持仍为 false。冻结拟合的轨迹保留，新的观测标记与原始窗口骨长评分由 `evalkit_k3_rescore.py` 单独生成，并检查未经掩码筛选的原始误差完全一致。

缓存同时记录输入文件与内存视图、来源约定、投影策略、拟合参数、模型和实现哈希。四组消融固定读取 `config/evalkit_k3_20261116/experiment_config.json`，最终运行配置中的逐格式策略不会覆盖消融分组。开发名单、冻结检出及已使用留出的历史均独立保存；HYOID 保持原状。
