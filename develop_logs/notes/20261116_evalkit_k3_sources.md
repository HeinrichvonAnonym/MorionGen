# K3 来源定义核查

本阶段回答“源端已经定义什么，以及 evalkit 增加了什么假设”。八种格式的旋转读取、父子层级和世界坐标转换已有来源证据；头胸模型轴对应人体解剖轴的解释需要另列。此次没有通过重新估计轴来降低受控误差。

| 来源 | 已核查的来源约定 | 项目解释与剩余证据 |
| --- | --- | --- |
| MotionLCM / OmniControl | HumanML 解码中 Y 为高度，根旋转绕 Y；内部四元数为 wxyz；位置解码保留根 XZ 轨迹 | D1/D2 只传位置及其派生特征，终端头朝向仍缺独立观测 |
| ARDY | FK 返回源骨架的世界旋转与关节位置，中立节点和 parents 来自资产 | 中立模型头胸轴作为段约定，尚未独立验证人体解剖零位 |
| CHOIS | body22 世界矩阵，外加手点，数据处理使用 Z 为地面高度 | 已知序列复用体型资产；未知序列的静止骨架回推属于输入估计 |
| CLoSD | rigid-body xyzw 世界旋转；上游明确使用 modified upright-start SMPL | upright-start 固定基底转换与人体解剖轴对应分别记录 |
| DART | root 世界矩阵加 21 个局部矩阵，按父子关系组合 | 源位置和旋转 FK 的差异仍单列，源模型中立轴不自动获得外部解剖精度 |
| Kimodo | 输出文件明确提供 local/global matrices，位置来自骨架 FK，单位米；原生 Y-up | 77 节点按资产命名，模板末端不视作独立观测 |
| PRIMAL | translation、root/body axis-angle，原生参数与公共格式分别读取 | 静止模型与解剖轴对应保留假设；受控导出前两帧是历史上下文 |
| Andy | CSV 明确秒和毫米，原始标记名称齐备 | Z-up 由原有导入与几何核查支持，CSV 未逐文件声明世界轴；头胸四点框架由项目定义 |
| Han | B3D 的 markerObservations 与处理后的关节结果分开读取 | 保持容器坐标；头胸标记框架是项目几何参考，不等同于发布的解剖关节角 |
| CoLLaboratE | 每文件 CHANNELS 决定旋转顺序与平移，保留非根平移 | cm、Y-up 依赖几何推断；发布页未声明段解剖轴。当前共用胸段静止基底属于项目假设 |

全量来源文件、行号、哈希和字段分类保存在 [conventions.json](/home/heinrich/MorionGen/config/evalkit_k3_20261116/conventions.json)。位置、旋转、角度及生物力学解释的来源分开记录，不能仅凭自洽的格式往返把最后一项确认为已知。

CoLLaboratE 发布页确认 BVH 格式及站立、坐姿、跪姿等任务定义；本次只用开发子集，未读取电视留出运动。[官方发布页](https://zenodo.org/records/6472155)

Han 的原始标记与模型估计结果分离遵循 B3D 的输入／处理阶段结构。[Nimble 官方说明](https://nimblephysics.org/docs/working-with-addbiomechanics-data.html)

已保存 K2 独立检出与冻结文件哈希。HYOID 及旧配置、留出使用记录保持原值。全量几何核查和八路拟合另附结果，不把本表的来源审计作为精度验收。
