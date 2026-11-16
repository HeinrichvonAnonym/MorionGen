# K3 来源定义核查

本阶段回答“源端已经定义什么，以及 evalkit 增加了什么假设”。八种格式的旋转读取、父子层级和世界坐标转换已有来源证据；头胸模型轴对应人体解剖轴的解释需要另列。此次没有通过重新估计轴来降低受控误差。

本轮发现并修正了 CoLLaboratE 的平移读取约定：六通道节点的平移数值已包含局部位置，原读取器又加上 `OFFSET`，重复计入骨段。S1P01R1 的关节点垂直跨度由约 3.06 米修正为 1.53 米。Blender 官方导入器对动画位置减去静止局部位置，BVH SDK 则对带平移通道的节点直接返回该帧平移，两者支持同一解释。[Blender 源码](https://raw.githubusercontent.com/blender/blender-addons/main/io_anim_bvh/import_bvh.py)、[BVH SDK 源码](https://bvhsdk.readthedocs.io/en/latest/_modules/bvhsdk/anim.html#Joints.getLocalTranslation)

这一修正影响原始参考位置、骨长及转入目标骨架后的世界位置。此前将较大的位置误差主要解释为目标骨架差异，解释不完整。原冻结资料保持原值；K3 的最终对照重新生成参考及八路输入，再将相同输入分别交给原拟合与优化拟合。[修正依据与逐例数据](/home/heinrich/MorionGen/config/evalkit_k3_20261116/reference_translation_correction.json)单列来源修正，求解器改善由修正后的配对实验判断。

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
| CoLLaboratE | 每文件 CHANNELS 决定旋转顺序；有平移通道的轴直接采用通道位置，其余轴采用 OFFSET | cm、Y-up 依赖几何推断；发布页未声明段解剖轴。当前共用胸段静止基底属于项目假设 |

全量来源文件、行号、哈希和字段分类保存在 [conventions.json](/home/heinrich/MorionGen/config/evalkit_k3_20261116/conventions.json)。位置、旋转、角度及生物力学解释的来源分开记录，不能仅凭自洽的格式往返把最后一项确认为已知。

CoLLaboratE 发布页确认 BVH 格式及站立、坐姿、跪姿等任务定义；本次只用开发子集，未读取电视留出运动。[官方发布页](https://zenodo.org/records/6472155)

Han 的原始标记与模型估计结果分离遵循 B3D 的输入／处理阶段结构。[Nimble 官方说明](https://nimblephysics.org/docs/working-with-addbiomechanics-data.html)

已保存 K2 独立检出与冻结文件哈希。HYOID 及旧配置、留出使用记录保持原值。全量几何核查和八路拟合另附结果，不把本表的来源审计作为精度验收。

八份真实样例进一步区分了上游约定与交付文件：MotionLCM、OmniControl、ARDY、CHOIS、CLoSD、DART 的公共归档均明确声明米制、右手 Z-up；PRIMAL 和 Kimodo 的原生文件沿用各自格式约定，再进入 K 的右手 Y-up 内部坐标。公共归档声明属于项目适配层，不能倒推为发布者对原始模型的全部定义。[逐样例契约与上游版本](/home/heinrich/MorionGen/config/evalkit_k3_20261116/source_contract_checks.json)保存这两个层次及输入哈希。K3 对文件中明确声明的轴或单位冲突给出错误，缺少声明的旧文件继续使用登记的格式契约。

Kimodo 原生 SOMA77 骨架资产与之前复用的 ARDY SOMA77 资产哈希相同，且与冻结标定资源一致。[资产核查](/home/heinrich/MorionGen/config/evalkit_k3_20261116/source_asset_checks.json)确认了这处复用的依据。CHOIS 当前真实样例走的是逐输入逆 FK 回推，保留为估计来源；DART 真实样例的上身骨方向一致性 P95 为 3.73°，其余五种原生旋转样例为约 0.052°或更低。这些量检验模型坐标与输入几何的匹配，解剖轴精度仍需独立证据。

受控退化的体型选择也单独登记：MotionLCM、OmniControl 和 CHOIS 的测试夹具复用 PRIMAL 的 SMPL22 静止几何。这是项目实验约定；真实生成输出按各自实际关节位置和资源核查。统一骨长评分用于分离该体型选择与方向误差，真实输出保真度另表报告。

全量轻量检查中的 7 条丝织记录具有异常根平移，源文件幅值从约 601,289 米到 (6.30\times10^{35}) 米。将这些世界坐标转为 float32 又会损失肢体尺度的差异。先在原生局部层级去掉根平移，再进行双精度 FK，可保留有限的局部姿态；世界轨迹、地面和接触依据继续受限。这 7 条保持在全量登记中，固定拟合名单维持原值。[逐条诊断](/home/heinrich/MorionGen/config/evalkit_k3_20261116/excluded_source_quality.json)记录原始根范围、浮点误差和局部姿态状态。
