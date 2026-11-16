"""Finalize the causal report and explicit runtime configuration after verification."""
import json,shutil
from pathlib import Path
import hashlib
ROOT=Path(__file__).resolve().parents[2];CFG=ROOT/'config/evalkit_k33_20261117';LOG=ROOT/'develop_logs/test_runs/20261117_k_conversion'
read=lambda p:json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
s=read(CFG/'summary.json');v=read(CFG/'verification.json');coupling=read(LOG/'coupling_check.json');runtime=read(LOG/'coupling_runtime.json');scan=read(LOG/'coupling_scan.json')
assert len(coupling)==17 and all(r['endpoint_max_difference_m']==0 for r in coupling)
assert all(c['after_max_coupler_error_deg']==0 for r in coupling for c in r['sides'])
assert runtime['status']=='passed';assert 'Ran 60 tests' in (LOG/'tests.log').read_text()
write(CFG/'coupling_check.json',coupling);write(CFG/'coupling_runtime.json',runtime);write(CFG/'coupling_scan.json',scan)
config=read(ROOT/'config/evalkit_k_v3_1.json');config['version']='k-3.3-development-topology-and-model-couplers'
config['fitting']['enforce_coordinate_couplers']=True
config['controlled_conversion_policy']='topology_v2'
config['selection']={'scope':'conversion correction: 82x8 lightweight FK checks, 16 actual-input full-window fits; body scores preserved after dependent-coordinate repair',
 'evidence':'config/evalkit_k33_20261117/summary.json','constraint_evidence':'config/evalkit_k33_20261117/model_audit.json',
 'policy':'retain model bounds and locks; enforce declared identity couplers; K3.1 fitting strategy retained to isolate conversion changes',
 'limitations':'ARDY corrected mapping raises most original-reference trunk errors; fixed-target local geometry remains a material transfer loss; no full 82x8 refit or new holdout',
 'compatibility':'prior configs remain byte-identical; retarget defaults to legacy; explicit topology_v2 export and this runtime config form the new development path'}
write(CFG/'runtime_config.json',config)
shape_evidence={}
for method in ['motionlcm','omnicontrol','chois','ardy','dart','kimodo','closd','primal']:
    key='primal' if method in ['motionlcm','omnicontrol','chois'] else method
    path=ROOT/'config/evalkit_k_calibration'/f'{key}.json';cal=read(path)
    shape_evidence[method]={'template':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'asset_metadata':{k:cal[k] for k in ['source_path','sha256','shape','betas','gender'] if k in cal},'selection_status':'project controlled-degradation template choice; raw asset metadata retained from previous audit, not reverified this round'}
write(CFG/'template_selection.json',shape_evidence)
v.update(coupling_repair_records=17,coupling_endpoint_invariance_exact=True,coupling_runtime=runtime['status'],
 historical_coupling_over_1deg=sum(r['max_difference_deg']>1 for r in scan['records']),new_coupled_outputs='data disk / evalkit_k33_conversion_20261117/coupled_fits')
write(CFG/'verification.json',v)
lines=['# K3.3：躯干退化转换修正与下肢模型约束核查', '',
'本轮要回答两件事：躯干偏差有多少来自我们构造退化输入的方式；下肢限制哪些来自模型资产，哪些是项目的处理选择。继续以退化前原始动作 R 为验收依据，D 表示实际退化读回，F 表示拟合输出。原冻结资料与留出使用记录保留。', '',
'完成固定 82 个开发窗口 × 八种格式的 656 路轻量转换检查，以及预先固定的 16 个格式—窗口组合的完整拟合。16 个组合包括每种格式上一轮的躯干高误差窗口和原登记典型窗口；使用原窗口及原评分时钟。拟合保持 K3.1 策略，单独观察转换变化。随后对模型从属坐标作一致性修复并验证端点不变。', '',
'结论：已修正脊柱重复映射和节点压缩的隐含对应，六种格式的躯干降级误差整体降低；ARDY 的结构修正暴露了旧映射与体型偏差之间的抵消，误差上升需要保留。下肢范围和锁定来自官方模型，扩范围/解锁仍保留为诊断。另发现并修复了髌骨从属坐标未随膝角更新的问题，这是现有端点评分看不到的内部一致性缺口。', '',
'## 1. 躯干转换：明确的修正与仍然存在的形状损失', '',
'旧 ARDY 路径把目标 `Spine` 与 `spine1` 都对应到源 `Spine`，遗漏了一个不同的源段旋转。新版 `topology_v2` 按链条顺序建立对应：四节点对应四个不同源节点；节点数不同时，以链条累计长度比例建立单调对应，并保存每条记录的依据。原始层级、位置、旋转和目标资产骨段是已有资料；压缩后的节点对应属于项目推导的约定。源节点不足或骨段退化时显式报出不支持，避免重复节点伪装为独立观测。', '',
'需要分清资产数值与选用策略：当前受控退化里，MotionLCM、OmniControl 和 CHOIS 共用 PRIMAL 的中性 SMPL-X 模板，其余格式采用各自已登记模板。选用固定体型、以及这三种格式共用模板，是项目实验约定；它不表示这些方法必须使用同一种体型。模板值沿用已有源模型标定资料，本轮核对的是缓存值与使用路径，原始资产来源元数据在 template_selection.json 中保留。', '',
'独立解析 FK 测试给四个脊柱节点不同旋转，在相同骨架下验证新版重建原始位置；同时检查世界旋转等变性、旋转矩阵有效性和序列化信息隔离。头、胸原生朝向约定及 HYOID 保持原值。', '',
'以下为每条窗口躯干倾角误差 P95 先按参与者组平均、再对组等权的结果；不是把全部帧混合后的 P95。轻量检查评价目标 FK 相对 R，16 个拟合窗口另评价真实读回。', '',
'|格式|旧转换|修正转换|窗口恶化 >1°|','|---|---:|---:|---:|']
for m,d in s['methods'].items():lines.append(f"|{m}|{d['old_trunk_p95_group_mean_deg']:.2f}°|{d['new_trunk_p95_group_mean_deg']:.2f}°|{d['worse_than_1deg']}/82|")
lines+=['', 'ARDY 的 74 条窗口超过 1° 恶化。结构对应正确与整体倾角误差更小是两项不同检查：旧重复映射恰好抵消了部分目标体型偏差。新版保留正确对应，同时把更大的实际损失报告出来。旧结果继续作为历史版本，后续比较应使用同一转换版本。', '',
'体型替换消融进一步区分了骨长与静止方向。以 MotionLCM 高误差窗口为例，修正映射后的误差为 14.13°；只替换为源骨长后为 8.31°；替换完整局部骨段向量（长度与静止方向）后为 0.13°。ARDY 对应窗口为 6.97°、7.20°、0.34°。因此剩余偏差涉及整条脊柱和肩部链的静止几何，统一骨长只能解释其中一部分。', '',
'这些源形状替换仅用于归因，改变了固定目标骨架，未作为真实基线格式输入导出。正式转换仍采用既定目标资产。当前证据支持继续研究形状与姿态分离的转换，不能把诊断变体的小误差当作现有格式已经达到的精度。', '',
'## 2. 实际拟合：分开看 D−R、F−D、F−R', '',
'以下代表窗口全部对照原始 R，表中为躯干倾角误差 P95，旧→新；三项分别统计，不相减解释贡献率。', '',
'|格式／窗口|降级损失 D−R|拟合偏差 F−D|端到端 F−R|','|---|---:|---:|---:|']
for method,key in [('motionlcm','031'),('ardy','051'),('ardy','070'),('kimodo','010')]:
 r=next(r for r in s['fits'] if r['method']==method and r['window_key']==key)
 lines.append('|'+method+' / '+key+'|'+'|'.join(f"{r['trunk']['old'][k]['p95_abs']:.2f}° → {r['trunk']['new'][k]['p95_abs']:.2f}°" for k in ['degradation','fit_deviation','end_to_end'])+'|')
lines+=['', '![原始、退化、拟合曲线](/home/heinrich/MorionGen/develop_logs/notes/20261117_k33/trunk.png)', '',
'16 个窗口的 96 个区域中，新增 >1° 的姿态告警出现在 ARDY 051 躯干；未触发新增角速度、单帧跳变或覆盖下降门限。另有两项手脚位置代价：ARDY 051 右腕统一骨长后的世界位置 P95 从 6.23 cm 升至 7.61 cm；ARDY 070 右足原始几何位置 P95 从 10.30 cm 升至 11.68 cm。完整逐区域和原始/统一骨长结果保存在 summary.json。', '',
'有效覆盖仍描述对 D 的方向贴合与观测支持；与原始动作的一致性由 F−R 单独验收。所有对照保留原时钟、原始参考数组和评分数量，未通过收紧掩码获得改善。', '',
'## 3. 下肢约束：资产给定的范围与项目处理选择', '',
'本机 RajagopalLaiUhlrich2023 的 16 个下肢坐标定义，与本轮下载的官方资产逐项一致；已拟合模型的范围、锁定和夹紧属性也保持这些值。官方说明指出该版本包含针对较大髋、膝屈曲的关节范围、肌肉路径和肌肉参数更新。应将它们视为该模型的工作范围，而非普遍人体极限。[官方模型说明](https://github.com/opensim-org/opensim-models/blob/master/Models/Rajagopal/README.txt)', '',
'|坐标|资产范围|默认状态|本轮判断|','|---|---|---|---|',
'|髋屈曲|−30°～120°|可动|保留作者模型域|','|髋内收|−50°～30°|可动|保留作者模型域|','|髋轴向旋转|−40°～40°|可动|保留作者模型域|','|膝屈曲|0°～140°|可动，关联髌骨从属坐标|保留范围并同步从属坐标|','|踝角|−50°～50°|可动|保留作者模型域|','|距下关节|−35°～35°|锁定于 0|模型简化；本轮解锁实验不支持直接推广|','|跖趾关节|−45°～30°|锁定于 0|现有脚趾根位置不足以独立观察其屈曲|', '',
'距下关节位于踝下方，模型用一个转轴表达足部内外翻；跖趾关节连接前脚掌与脚趾。锁定来自官方资产，表示模型配置固定了自由度。项目自己的选择包括：把资产范围作为最小二乘硬边界；优化器仍列出锁定的距下变量；前轮将膝上限改为 175°并扩大髋踝范围的诊断；肌肉阶段另行解锁距下关节。新 K 收据明确记录固定变量，防止把优化变量列表误读成有效自由度。', '',
'实际接口探针确认：锁定时请求改变距下角，坐标及足端位置保持原值；解锁后相同改变可移动足端。跖趾解锁改变的是脚趾远端朝向，脚趾根位置本身保持不动。因此验证该自由度需要更远端的观察。', '',
'当前肌肉配置的 `unlock_subtalar=true` 是项目在静态优化阶段的选择：单独的模型读取既有角度轨迹，并进行力矩分配。它没有补充 K 中缺失的观测。本轮保持这一旧配置，只把两阶段的约定分开记录。', '',
'## 4. 跪坐隔离实验：约束存在作用，解锁没有带来稳定收益', '',
'以下沿用 Kimodo 074 的冻结退化输入，独立于本轮躯干转换实验。全窗口左腿误差为区域内几何角 P95 的均值。', '',
'|条件|降级损失|拟合偏差|端到端|输入方向覆盖|','|---|---:|---:|---:|---:|',
'|K3.1 原范围与锁定|0.61°|保留在历史结果|26.21°|0%|',
'|原范围，仅解锁左距下|0.61°|27.81°|27.68°|0%|',
'|扩大髋膝踝，保持锁定（前轮）|0.61°|8.74°|8.64°|52.92%|',
'|扩大髋膝踝并解锁左距下|0.61°|9.92°|9.45°|59.15%|', '',
'解锁后的方向覆盖略有增加，但扩范围条件下最大角步长由 2.32° 升至 21.09°，端到端误差也略升。因此本轮保留官方范围与锁定，不将扩范围或解锁诊断加入运行配置。深屈膝的覆盖问题仍需面向该动作域的模型与求解验证。', '',
'## 5. 新发现：膝与髌骨从属坐标必须保持一致', '',
'模型内有两条明确耦合关系：左右髌骨从属坐标 `knee_angle_*_beta` 等于对应膝坐标。髌骨是膝前方的骨，其位置影响内部肌肉几何。K 细化改变膝角后，原代码保留了旧从属值；关闭约束投影的逐坐标写入允许这种不一致进入 FK。', '',
'这里还必须区分储存单位：OpenSim 将 type-1 膝旋转写为度，type-3 从属坐标仍以原生弧度储存。按单位换算后检查历史 656 条开发结果，648 条的最大耦合误差超过 1°，最大约 152.82°。这些数值是模型内部坐标不一致，不是新增的人体姿态误差统计。', '',
'新显式运行配置在插值后读取模型 XML 中启用的耦合关系，在原生单位下同步从属坐标；当前支持该资产的恒等线性耦合，其他函数或缺列会明确失败。原始运动和独立拟合坐标保持不变。', '',
'16 个新窗口加一个历史极端案例共 17 条检查，修复后耦合误差均为 0，现有评价端点位置逐值不变；新窗口修复前最大内部差异约 27.47°。极端例中髌骨位置可变化约 8.38 cm。完整生产两阶段复算还确认独立坐标最大差异约 0.00000108°、端点约 0.00000000702 m，处于数值等价范围。首次采用逐值级容差过严失败的日志保留，最终判据为 0.0001°／1 µm，仅用于数值复算。', '',
'这解释了为何端点误差良好仍不足以确认内部肌肉几何。新数据盘产物单独保存同步后的轨迹；历史冻结输出保持原值。后续肌肉负荷验收应使用修正轨迹，本轮未运行肌肉负荷实验。', '',
'## 交付与边界', '',
'交付 `topology_v2` 转换、显式 `runtime_config.json`、约束与耦合收据、可复跑脚本、656 路轻量结果索引、16 路完整对照、17 条耦合修复验证及可视化。`retarget` 默认保留 legacy 以复现旧输入，新研究输入显式选用 topology_v2；新运行配置以 K3.1 拟合为基准并启用耦合同步。肩部 K3.2 候选未混入本轮转换消融。', '',
f"60 项 K 测试通过；{v['protected_files_unchanged']} 个旧 config/plans 文件和 82 份原始参考哈希保持一致，新增留出运行 0 次。HYOID 保持原值。大产物存数据盘，剩余约 {v['free_gib']:.1f} GiB。", '',
'下一步的重点是固定目标形状下的躯干姿态保持，尤其 ARDY 修正映射后的偏差；下肢应在明确模型域和输入观测条件后再评估范围或自由度调整。本轮的结果支持结构与约束一致性修正，也明确保留尚未达到精度要求的分支。', '',
'复跑使用现有 OpenSim 环境，并设置 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`。`python/scripts/evalkit_k33_conversion.py --out <数据盘新目录> --light --fit --workers 3` 重建同一清单；其拟合刻意采用冻结 K3.1 以隔离转换作用。新的运行配置为 `config/evalkit_k33_20261117/runtime_config.json`，耦合一致性由 `evalkit_k33_coupling_check.py` 和 `evalkit_k33_runtime_check.py` 单独验收。模型核查、形状消融及报告脚本同目录，原实验源码快照保存在 `develop_logs/test_runs/20261117_k_conversion`。']
(ROOT/'develop_logs/notes/20261117_evalkit_k33_conversion_and_constraints.md').write_text('\n'.join(lines)+'\n')
print('finalized')
