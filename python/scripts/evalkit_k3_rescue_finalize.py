"""Record selection, fixed-window evidence and preservation checks."""
import json,hashlib,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k3_rescue_20261116')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
cfg=ROOT/'config/evalkit_k3_rescue_20261116';record=read(cfg/'experiment.json');data=read(BASE/'acceptance.json')
assert len(data['tail'])==24 and len(data['typical'])==8
for p,h in {**record['protected']['files'],**record['k3_sha256']}.items():assert sha(ROOT/p)==h,p
assert not subprocess.check_output(['git','diff','2bbbd42','--','plans'],cwd=ROOT)
assert read(BASE/'runtime_check.json')['status']=='passed'
assert (ROOT/'develop_logs/test_runs/20261116_k3_rescue/tests.log').read_text().rstrip().endswith('OK')
for r in data['tail']+data['typical']:assert sha(Path(r['reference']))==r['reference_sha256']
config=read(ROOT/'config/evalkit_k_v3.json');config['version']='k-3.1-development-interior-rescue'
for method,settings in config['fitting']['per_method'].items():
 settings.update(interior_rescue=method!='closd',rescue_after_baseline=method!='closd',baseline_max_nfev=35,max_nfev=35 if method=='closd' else 100)
config['selection']={'baseline_commit':'2bbbd42','scope':'eight known failure windows plus eight preselected typical development windows; no new holdout',
 'evidence':'config/evalkit_k3_rescue_20261116/acceptance.json','evidence_sha256':sha(BASE/'acceptance.json'),
 'policy':'same coordinate ranges; CLoSD retains baseline because temporal error increases; other seven use two-pass interior rescue',
 'default_compatibility':'explicit K3.1 only; K2 and K3 configurations retained'}
shutil.copy2(BASE/'acceptance.json',cfg/'acceptance.json');write(ROOT/'config/evalkit_k_v3_1.json',config)
shutil.copy2(BASE/'typical_selection.json',cfg/'typical_selection.json');shutil.copy2(BASE/'runtime_check.json',cfg/'runtime_check.json')
free=shutil.disk_usage(BASE).free/1024**3;assert free>=80
verification={'status':'passed_with_documented_limits','protected_files_unchanged':len(record['protected']['files'])+len(record['k3_sha256']),
 'new_holdout_runs':0,'frozen_commit':'2bbbd42','tests_passed':54,'tail_windows':8,'typical_windows':8,
 'full_tail_ablation_cells':24,'runtime_composition':'exact coordinate and position equality, full Kimodo window',
 'visual_review':'all eight temporal overlays inspected; CLoSD boundary transient and Kimodo late-window change retained',
 'figure_sha256':sha(ROOT/'develop_logs/notes/20261116_k3_rescue/temporal.png'),'free_gib':free,
 'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'utils/motiogen/evalkit/k/refinement.py',ROOT/'python/scripts/evalkit_k3_rescue_accept.py']}}
write(cfg/'verification.json',verification)
lines=['# K3.1：局部失败修复与开发验收','','本轮在原始 BVH 参考下，比较八个已知失败窗口的三种后续求解策略，并对八份预先固定的典型开发动作回归。全部使用原窗口和共同时间网；支持掩码独立报告。旧 K3 冻结提交为 `2bbbd42`。',
 '', '**主要发现：高误差包含明显的局部求解分支问题。** 保持原模型范围，增加内部初值搜索即可消除大部分工业手臂偏差，并显著改善 Kimodo 跪姿。增加迭代预算而沿用原种子，仍留下大误差；扩大范围的效果随案例变化。八个窗口的三组配置使用相同 100 次函数评估上限，因此该消融把初值策略与单纯增加预算分开。',
 '', '**选定七路启用同范围救援，CLoSD 保留 K3。** CLoSD 的姿态误差下降，但其约一秒记录的首段重建过渡使角速度误差 P95 从 173.8 增至 770.5°/s。该段原始运动及退化滤波本身也含边界过渡；本轮按完整窗口保留它，未删帧改善统计。Kimodo 的残余约束和末段变化继续作为部分支持范围。',
 '', '新入口执行原 K3 的 35 次预算第一阶段，再执行 100 次预算的多初值第二阶段。内部候选使用固定种子 1729，六组位于范围 15%–85% 内的初值；平方残差大于 0.01 时启用，按输入残差排序并提前停止。这些属于开发工程设置。原始参考从未进入求解器或候选选择。',
 '', '下表为目标部位内各几何角的逐窗口 P95 等权均值，单位度。D−R 为降级；F−D 为拟合偏差；F−R 为原始参考验收。CLoSD 行展示未采用候选以完整保留证据。',
 '', '| 格式／部位 | D−R | K3 F−R | 候选 F−D | 候选 F−R | 输入方向支持：K3→候选 |', '|---|---:|---:|---:|---:|---:|']
for r in data['tail']:
 if r['variant']!='interior_seed':continue
 v=r['regions'][r['focus']];lines.append(f'| {r["method"]}／{r["focus"]} | {v["D"]["mean_metric_p95_deg"]:.2f} | {v["K3"]["mean_metric_p95_deg"]:.2f} | {v["fit_deviation_p95_deg"]:.2f} | {v["new"]["mean_metric_p95_deg"]:.2f} | {v["K3"]["input_direction_coverage"]:.1%} → {v["new"]["input_direction_coverage"]:.1%} |')
lines+=['','Kimodo 右腿从 97.35° 降至 11.85°，方向支持达到约 88%。宽范围候选在峰值附近还可进一步降低误差，但本轮配置保留原模型范围。支持不足的帧继续进入完整误差统计，力学入口沿用配准及位置条件。输入方向支持描述对 D 的保真；原始参考精度由 F−R 单独判断。',
 '', '## 时序与空间代价','', '| 格式 | 角速度误差 P95：K3→候选，°/s | 最大逐帧角变化：K3→候选，° | 世界位置 P95：K3→候选，cm |','|---|---:|---:|---:|']
for r in data['tail']:
 if r['variant']!='interior_seed':continue
 v=r['regions'][r['focus']];p=r['world_position_p95_m'];lines.append(f'| {r["method"]} | {v["K3"]["velocity_error_p95_deg_s"]:.1f} → {v["new"]["velocity_error_p95_deg_s"]:.1f} | {v["K3"]["maximum_step_deg"]:.1f} → {v["new"]["maximum_step_deg"]:.1f} | {p["K3"]*100:.2f} → {p["new"]*100:.2f} |')
lines+=['','角速度误差采用相同时间网数值求导，相对 R 的差值统计；最大角变化独立保留。ARDY 的最大逐帧变化增加，但主要集中在源记录起始突变附近，窗口角速度 P95 下降。Kimodo 末段仍有局部变化，角速度 P95 略增；这限制动态用途，当前结果主要支持姿态几何。位置统计覆盖共同骨架，保留各自原始骨长，未通过评分归一改变拟合输出。',
 '', '![全窗口姿态及角速度对照](/home/heinrich/MorionGen/develop_logs/notes/20261116_k3_rescue/temporal.png)',
 '', '## 典型开发回归与边界','', '典型记录取自上轮已登记的八路“典型基准误差”案例，拟合前保存名单。八路候选所有部位的 P95 最大增加约 0.007°；MotionLCM 一侧手臂改善约 20.13°。失败窗口的其他部位均没有超过 1° 的 P95 恶化。此检查覆盖 16 个格式×窗口，未重跑先前全部 656 组，泛化范围据此限定。',
 '', '54 项既有自动检查通过。正式两阶段组合在完整 Kimodo 失败窗口与实验产物的坐标及关节位置逐项相同。固定窗口中 R、D、F 的参考哈希、输入来源、逐部位误差、覆盖和速度均保留在验收 JSON；完整窗口产物在数据盘。',
 '', f'冻结资料、旧配置与留出记录共 {verification["protected_files_unchanged"]} 项哈希一致；新增留出运行 0 次。plans 与 HYOID 保持原值。数据盘剩余 {free:.1f} GiB。',
 '', '显式选择 [evalkit_k_v3_1.json](/home/heinrich/MorionGen/config/evalkit_k_v3_1.json) 使用新策略。既有默认 K2 与显式 K3 配置保持原值。',
 '', '- [逐窗口验收](/home/heinrich/MorionGen/config/evalkit_k3_rescue_20261116/acceptance.json)', '- [典型动作名单](/home/heinrich/MorionGen/config/evalkit_k3_rescue_20261116/typical_selection.json)', '- [冻结核查](/home/heinrich/MorionGen/config/evalkit_k3_rescue_20261116/verification.json)',
 '', '复跑：现有 OpenSim 环境执行 `evalkit_k3_rescue_probe.py --full`、`evalkit_k3_rescue_accept.py --typical`，然后执行 `evalkit_k3_rescue_accept.py`、`evalkit_k3_rescue_runtime_check.py`、`evalkit_k3_rescue_figures.py`、`evalkit_k3_rescue_finalize.py`。并行工作进程总数保持不超过 3，数值库单线程。','']
(ROOT/'develop_logs/notes/20261116_evalkit_k3_rescue.md').write_text('\n'.join(lines))
print(verification)
