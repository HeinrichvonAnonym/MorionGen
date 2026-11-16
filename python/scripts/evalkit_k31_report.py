"""Render full K3.1 regression evidence, including all flagged local degradations."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from motiogen.evalkit.k.scoring import angles,region
REGIONS={'pelvis':'骨盆线','torso':'躯干','left_arm':'左臂','right_arm':'右臂','left_leg':'左腿／足','right_leg':'右腿／足'}
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False})
def read(p):return json.loads(Path(p).read_text())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();cfg=ROOT/'config/evalkit_k31_full_20261116';s=read(cfg/'summary.json')
 dest=ROOT/'develop_logs/notes/20261116_k31_full';dest.mkdir(exist_ok=True)
 picked=[]
 for method in s['methods']:
  pool=[r for r in s['all_region_changes'] if r['method']==method];picked.append((max(pool,key=lambda r:r['K31']),'残留最高误差'))
 picked += [(r,'新增局部恶化') for r in s['local_regressions'][:3]]+[(r,'角速度恶化') for r in s['temporal_regressions'][:3]]+[(r,'最大逐帧变化增加') for r in s['step_regressions']]
 seen=set();index=[]
 for case,reason in picked:
  key=(case['method'],case['window_key'],case['region'],reason)
  if key in seen:continue
  seen.add(key);r=read(case['result_path'])
  with np.load(r['arrays']) as z:t=z['times'];p={'R':z['R'],'D':z['D'],'K3':z['F:optimized_raw']}
  with np.load(Path(r['output'])/'comparison.npz') as z:p['K3.1']=z['F:k31']
  values={k:angles(v) for k,v in p.items()};metrics=[m for m in values['R'] if region(m)==case['region']]
  metric=max(metrics,key=lambda m:np.percentile(abs(values['K3.1'][m]-values['R'][m]),95))
  if reason=='最大逐帧变化增加':metric=max(metrics,key=lambda m:np.max(abs(np.diff(values['K3.1'][m]))))
  fig,axes=plt.subplots(1,2,figsize=(14,4),layout='constrained')
  for name,color,ls in [('R','#15293c','-'),('D','#2279b5','--'),('K3','#999999','-'),('K3.1','#cf5629','-')]:
   axes[0].plot(t,values[name][metric],label=name,color=color,ls=ls)
   axes[1].plot(t,np.gradient(values[name][metric],t)-np.gradient(values['R'][metric],t),label=name,color=color,ls=ls)
  axes[0].set_title(metric);axes[1].set_title('相对原始参考的角速度误差')
  axes[0].set_ylabel('度');axes[1].set_ylabel('度／秒')
  for ax in axes:ax.legend(ncol=4);ax.grid(alpha=.2);ax.set_xlabel('窗口时间／秒')
  fig.suptitle(f'{case["method"]} · {Path(case["id"]).stem} · {REGIONS[case["region"]]} · {reason}\n原始参考全窗口；部位角度 P95 汇总 {case["K3"]:.2f}° → {case["K31"]:.2f}°')
  path=dest/f'{len(index):02d}_{case["method"]}_{case["window_key"]}.png';fig.savefig(path,dpi=130);plt.close(fig)
  index.append({'case':case,'reason':reason,'path':str(path)})
 (dest/'index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n')
 lines=['# K3.1：82 个开发窗口、八种退化完整回归','',
 '本轮固定 K3.1 配置，对此前登记的 82 个开发窗口、八种退化共 656 组逐一验收。参考保持退化前的原始 BVH；原始几何与统一骨长评分同时保留。K3.1 第一阶段复用冻结 K3，七路重新计算第二阶段；CLoSD 策略相同，复用原轨迹重新评分。此次仅验收，未按新结果调参。原始与归一几何中的 D−R 逐项核对为原值，最终评分有效计数与参考可观测计数一致；CLoSD 的端到端评分另检查与冻结结果逐项相同。','',
 '**完整回归确认七路全身姿态汇总改善，CLoSD 保持原值。** Kimodo 从 2.39° 降至 1.45°，MotionLCM 从 3.35° 降至 2.85°。改善来自拟合阶段，降级损失保持不变。全部 3,936 个部位对照中，没有出现 P95 增加超过 1° 的情况；小于该阈值的变化也完整保留。','',
 '**时序与支持范围存在独立限制。** Kimodo 擦拭动作的右臂姿态 P95 从 18.05° 降到 5.29°，角速度误差 P95 却从 148.99 增到 260.86°/s；该窗口的时序验收仍受限。另有 4 处瞬时变化标记和 5 处方向支持覆盖下降。工业 EmbroideryB 窗口的左腿／足仍有约 25–26° 的区域误差，CLoSD 丝织窗口左臂仍约 45.3°；这些残留误差继续限制对应部位。','',
 '全身姿态汇总先对每条记录的几何角取 P95，再按六部位等权、参与者组等权汇总。它是 P95 的分层汇总，区别于混合全部帧后取一次 P95。LAB 与工业开发场景分别列出；此前用于 K3.1 调整的窗口仍保留在本批开发数据中，因此这批结果属于开发回归。','',
 '| 格式 | 全部：K3 → K3.1 / ° | LAB：K3 → K3.1 / ° | 工业：K3 → K3.1 / ° |','|---|---:|---:|---:|']
 for m,e in s['methods'].items():
  values=[(e['K3_body_p95'],e['K31_body_p95'])]+[(e['subsets'][k]['K3'],e['subsets'][k]['K31']) for k in ['LAB','industrial']]
  lines.append('| '+m+' | '+' | '.join(f'{a:.2f} → {b:.2f}' for a,b in values)+' |')
 lines+=['','## 降级、拟合与端到端误差','',
 'D−R、F−D、F−R 分别回答格式转换损失、对输入的拟合偏差、原始参考验收。三项各自统计 P95，范数和 P95 不作加法贡献分解。长度归一只改变评分几何，不进入拟合器；逐关节点世界／去根位置、骨方向、偏差、中位数和 5°／10°／20°覆盖均保存在数据盘逐记录结果。','',
 '| 格式 | 双臂 D−R / F−D / F−R，° | 双腿 D−R / F−D / F−R，° | 躯干 D−R / F−D / F−R，° | 输入方向支持／配准位置支持 |','|---|---:|---:|---:|---:|']
 for m,e in s['methods'].items():
  cols=[]
  for regs in [['left_arm','right_arm'],['left_leg','right_leg'],['torso']]:
   cols.append(' / '.join(f'{np.mean([e["regions"][r][k] for r in regs]):.2f}' for k in ['degradation','fit_deviation','end_to_end']))
  coverage=[np.mean([v['coverage'][k] for v in e['regions'].values()]) for k in ['angle','registered_position']]
  lines.append('| '+m+' | '+' | '.join(cols)+f' | {coverage[0]:.1%} / {coverage[1]:.1%} |')
 lines+=['','原始参考误差使用完整共同可观测窗口，输入方向支持掩码单独报告；配准位置支持还受直立标定和位置误差条件约束。头胸原生旋转通路与 HYOID 保持原状，本轮全身拟合也未增加独立头部位置求解。','',
 '## 局部恶化与时序连续性','',
 f'共检查 3,936 个格式×窗口×部位组合。相对 K3，部位 P95 增加超过 1° 的有 **{len(s["local_regressions"])}** 个；角速度误差 P95 同时增加超过 20°/s 且超过原值 20% 的有 **{len(s["temporal_regressions"])}** 个；最大逐帧角变化同时增加超过 5° 且超过原值 20% 的有 **{len(s["step_regressions"])}** 个；输入方向支持下降超过 5 个百分点的有 **{len(s["coverage_drops"])}** 个。阈值用于定位复核案例，属于工程筛查约定。全部差值及记录名单均保留。','',
 '| 格式／记录／部位 | 姿态 P95：K3 → K3.1 / ° | 角速度误差：K3 → K3.1 / °/s | 方向支持：K3 → K3.1 |','|---|---:|---:|---:|']
 cases=[];seen=set()
 for c in s['local_regressions'][:8]+s['temporal_regressions'][:8]+s['step_regressions'][:8]:
  key=(c['method'],c['id'],c['region'])
  if key in seen:continue
  seen.add(key);cases.append(c)
  lines.append(f'| {c["method"]}／{Path(c["id"]).stem}／{REGIONS[c["region"]]} | {c["K3"]:.2f} → {c["K31"]:.2f} | {c["velocity_K3"]:.1f} → {c["velocity_K31"]:.1f} | {c["input_direction_coverage_K3"]:.1%} → {c["input_direction_coverage_K31"]:.1%} |')
 lines+=['','角速度由同一时间网上的几何角数值求导得到；另保留最大逐帧角变化。源记录首段突变、退化滤波与求解跳变都在完整窗口内展示，未通过删边界帧改善统计。局部均值或 P95 改善不替代峰值及曲线检查。','',
 'Kimodo 擦拭案例的证据定位到求解分支与坐标插值的衔接：相邻 20 Hz 求解节点的肩部坐标发生约 160–180° 切换，对应区间的中间帧出现尖峰。仅用于诊断地查看首 0.1 秒之后，上臂抬举在求解节点的误差 P95 为 1.36°，中间帧为 9.59°；正式验收仍使用完整窗口。这说明节点拟合改善与插值轨迹连续性需要分别验证。本轮保留冻结配置，后续应优先处理肩部解分支连续性。','',
 '4 处最大逐帧变化标记均落在窗口 0.02–0.03 秒。该区间 R 的变化仅约 0.16–0.23°，D 约 10.5–27.8°，K3.1 约 27.0–30.6°：源窗口首段变化经受控退化滤波后向邻近帧扩散，拟合插值又改变了局部峰值。它们按完整窗口保留，不能只归为原始动作本身的变化。','',
 '覆盖下降也有两种情况。S10P25R3 的四路躯干方向误差在 10° 门槛附近，约 0.03° 的变化就使约 8% 的帧跨过门槛；几何倾斜误差同时略降，因为该标量与完整三维方向判据不同。Kimodo S1P13R3 左腿的方向误差中位数则从 11.18° 升至 12.06°，支持从 36.5% 降至 28.9%，属于实质的局部方向退化，虽然区域角度 P95 增幅仅 0.29°。[逐标记诊断](/home/heinrich/MorionGen/config/evalkit_k31_full_20261116/flag_diagnostics.json)保留节点、时段和阈值附近分布。','',
 '## 位置代价','', '| 格式 | 手端世界位置 P95，K3→K3.1：原几何／统一骨长 cm | 足端世界位置 P95，K3→K3.1：原几何／统一骨长 cm |','|---|---:|---:|']
 for m,e in s['methods'].items():
  cols=[]
  for joint in ['wrist','foot']:
   cols.append(' / '.join(' → '.join(f'{100*max(e["positions"][side+"_"+joint][space][stage] for side in ["left","right"]):.2f}' for stage in ['K3_end_to_end','end_to_end']) for space in ['raw','normalized']))
  lines.append('| '+m+' | '+' | '.join(cols)+' |')
 lines += ['',f'逐记录另有 {len(s["position_regressions"])} 个手／足端点的原几何世界位置 P95 增加超过 1 cm，完整名单保存在汇总文件。该工程筛查与角度、时序指标分别报告。',
 '其中 Kimodo S1P13R3 的右腕相对原始参考从 14.76 增至 20.59 cm；统一骨长后仍从 10.74 增至 15.86 cm，说明该代价还包含姿态方向与根／关节中心对应的影响。该例对退化输入的归一位置偏差却从 8.93 降至 6.63 cm，进一步说明对输入拟合更好与对原始参考更准需要分别验收。']
 lines+=['','位置统计保留骨长和关节中心对应的影响；统一骨长后仍保留的差异还包含根运动与姿态方向。足部接触代理和各阶段位置误差见逐记录 JSON，当前研究结论仍按 K 的运动学范围解释。','',
 '<details><summary>逐部位精度档与有效计数</summary>','',
 '统计按记录、参与者组等权汇总；有效计数按指标×帧累计，同一帧可贡献多个几何角。5°／10°／20°为工程精度档。','',
 '| 格式／部位 | 中位数／P95／有符号偏差，° | ≤5°／≤10°／≤20° | 有效指标帧／总指标帧 |','|---|---:|---:|---:|']
 for m,e in s['methods'].items():
  for reg,v in e['regions'].items():
   tiers=' / '.join(f'{v["tiers"][str(k)]:.1%}' for k in [5,10,20]);n=v['metric_frames']
   lines.append(f'| {m}／{REGIONS[reg]} | {v["median"]:.2f} / {v["end_to_end"]:.2f} / {v["signed_bias"]:.2f} | {tiers} | {n["valid"]} / {n["total"]} |')
 lines+=['','</details>','','## 复核图与复现资料','']
 for item in index:
  lines.append(f'- [{item["case"]["method"]}／{Path(item["case"]["id"]).stem}：{item["reason"]}]({item["path"]})')
 lines+=['', '- [完整汇总与所有局部差值](/home/heinrich/MorionGen/config/evalkit_k31_full_20261116/summary.json)',
 '- [固定名单、配置与实现哈希](/home/heinrich/MorionGen/config/evalkit_k31_full_20261116/freeze.json)',
 f'- 数据盘目录：`{a.out}`。',
 '', f'656 组运行完成，运行失败 0；冻结资料和旧配置的 {s["preservation"]["protected_files"]} 个文件哈希保持原值，K3.1 配置哈希与本轮运行前冻结一致。新增留出运行 0 次；plans 与 HYOID 保持原值。最多 3 个 CPU 工作进程、数值库单线程，剩余磁盘 {s["resource"]["free_gib"]:.1f} GiB。',
 '', '54 项既有自动检查通过；13 张残留误差与时序复核图已逐张检查。完整性与哈希检查完成，时序和覆盖限制按上述结果保留。',
 '', '复跑入口：现有 OpenSim 环境运行 `evalkit_k31_full.py --out DATA_DIR --workers 3`，然后运行 `evalkit_k31_summarize.py --out DATA_DIR`、`evalkit_k31_diagnostics.py`、`evalkit_k31_report.py --out DATA_DIR`。设置 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`。每组有独立完成记录和冻结身份校验，支持断点续跑。','']
 (ROOT/'develop_logs/notes/20261116_evalkit_k31_full.md').write_text('\n'.join(lines))
 print('report and',len(index),'plots written')
if __name__=='__main__':main()
