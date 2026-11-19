from pathlib import Path
import json,html,datetime,hashlib,statistics
R=Path('/home/heinrich/MorionGen');B=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k4_20261117'); E=B/'gpu_engineering_v1_checkout';G=B/'gpu_engineering_20261119_v1'
OUT=R/'docs/assets/20261119_k4_method_atlas';OUT.mkdir(exist_ok=True,parents=True)
protocol=json.loads((B/'full_candidate_ablation_20261118_v1/protocol.json').read_text());perf=json.loads((G/'performance_v2/progress.json').read_text());now=datetime.datetime.now().isoformat(timespec='seconds')
C={'ink':'#19334a','blue':'#315b92','teal':'#137e79','amber':'#b57418','red':'#ae5361','muted':'#657789','gray':'#a6b3be','paper':'#ffffff'}
class Figure:
 def __init__(self,num,title,sub,h=550):
  self.num=num;self.h=h;self.s=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 {h}" role="img" aria-labelledby="t{num} d{num}"><title id="t{num}">{html.escape(title)}</title><desc id="d{num}">{html.escape(sub)}</desc><defs><marker id="arrow{num}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10Z" fill="#657789"/></marker></defs><rect width="1200" height="{h}" fill="white"/><style>text{{font-family:Arial,"Noto Sans CJK SC","Microsoft YaHei",sans-serif;fill:#19334a}}.small{{font-size:16px;fill:#657789}}.body{{font-size:18px}}.head{{font-size:20px;font-weight:700}}</style>']
  self.text(32,37,f'{num:02d}  {title}',26,True);self.text(32,66,sub,16,color='muted')
 def text(self,x,y,t,size=18,bold=False,color='ink',anchor='start'):
  self.s.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{700 if bold else 400}" fill="{C.get(color,color)}" style="fill:{C.get(color,color)}" text-anchor="{anchor}">{html.escape(t)}</text>')
 def rect(self,x,y,w,h,color='blue',fill=None,dash=False):
  col=C.get(color,color);fill=fill or {'blue':'#eff4fa','teal':'#edf7f5','amber':'#fff7e8','red':'#fbf0f2','gray':'#f4f6f8'}.get(color,'#fff');self.s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{col}" stroke-width="1.4" '+('stroke-dasharray="6 5"' if dash else '')+'/>')
 def box(self,x,y,w,title,lines=(),color='blue',h=100):
  self.rect(x,y,w,h,color);self.text(x+16,y+29,title,19,True,color)
  for i,line in enumerate(lines):self.text(x+16,y+56+23*i,line,16)
 def arrow(self,pts,dash=False):
  d='M'+' L'.join(f'{x},{y}' for x,y in pts);self.s.append(f'<path d="{d}" stroke="#657789" stroke-width="1.7" fill="none" marker-end="url(#arrow{self.num})" '+('stroke-dasharray="6 5"' if dash else '')+'/>')
 def line(self,x1,y1,x2,y2,color='gray',width=2):self.s.append(f'<path d="M{x1} {y1} L{x2} {y2}" stroke="{C.get(color,color)}" stroke-width="{width}" fill="none"/>')
 def circle(self,x,y,r=7,color='teal'):self.s.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{C.get(color,color)}"/>')
 def done(self,name):
  v=''.join(self.s)+'</svg>';(OUT/(name+'.svg')).write_text(v);return v
figs=[]
f=Figure(1,'K alignment：观测驱动的模型重建','运行时求解只接收 D 与声明输入；R 在独立评分端使用。图示为方法结构，非实测轨迹。',560)
f.rect(25,92,1150,259,'blue',fill='#f8fafc',dash=True);f.text(43,117,'RUNTIME  ·  输入侧计算与候选输出',16,True,'blue')
for x,title,lines in [(45,'D：八格式观测',['位置 / 可用旋转','时钟、掩码、input_rig']),(275,'固定几何 θ',['节点注册、长度、范围','片段内保持固定']),(505,'姿态候选 q₁:T',['初值 → 逐帧 / 整段求解','方向、位置、时序目标']),(735,'组合 FK → F',['root 定位 + 头颈链','原生解 / 导出回读'])]:f.box(x,146,207,title,lines,h=107)
for x in [252,482,712]:f.arrow([(x,199),(x+21,199)])
f.box(965,146,190,'D-only 质量选择',['保留候选 / 回退','整段选择或分支输出'],h=107,color='teal');f.arrow([(942,199),(963,199)])
f.text(47,293,'FK（正运动学）：由关节坐标 q 与固定模型 θ 重新计算节点位置和段旋转。',18)
f.text(47,321,'头胸原生旋转读出另列；欠观测分量保持未知。模型几何与评分用共同骨长分别管理。',16,color='muted')
f.box(45,398,207,'R：退化前原动作',['只进入受控构造与验收','运行时无 R 回路'],color='amber',h=111)
f.box(320,398,445,'独立评分：D−R / F−D / F−R',['原始几何 + 统一骨长；native / export × D / R 时钟','完整窗口、来源跳变分层、有效覆盖、实际文件回读'],color='amber',h=111)
f.box(820,398,330,'两类不同的结论',['原动作准确度：5° / 5 cm 与时序','计算迁移一致性：CPU ↔ GPU'],color='amber',h=111)
f.arrow([(252,453),(318,453)]);f.arrow([(843,253),(843,374),(633,374),(633,397)]);f.arrow([(149,253),(149,371),(431,371),(431,397)],True);f.arrow([(765,453),(818,453)])
figs.append(('pipeline',f.done('01_pipeline'),'先保持输入语义，再求解模型姿态。R用于验收与开发选型；实际拟合、补偿和运行时选择使用D。位置输入缺少的独立轴向旋转保留未知。'))
f=Figure(2,'候选方法：几何、目标与求解方式的组合','当前消融协议声明 13 个原生配置 + 11 个协调导出变体 = 24 列；按输入契约判定适用性。',580)
f.box(34,110,270,'A  固定几何 / 注册',['源骨架关节中心与肩宽','位置链长代理 / 总链长代理','足端注册 / 输入足部几何'],h=136)
f.box(34,275,270,'B  范围与耦合',['独立运动学坐标域','膝辅助耦合延拓','记录原模型范围与变化'],h=136,color='amber')
f.box(344,110,362,'C  逐帧方向—位置拟合',['nominal / continuity / 条件追加初值','骨盆线、方向、几何角、端点位置','正向 / 反向遍历；新增变化率约束'],h=136)
f.box(344,275,362,'D  整段轨迹联合求解',['位置输入轨迹与等价坐标图','全窗口 q₁:T + 相邻方向误差变化','有界 GN / TRF–LSMR（按配置）'],h=136,color='teal')
f.box(748,110,416,'E  头颈、root 与输出路径',['头颈链附着身体胸段，固定长度','双髋中点定位 root 后重新 FK','SO(3) 初始插值 + 可选跨关节协调'],h=136)
f.box(748,275,416,'F  候选验收 / 选择分列',['每个适用候选分别对 R 验收','D-only 选择器单独评价命中 / 漏选','完整消融保留弱对照与失败'],h=136,color='teal')
f.arrow([(304,178),(342,178)]);f.arrow([(304,343),(342,343)]);f.arrow([(706,178),(746,178)]);f.arrow([(706,343),(746,343)]);f.arrow([(956,246),(956,273)])
f.rect(34,448,1130,98,'gray');f.text(54,479,'min  E(q) = Eposition + Edirection + Eangle + Etemporal     s.t. q ∈ Qθ',23,True)
f.text(54,513,'概念分解；Qθ表示坐标域与耦合。权重与停止规则按配置固定；多因素候选比较需另做因果消融。',17)
figs.append(('candidates',f.done('02_candidates'),'同一方法家族会在几何、遍历方向和导出策略上产生不同候选。完整名称、启用项及适用分母见后面的24列清单。'))
f=Figure(3,'并行的边界：沿 Batch 扩展，沿状态依赖推进','灰蓝箭头是必须保留的状态依赖；绿色列表示同一时刻可聚合的计算。',595)
f.text(35,110,'A  连续逐帧拟合：qₜ₋₁ 会影响当前初值与时序残差',19,True)
for j,seq in enumerate(['序列 A','序列 B','序列 C']):
 y=140+j*81;f.text(40,y+32,seq,18,True)
 for i in range(4):
  x=160+i*220;f.box(x,y,175,f'帧 t{("+"+str(i)) if i else ""}',[f'独立状态 {seq[-1]} · FK / J'],h=60,color='blue')
  if i<3:f.arrow([(x+175,y+30),(x+218,y+30)])
f.rect(149,132,197,235,'teal',fill='none',dash=True);f.text(175,392,'跨序列微批',17,True,'teal');f.text(390,392,'不同序列保持各自时钟、活动坐标与停止状态',17)
f.box(35,432,350,'B  同一迭代可批量',['有限差分扰动列；独立常规初值','独立候选；输入已就绪的序列'],h=122,color='teal')
f.box(425,432,350,'C  整段求解可批量',['给定 q₁:T 后，全帧 FK / 残差','时序产生块稀疏耦合，保留结构'],h=122,color='teal')
f.box(815,432,350,'D  保持顺序的部分',['迭代更新、触发条件、接受决策','逻辑顺序可保留在 GPU 上执行'],h=122,color='amber')
figs.append(('batch',f.done('03_batch'),'自回归依赖限制同一序列沿时间直接独立求解，但不限制跨序列合批。GPU上的掩码控制可以保留决策顺序；这部分目前尚未完成。'))
f=Figure(4,'CPU ↔ GPU：当前工程路径与剩余成本','蓝色为当前 CPU 控制；绿色为已接入计算；虚线框为待完成的工程能力。',600)
f.rect(30,105,470,350,'blue',fill='#f5f8fc');f.text(53,136,'CPU · 冻结算法与残差定义',22,True,'blue')
f.box(53,157,424,'① 初值 / 边界 / 收敛与预算',['SciPy TRF；整段 GN / LSMR','每个样本保留原停止状态'],h=98)
f.box(53,292,424,'④ 线性子问题与接受步骤',['残差组装、稀疏线性代数、线搜索','迭代仍在 CPU，尚无完整设备控制'],h=110,color='amber')
f.rect(700,105,470,350,'teal',fill='#f2faf8');f.text(723,136,'GPU · 常驻固定模型',22,True,'teal')
f.box(723,157,424,'② 批量 FK / 有限差分扰动',['float64；关闭 TF32 / 混合精度','模型缓存、CUDA Graph、坐标批变换'],h=98,color='teal')
f.box(723,292,424,'③ 回传所需数值',['轻量数值协议；按需旋转输出','CPU 继续残差与决策'],h=110,color='teal')
f.arrow([(477,203),(721,203)]);f.text(528,188,'q 与扰动批',17)
f.arrow([(935,255),(935,290)]);f.arrow([(723,347),(479,347)]);f.text(540,331,'位置 / 旋转',17)
f.arrow([(63,292),(42,292),(42,267),(63,267),(63,255)])
f.box(30,488,540,'为什么需要回传？',['当前决策仍在 SciPy / NumPy；协议隔离两个既有环境。','保留同一算法迁移到设备控制后，可减少这些边界。'],h=89,color='gray')
f.rect(610,488,560,89,'amber',dash=True);f.text(630,517,'待完成：设备端线性求解 + 收敛掩码 + 异构几何 Batch',18,True,'amber');f.text(630,548,'迁移逐项验收；更换求解算法另设条件。',17)
figs.append(('execution',f.done('04_execution'),'GPU已加速的是反复调用的模型计算。CPU还承担算法控制和大部分整段线性求解，主机与设备之间因此仍有往返。整套循环驻留GPU属于后续工程工作。'))
f=Figure(5,'输出路径：原生拟合、插值与回读分别验收','关节旋转采用 SO(3) 路径；多个关节分别平滑，仍可能产生不理想的末端运动。',555)
f.text(35,115,'A  原生锚点固定',19,True);f.line(65,190,1110,190)
for x in [65,410,755,1110]:f.circle(x,190,9,'blue');f.text(x,165,'qₜ',19,True,anchor='middle')
for x in [151,237,323,496,582,668,841,927,1013]:f.circle(x,190,5,'gray')
f.text(35,230,'蓝点：已拟合原生坐标；灰点：新增导出采样。全部输出仍由同一固定模型 FK 产生。',18)
f.box(35,274,340,'B  SO(3) 插值',['旋转群上的连续旋转路径','回到模型坐标与耦合','保留原生采样点'],h=130)
f.box(430,274,340,'C  协调中间姿态',['原生 FK 端点线性插值作目标','按 D 的支持掩码门控','联合关节重求解；锚点不动'],h=130,color='teal')
f.box(825,274,340,'D  文件输出与独立回读',['协调前 → 协调后 → 实际文件','D 时钟 / R 时钟分别验证','报告插值新增误差与原生误差'],h=130,color='amber')
f.arrow([(375,338),(428,338)]);f.arrow([(770,338),(823,338)])
f.rect(35,439,1130,80,'gray');f.text(52,470,'批处理空间：独立区间与中间帧的 FK / 残差。当前复用逐帧 refine，连续初值仍需显式处理。',18)
f.text(52,499,'线性端点路径是插值假设；它未增加观测信息。直接移除连续初值或减少导出采样要另作算法条件。',17)
figs.append(('export',f.done('05_export'),'协调导出是独立方法因素，可复用同一原生拟合结果。导出路径质量、原生姿态质量与文件精度分别报告，避免把插值改善归给原生求解。'))
f=Figure(6,'性能证据：局部计算快，整体循环仍受限制','两个面板来自不同实验，计时范围不同；不可把内核倍数直接视为整条 K 加速倍数。',530)
f.text(35,112,'A  固定32帧 FK 请求 · 预热中位数 / ms',19,True)
for i,(name,v) in enumerate([('逐算子执行',34.20),('CUDA Graph',7.30),('Graph + 坐标批变换',5.07)]):
 y=147+i*76;f.text(35,y+17,name,17);f.rect(239,y,round(v*6.3),30,'teal');f.text(250+v*6.3,y+22,f'{v:.2f}',17,True)
f.text(35,402,'局部约 6.8×；同请求边界，含往返。',18,True,'teal')
f.text(625,112,'B  150帧姿态细化 · 预热中位数 / s',19,True)
for i,(name,v) in enumerate([('控制实现',76.567),('轻量传输',69.059),('组合优化',66.039)]):
 y=147+i*76;f.text(625,y+17,name,17);f.rect(756,y,v*3.8,30,'blue');f.text(768+v*3.8,y+22,f'{v:.2f}',17,True)
f.text(625,402,'局部姿态细化减少 13.7%；约 2.27 帧/s。',18,True,'blue')
f.line(590,102,590,421)
f.text(35,461,'整段9项已剖析任务：CPU线性子问题占合计求解时间约83.2%。范围仅覆盖这9项。',19,True,'amber')
f.text(35,494,'相同输入 / 候选，原14个CPU worker并行；计时复用既有模型与种子，完整部署延迟、全候选吞吐与GPU默认资格仍待验证。',17)
figs.append(('evidence',f.done('06_evidence'),'FK局部加速与传输优化已经有实测收益。整段任务的主要成本仍在CPU线性子问题，逐帧任务则反复支付控制和通信成本。进一步收益需要改变执行位置和批规模。'))
# Explicit implementation/evidence states, not claims of production qualification.
rows=[
('输入读取、格式适配、缓存','跨文件；数值转换可批','已接入','八格式CPU读取器复用；按内容哈希缓存与数值列缓存已有','完整读取/准备流程的GPU化与统一流水预取尚未完成；先减少重复解析','motion.py / input_rig.py / backends/score_store.py','done'),
('固定模型与几何准备','跨候选/序列；同源资产复用','部分接入','输入驱动固定几何与OpenSim→Torch模型导出已有','每个固定模型仍须建模与核验；不同几何的通用批结构待完成','fixed_geometry.py / backends/opensim_export.py','partial'),
('OpenSim基础IK与种子生成','跨候选/独立初值可并行','CPU已有','标记IK、输入多初值及全窗口种子选择保持原实现','本轮GPU面板复用固定模型和求解前状态；基础IK尚未GPU迁移','initialization.py / k4_fitting.py','todo'),
('模型 FK、坐标变换','同模型的帧/扰动/请求批','已接入','float64批量FK、耦合、模型驻留、CUDA Graph、坐标批变换','完整算法覆盖与默认替换仍由独立验收决定','backends/torch_model.py / engineering/graph.py','done'),
('有限差分雅可比','同次迭代的扰动列可批','已接入','沿用CPU步长/边界；批量计算扰动FK','设备端直接生成全部残差/雅可比尚未完成；自动微分属独立导数条件','engineering/derivatives.py / kernel.py','done'),
('同帧多初值','独立初值可并行；追加初值有触发条件','局部实验','双CPU控制器共享设备计算已实现；4帧3.22s，顺序2.65s','该测试更慢，主面板仍顺序；设备端批控制待完成','engineering/batching.py / contracts.py','local'),
('连续逐帧拟合','跨序列/候选可批；同序列有前态依赖','部分接入','GPU模型计算 + CPU原TRF控制','完整设备驻留TRF、活动样本掩码和停止管理尚未实现','refinement.py / engineering/kernel.py','partial'),
('整段轨迹残差','给定当前轨迹后可全帧批','部分接入','全帧模型计算与有限差分批；原稀疏目标保留','稀疏雅可比组装与整个迭代尚未完整驻留GPU','trajectory_objective.py / trajectory.py','partial'),
('有界稀疏线性子问题','矩阵算子可GPU；独立系统可批','局部实验','CPU转置CSR缓存：两矩阵约5.6% / 7.9%局部提速','CPU等价LSMR/边界管理的GPU版本待实现；不能把现有独立GN算作已完成迁移','trajectory_solver.py / engineering/linear.py','local'),
('迭代、线搜索与停止','有依赖但可设备执行；独立样本掩码批','尚未实现','当前按CPU原预算与顺序决策','设备端信赖域更新、接受/回退及投机计算预算需保持','engineering/contracts.py / trajectory_solver.py','todo'),
('跨序列批处理','相同模型可微批；异构几何可另建批模型','局部实验','两个6帧片段7.65→5.54–6.15s；位置差0.0163mm','完整旋转/时序资格待验；40项仅两组共8项共享模型，异构几何未实现','engineering/sessions.py','local'),
('跨候选调度','独立配置可并行；完全相同产物可复用','部分接入','CPU独立候选、内容哈希复用及全适用消融已有','GPU全候选动态队列和异构固定几何分桶尚未完成','candidate_pool.py / candidate_ablation.py','partial'),
('root补偿与附着头颈链','已知身体姿态后，多帧可批','CPU已有','双髋中点root与头颈组合FK已有；原生旋转对照单列','完整头颈/补偿路径的GPU批化尚未端到端接入；欠观测轴向旋转保持未知','root.py / head_neck.py / output.py','todo'),
('SO(3)插值与协调导出','区间独立计算；中间拟合需管理前态','CPU已有','SO(3)、固定锚点协调路径、原生结果复用已落地','完整协调导出GPU批求解尚未接入；插值目标属于假设','resampling/coordinated.py / rotation_resampling.py','todo'),
('评分与统一骨长','窗口/比较/部位/时钟可批','已实现独立后端','Torch score_batch保留有效掩码、跳变分层及分位数','当前冻结CPU评分继续；完整头颈/回读验收链的统一GPU覆盖仍待证明','backends/torch_scoring.py','partial'),
('独立OpenSim FK / 文件回读','可跨文件CPU并行与严格缓存','CPU已有','作为独立计算基准与导出检查','保留独立性；无需把验收与求解合成同一GPU实现','k4_fitting.py / engineering/acceptance.py','partial'),
('报告聚合与落盘','可增量/缓存；I/O可流水','已接入','哈希复用、紧凑评分列和报告节流','HTML/文件I/O优先CPU与缓存，GPU计算收益需以实测决定','backends/score_store.py / reporting/candidate_ablation.py','done')]
# Candidate catalogue derived directly from frozen protocol.
desc={
'pelvis_continuity':'加强骨盆线与方向变化的连续性；逐帧多初值。',
'balanced_targets':'统一归一方向、位置、几何角及超目标惩罚。',
'rig_centres':'采用声明源骨架关节中心，检验模型节点注册。',
'rig_arm_domain_centres':'声明中心与足几何，扩展足踝与上臂轴向坐标域。',
'rig_reverse_leg_path':'声明足几何与足踝范围，反向遍历。',
'rig_reverse_rate_leg_path':'反向腿部分支增加方向误差变化率惩罚。',
'reverse_full_path':'反向原生求解并自带全身协调导出。',
'position_chain':'位置输入的观察链长代理，保持固定几何。',
'position_fixed_foot_chain':'链长代理加输入可辨识的固定足部几何。',
'position_trajectory':'从逐帧种子进入整段有界Gauss–Newton。',
'position_trajectory_chart':'整段求解前选择余量更大的等价坐标表示。',
'forward_full_path':'正向原生求解并自带全身协调导出。',
'position_total_proxy':'总链长代理加等价坐标图与整段求解。'}
assert len(protocol['templates'])==24 and len(desc)==13
sources=[B/'full_candidate_ablation_20261118_v1/protocol.json',G/'performance_v2/progress.json',G/'acceptance_v2/latest.json',R/'develop_logs/notes/20261119_k4_gpu_engineering.md',R/'develop_logs/notes/20261119_k4_gpu_continuity_performance.md']
for rel in ['candidate_pool.py','k4_fitting.py','refinement.py','trajectory_solver.py','head_neck.py','resampling/coordinated.py','backends/torch_scoring.py','backends/engineering/contracts.py','backends/engineering/sessions.py','backends/engineering/linear.py']:
 sources.append(E/'utils/motiogen/evalkit/k'/rel)
sources.append(E/'python/scripts/evalkit_k4_gpu_engineering_case.py')
manifest={'time':now,'scope':'read-only methodology and acceleration audit; no experiment mutations','sources':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources],'performance_snapshot':perf,'declared_candidates':protocol['templates']}
(R/'develop_logs/test_runs/20261119_k4_method_atlas/evidence.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
esc=html.escape
css='''*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f3f5f8;color:#19334a;font:16px/1.75 system-ui,-apple-system,"Noto Sans CJK SC",sans-serif}header{background:#173149;color:white;padding:54px max(24px,calc((100vw - 1220px)/2)) 44px}header h1{font-size:38px;line-height:1.3;margin:12px 0}header p{max-width:1000px;color:#d2dee7}small,.muted{color:#657789}header small{color:#bed1df}.eyebrow{font-size:12px;letter-spacing:.2em;font-weight:700;color:#8bd0c7}main{max-width:1268px;margin:auto;padding:24px}nav{position:sticky;top:0;background:#fffefa;z-index:2;padding:10px 24px;display:flex;gap:20px;flex-wrap:wrap;border-bottom:1px solid #d7e0e7}a{color:#315b92;text-decoration:none}a:hover{text-decoration:underline}section{background:white;border:1px solid #dce4eb;border-radius:10px;padding:28px;margin:24px 0;scroll-margin-top:70px}h2{font-size:25px;margin:0 0 12px}h3{font-size:19px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}.card{padding:20px;background:#f1f6f9;border-left:3px solid #315b92}.card strong{display:block;font-size:24px}.card p{font-size:14px;margin-bottom:0}.lead{font-size:19px}.callout{padding:18px 22px;border-left:4px solid #137e79;background:#edf7f5}.warn{background:#fff7e8;border-color:#b57418}.figure{overflow:auto;margin:20px -10px 8px}svg{width:100%;height:auto;min-width:0;display:block}.caption{color:#657789;font-size:14px}.bar{display:flex;justify-content:space-between;gap:15px;align-items:center}.download{font-size:13px;white-space:nowrap}table{border-collapse:collapse;width:100%;font-size:14px}th{background:#edf3f7;text-align:left;padding:11px;position:sticky;top:0}td{padding:12px 11px;vertical-align:top;border-bottom:1px solid #dce4eb}td:first-child{font-weight:650}code{background:#eef2f5;font-size:12px;padding:2px 4px;overflow-wrap:anywhere}pre{font-size:12px;overflow:auto;background:#f5f7f9;padding:16px;max-height:330px}.table-scroll{overflow:auto}.badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:12px;white-space:nowrap}.done{background:#e0f2ed;color:#14695f}.partial{background:#e6eef9;color:#315b92}.local{background:#fff0d6;color:#855710}.todo{background:#eff0f2;color:#657789}button{background:#fff;border:1px solid #b7c6d2;padding:7px 12px;border-radius:5px;color:#19334a;cursor:pointer}button.active{background:#19334a;color:white}.filters{display:flex;gap:8px;flex-wrap:wrap;margin:16px 0}details{border-top:1px solid #dce4eb;padding:14px 0}summary{cursor:pointer;font-weight:600}.metric{font-variant-numeric:tabular-nums}input[type=range]{width:240px;vertical-align:middle}.calculator{display:grid;grid-template-columns:1.5fr 1fr;gap:24px;padding:20px;background:#f4f8fa}footer{padding:20px;font-size:13px;color:#657789}@media(max-width:600px){svg{min-width:760px}}@media(max-width:750px){header h1{font-size:28px}.grid,.calculator{grid-template-columns:1fr}main{padding:12px}section{padding:18px}nav{position:static;font-size:13px;gap:12px}}@media print{nav,.filters,.download,button{display:none}body{background:white}header{padding:20px;background:white;color:#19334a}header p,header small{color:#657789}section{break-inside:avoid;padding:15px;margin:10px 0}.figure{overflow:visible}svg{min-width:0}.table-scroll{overflow:visible}th{position:static}details{display:block}}'''
parts=[f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>K4 方法图谱与加速边界</title><style>{css}</style></head><body><header><div class="eyebrow">K4 / METHODS & COMPUTATIONAL STRUCTURE</div><h1>从动作观测到可信重建<br>方法图谱与批处理空间</h1><p class="lead">当前收益有限，是因为加速主要覆盖了模型计算，整体求解仍保留CPU控制、线性子问题和连续状态依赖。下一步的重点应从单次FK转向完整迭代与跨任务批处理。</p><small>证据快照：{now} · 论文方法图风格的矢量示意 · 图中不使用虚构运动轨迹 · 可离线阅读</small></header><nav><a href="#overview">结论</a><a href="#pipeline">方法流程</a><a href="#candidates">候选家族</a><a href="#batch">并行边界</a><a href="#execution">CPU/GPU</a><a href="#export">导出</a><a href="#matrix">完成度矩阵</a><a href="#catalogue">全部24列</a><a href="#priorities">下一步重点</a></nav><main>']
parts.append('''<section id="overview"><h2>收益已经出现，工程覆盖还未到主要瓶颈</h2><div class="grid"><div class="card"><strong>约6.8×</strong><p>32帧固定模型FK请求的局部提速：34.20→5.07 ms。CUDA Graph与坐标批变换已接入。</p></div><div class="card"><strong>13.7% 减时</strong><p>000 Kimodo同候选150帧：76.57→66.04 s。轻量传输与按需回读的组合结果。</p></div><div class="card"><strong>26 / 40</strong><p>CPU算法一致性完整面板通过数。性能改善和算法一致性分别验收，默认后端尚未替换。</p></div></div><p>这段开发已经建立了算法契约、独立验收、模型计算后端与诊断工具，但<strong>当前加速路径仍是CPU控制求解、GPU计算模型的混合实现</strong>。因此局部内核加速与用户体验到的等待时间之间存在明显差距。</p><p>“全部方法”在本页指当前K4运行流程、冻结消融协议的24列配置，以及独立GPU工程后端。历史已回退实验作为背景保留；它们未合并进当前方法或充当改善证据。</p><div class="callout warn">当前最佳已完成局部性能条件约2.27帧/秒；约5秒动作需要66秒求解，仍是离线处理。这里66秒计量的是固定模型和既有种子之后的姿态细化求解，模型准备、基础IK、多候选、导出和独立验收还会增加总耗时。高吞吐与低延迟、离线与在线是不同目标。</div></section>''')
for ident,svg,cap in figs:
 parts.append(f'<section id="{ident}"><div class="figure">{svg}</div><div class="bar"><p class="caption">{cap}</p><a class="download" download href="assets/20261119_k4_method_atlas/{["01_pipeline","02_candidates","03_batch","04_execution","05_export","06_evidence"][[x[0] for x in figs].index(ident)]}.svg">下载矢量 SVG ↗</a></div>')
 if ident=='pipeline':parts.append('<p><strong>输入适配</strong>统一单位、轴、层级与时钟；ARDy、CHOIS、CLoSD、DART、Kimodo、PRIMAL六种旋转格式保留声明骨架及独立旋转，MotionLCM / OmniControl使用实际位置字段。<strong>几何标定</strong>确定固定长度、关节中心和标记对应；<strong>姿态求解</strong>在这些固定条件下寻找关节坐标。双髋中点root补偿、头颈附着以及后续FK共同产生最终输出。</p><p>统一骨长是评分端操作；它不会向运行时提供原动作的参考骨长。几何方向、头胸相对旋转、模型胸段旋转和腰椎坐标分开命名，避免将整体方向达标解释为内部解剖角准确。</p>')
 if ident=='candidates':parts.append('<p>IK（逆运动学）从目标节点位置反求关节坐标；FK完成反向计算。多初值降低陷入局部解的风险，正反遍历改变连续状态的起点。整段求解同时优化整个窗口的关节轨迹，利用相邻帧误差变化约束连续性。GN是高斯–牛顿近似；TRF是有界信赖域方法；LSMR用于稀疏线性最小二乘。</p><p>输入驱动的固定几何与扩展坐标域是不同因素；后者使用独立运动学配置、逐项核验耦合，不授予肌肉与受力资格。HYOID仍保持现状。</p>')
 if ident=='batch':parts.append('<div class="callout"><strong>有顺序依赖 ≠ 必须留在CPU。</strong>一条序列的前态传播需要保持；不同序列、不同候选可以同时推进，设备端也能执行有顺序的迭代。当前未完成的是保持这些算法行为的批控制工程。</div><p>同一迭代的有限差分列可以同时计算；下一次迭代依赖本次更新。整段轨迹给定当前q后，所有帧的FK和局部残差可批量计算，但求解步受相邻帧耦合。直接把每帧拆成独立优化会改变问题，应作为新算法条件验证。</p>')
 if ident=='execution':parts.append('<p>当前两个既有环境通过常驻设备进程交换数值，避免修改正在运行的CPU环境。轻量传输减少打包/解包，按需回读减少输出量。下一层改进需要将残差、雅可比、线性子问题和接受判断放在同一设备循环中，同时保留CPU算法的边界、预算、停止与回退语义。</p><p>历史独立Torch GN已经有实现，属于另一种求解算法；它与本轮“CPU算法一致性迁移”分别命名和验收。自动微分、更少初值或更小计算预算也应单列，而不能算作等价执行的免费加速。</p>')
 if ident=='evidence':parts.append('''<div class="calculator"><div><h3>瓶颈上界演示：Amdahl公式</h3><p>若可加速部分占时间p、该部分加速s倍，忽略新增开销时，总加速比为 1 / [(1−p) + p/s]。</p><label>可加速占比 p：<input id="fraction" type="range" min="5" max="95" value="20"><output id="fractionValue">20%</output></label><br><label>局部加速 s：<input id="speed" type="range" min="1" max="20" value="7"><output id="speedValue">7×</output></label></div><div><h3 id="amdahl">总加速 1.21×</h3><p id="saving">耗时减少17.1%</p><small>这是可调的理论示意，默认p=20%为假设。9项整段任务的83.2%仅用于对应线性子问题分析，不能推广为全部40项或所有K路径的时间比例。</small></div></div>''')
 parts.append('</section>')
parts.append('<section id="matrix"><h2>哪些能加速、能Batch，哪些做了？</h2><p>状态描述当前工程实现与证据范围。绿色“已接入”也不代表整套GPU默认资格通过。筛选只改变显示，完整清单保留17个环节。</p><div class="filters"><button class="active" data-filter="all">全部环节</button><button data-filter="done">已接入</button><button data-filter="partial">部分覆盖</button><button data-filter="local">局部实验</button><button data-filter="todo">待工程化</button></div><div class="table-scroll"><table id="statusTable"><thead><tr><th>环节</th><th>并行空间</th><th>当前状态</th><th>已经做了</th><th>剩余工作 / 边界</th></tr></thead><tbody>')
for name,space,status,done,todo,src,state in rows:parts.append(f'<tr data-state="{state}"><td>{name}<br><small>{src}</small></td><td>{space}</td><td><span class="badge {state}">{status}</span></td><td>{done}</td><td>{todo}</td></tr>')
parts.append('</tbody></table></div></section>')
parts.append('<section id="catalogue"><h2>当前全部24列候选：逐项可追溯</h2><p>13个原生配置中，<code>forward_full_path</code>与<code>reverse_full_path</code>自带协调导出；其余11个另有<code>__coordinated</code>变体，共24列。每列适用分母由冻结协议逐路统计；合计9512个适用单元（包含已证明等价项）。这是算法消融列数，40项GPU面板尚未覆盖全部9512单元。</p><div class="table-scroll"><table><thead><tr><th>候选列</th><th>方法变化</th><th>适用路数 / 656</th><th>配置证据</th></tr></thead><tbody>')
for name,t in protocol['templates'].items():
 base=t['base'];isexp=t['kind']=='export';counts={}
 for route in protocol['routes']:
  st=route['cells'][name]['status'];counts[st]=counts.get(st,0)+1
 applicable=sum(v for k,v in counts.items() if k!='inapplicable')
 # exact status counts shown to avoid silently equating unknown statuses.
 n=sum(v for k,v in counts.items() if k in ('applicable','equivalent'))
 applicable=sum(1 for route in protocol['routes'] if route['cells'][name]['status']!='inapplicable')
 description=('复用 '+base+' 的原生拟合；增加固定锚点协调导出。' if isexp else desc[base])
 parts.append(f'<tr><td><code>{name}</code></td><td>{description}</td><td>{applicable}<br><small>{esc(str(counts))}</small></td><td><details><summary>完整冻结配置</summary><pre>{esc(json.dumps(t,ensure_ascii=False,indent=2))}</pre></details></td></tr>')
parts.append('</tbody></table></div><p>不适用与未通过分别记账。完整消融要求每个适用候选有独立验收；候选自身通过率、至少一个候选通过的覆盖与运行时选择器效果分别汇报。</p></section>')
parts.append('''<section id="priorities"><h2>接下来值得优先投入的工作</h2><table><thead><tr><th>优先级</th><th>工程方向</th><th>预期作用与验收</th></tr></thead><tbody><tr><td>1 · 整段路径</td><td>稀疏线性子问题与残差/雅可比设备驻留</td><td>直面已测9项任务约83.2%的线性成本。保持原边界与停止规则，先验证单次实际线性化，再验证全轨迹。收益仍需实测。</td></tr><tr><td>1 · 逐帧路径</td><td>异构固定几何 + 跨序列动态Batch</td><td>把许多小请求合成设备工作批；每条序列保持自己的前态。40项中34个不同固定模型使“只合相同模型”覆盖有限。</td></tr><tr><td>2 · 迭代控制</td><td>GPU端收敛掩码、信赖域与接受步骤</td><td>减少往返和CPU控制开销。先保留算法，再比较延迟、吞吐与差异传播；提前结束样本冻结。</td></tr><tr><td>2 · 导出和评分</td><td>协调路径批处理、批评分、复用同一原生解</td><td>减少验证轮次总耗时；分别统计求解耗时、导出耗时、验收耗时，避免将报告提速算成算法提速。</td></tr><tr><td>持续</td><td>完成一致性诊断和全路径资格</td><td>26/40只是当前通过范围；010/015精确状态对照用于区分单帧偏差与传播。保持CPU原结果和完整分母。</td></tr></tbody></table><p>这些方向可以并行推进研发，但当前GPU进程和CPU实验仍按既有资源规则运行。本次只制作图谱与证据分析，没有调整实验、减少候选或改变评分阈值。</p></section>''')
parts.append('<section id="sources"><h2>证据与版本</h2><p>本文图示描述已核查源码与冻结配置。性能数字为明确局部条件的测量；公式演示与后续工程方向分别标注。GPU性能面板在本次取数时完成 '+str(perf['evaluated'])+'/10；000 Kimodo四项局部条件已经完成。</p><p><a href="20261119_k4_gpu_status_report.html">一致性与速度详细报告</a> · <a href="20261119_k4_gpu_diagnostics.html">后台诊断动态进度</a> · <a href="20261118_k4_candidate_ablation.html">全部候选消融</a></p><details><summary>本次核查来源及内容哈希</summary><pre>'+esc(json.dumps(manifest['sources'],ensure_ascii=False,indent=2))+'</pre></details><p>主CPU候选协议：dfe8239；40项GPU面板：117cb84；性能条件：00a9e65；精确状态诊断：9c9b3ea。早期note中的严格初值选择判据由用户后续批准的输出一致性口径更新；历史结果保留。</p></section>')
parts.append('''</main><footer>K4方法图谱 · 所有图形与页面样式内嵌，断网仍可阅读；独立SVG用于论文排版与审阅。方法概念图不等于已通过的实验结论。</footer><script>
const buttons=document.querySelectorAll('[data-filter]');buttons.forEach(b=>b.addEventListener('click',()=>{buttons.forEach(x=>x.classList.toggle('active',x===b));document.querySelectorAll('#statusTable tbody tr').forEach(r=>r.hidden=b.dataset.filter!=='all'&&r.dataset.state!==b.dataset.filter)}));
function calc(){const p=Number(document.getElementById('fraction').value)/100,s=Number(document.getElementById('speed').value),t=1-p+p/s;document.getElementById('fractionValue').textContent=Math.round(p*100)+'%';document.getElementById('speedValue').textContent=s+'×';document.getElementById('amdahl').textContent='理论总加速 '+(1/t).toFixed(2)+'×';document.getElementById('saving').textContent='理论耗时减少 '+((1-t)*100).toFixed(1)+'%';}document.getElementById('fraction').addEventListener('input',calc);document.getElementById('speed').addEventListener('input',calc);calc();
</script></body></html>''')
(R/'docs/20261119_k4_method_atlas.html').write_text(''.join(parts))
print('wrote HTML',len(''.join(parts)),'chars; figures',len(figs),'states',len(rows),'candidates',len(protocol['templates']),'perf',perf['evaluated'])
