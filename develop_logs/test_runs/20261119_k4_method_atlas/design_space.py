from pathlib import Path
import json,hashlib,html,re,math
R=Path('/home/heinrich/MorionGen');B=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k4_20261117');O=R/'docs/assets/20261119_k4_design_space';O.mkdir(exist_ok=True)
protocol_path=B/'full_candidate_ablation_20261118_v1/protocol.json';protocol=json.loads(protocol_path.read_text());T=protocol['templates']
# Presentation order is independent of the frozen experimental protocol.
DISPLAY_GROUPS = [
 ('共同基准', ['balanced_targets']),
 ('固定几何与可行域', ['rig_centres','rig_arm_domain_centres','position_chain','position_fixed_foot_chain']),
 ('目标函数权衡', ['pelvis_continuity']),
 ('遍历、连续性与整段求解', ['rig_reverse_leg_path','rig_reverse_rate_leg_path','position_trajectory','position_trajectory_chart','position_total_proxy']),
 ('全身协调导出组合', ['forward_full_path','reverse_full_path']),
]
original_templates = T
original_ids = {n:f'C{i:02}' for i,n in enumerate((n for n,t in T.items() if t['kind']=='native'),1)}
display_names = [n for _,names in DISPLAY_GROUPS for b in names for n in (b,b+'__coordinated') if n in T]
assert len(display_names)==24 and set(display_names)==set(T)
T = {n:T[n] for n in display_names}
group_for = {n:g for g,names in DISPLAY_GROUPS for n in names}
# Keep the previous cards available as secondary evidence, not as the method explanation.
page=R/'docs/20261119_k4_method_atlas.html';archive=R/'docs/20261119_k4_candidate_cards.html'
if not archive.exists():archive.write_bytes(page.read_bytes())
# Keep the optional per-candidate appendix in the same reading order.
archive_text=archive.read_text()
card_pattern=r'<section class="method" id="([^"]+)" data-kind="[^"]+">.*?</section>'
card_matches=list(re.finditer(card_pattern,archive_text,re.S))
assert len(card_matches)==24
cards={m.group(1):m.group(0) for m in card_matches}
archive_text=archive_text[:card_matches[0].start()]+''.join(cards[n] for n in display_names)+archive_text[card_matches[-1].end():]
archive_text=re.sub(r'(<select[^>]*id="jump"[^>]*>).*?(</select>)',lambda m:m.group(1)+'<option value="">选择具体方案</option>'+''.join(f'<optgroup label="{g}">'+''.join(f'<option value="{n}">{n}</option>' for b in ns for n in (b,b+'__coordinated') if n in T)+'</optgroup>' for g,ns in DISPLAY_GROUPS)+m.group(2),archive_text,flags=re.S)
archive.write_text(archive_text)

C={'ink':'#173247','blue':'#2d6197','orange':'#bd7830','teal':'#167c77','purple':'#7a65a0','red':'#aa596c','gray':'#738698','line':'#cbd8e2'}
esc=html.escape
s=['''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1640 1080" role="img" aria-labelledby="spaceTitle spaceDesc"><title id="spaceTitle">K4 comparative method design space</title><desc id="spaceDesc">共同计算链只画一次。模型表示、优化目标、时序求解和输出路径四个维度展开选择，固定候选通过下方矩阵映射；选择一个候选时高亮其方法坐标。</desc><defs><marker id="arrowSpace" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10Z" fill="#738698"/></marker></defs><rect width="1640" height="1080" fill="white"/><style>text{font-family:Arial,"Noto Sans CJK SC",sans-serif}path{stroke-linecap:round;stroke-linejoin:round}.choice{transition:opacity .15s}.choice.active rect{stroke-width:3}.space-selected .choice:not(.active){opacity:.22}</style>''']
def text(x,y,t,size=18,c='ink',bold=False,anchor='start'):
 s.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{C.get(c,c)}" font-weight="{700 if bold else 400}" text-anchor="{anchor}">{esc(str(t))}</text>')
def line(x,y,X,Y,c='gray',w=1.5,dash=False,arrow=False):s.append(f'<path d="M{x} {y} L{X} {Y}" fill="none" stroke="{C.get(c,c)}" stroke-width="{w}" '+('stroke-dasharray="5 5" ' if dash else '')+('marker-end="url(#arrowSpace)"' if arrow else '')+'/>')
def rect(x,y,w,h,c='line',fill='white',rx=5):s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" stroke="{C.get(c,c)}" fill="{fill}"/>')
def path(d,c='gray',w=1.5,dash=False):s.append(f'<path d="{d}" fill="none" stroke="{C.get(c,c)}" stroke-width="{w}" '+('stroke-dasharray="5 5"' if dash else '')+'/>')
def dot(x,y,r=4,c='blue',fill='white'):s.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{C.get(c,c)}" stroke-width="1.5"/>')
def human(x,y,scale=.5,c='blue',lean=0,model=False):
 q={'p':(0,0),'lh':(-18,0),'rh':(18,0),'t':(lean,-70),'ls':(-35+lean,-70),'rs':(35+lean,-70),'n':(lean,-90),'h':(lean+3,-110),'le':(-50+lean,-33),'re':(57+lean,-44),'lw':(-52+lean,5),'rw':(80+lean,-64),'lk':(-9,48),'rk':(35,47),'la':(-23,99),'ra':(21,99),'lf':(-3,101),'rf':(41,101)}
 q={n:(x+scale*a,y+scale*b) for n,(a,b) in q.items()}
 for a,b in [('lh','rh'),('p','t'),('t','ls'),('t','rs'),('t','n'),('n','h'),('ls','le'),('le','lw'),('rs','re'),('re','rw'),('lh','lk'),('lk','la'),('la','lf'),('rh','rk'),('rk','ra'),('ra','rf')]:line(*q[a],*q[b],c,4 if model else 1.6)
 for a,b in q.values():dot(a,b,2.5,c)
 dot(*q['h'],6,c);return q
nodes={}
def option(key,x,y,w,title,c='blue',sub=None):
 h=55 if sub else 34;s.append(f'<g class="choice" data-option="{key}">');rect(x,y,w,h,c,{'blue':'#f0f5fa','orange':'#fff7ec','teal':'#edf7f4','purple':'#f5f2fa'}.get(c,'white'));text(x+10,y+22,title,16,c,True)
 if sub:text(x+10,y+44,sub,14,'gray')
 s.append('</g>');nodes[key]=(x+w/2,y+h/2)
text(26,39,'K4  |  One reconstruction framework, four axes of method variation',29,bold=True)
text(26,72,'共同计算链 × 固定候选的设计坐标。× 表示同组择一；⊕ 表示附加步骤。不是任意笛卡尔积，适用组合由输入契约限定。',18,'gray')
# Shared backbone visually grounded in human motion / model / trajectory / FK.
for x,title in [(37,'D：异构观测'),(321,'输入驱动固定模型'),(671,'原生轨迹求解'),(1101,'FK / root / 头颈'),(1434,'输出与支持范围')]:text(x,114,title,18,bold=True)
for i in range(3):human(59+i*69,183,.37,'orange',i*7)
human(391,183,.48,'blue',0,True);text(444,171,'θₖ, Qθₖ',22,'blue',True);text(444,202,'片段内固定',16,'gray')
for i in range(4):
 dot(697+i*75,175,15,'blue');text(697+i*75,181,'q'+str(i),15,anchor='middle')
 if i<3:line(712+i*75,175,757+i*75,175,arrow=True)
text(678,220,'IK seed → constrained refinement',16,'gray')
human(1150,183,.43,'teal',6,True);text(1203,174,'Fₖ = FKθₖ(Qₖ)',20,'teal',True);text(1203,204,'原生 / 导出分列',16,'gray')
text(1434,165,'位置 · 旋转 · 时序',18,'teal',True);text(1434,195,'来源 · 掩码 · 回退',17,'gray');text(1434,224,'候选与选择器分开验收',15,'gray')
for x,X in [(244,303),(597,654),(994,1081),(1375,1418)]:line(x,177,X,177,w=2,arrow=True)
line(26,254,1614,254,'line',1.3)
# Four true intervention families; all choice groups get their own symbols.
for x,c,title,sub in [(30,'blue','I  模型表示 θ 与可行域 Qθ','改变能否表达目标姿态'),(441,'orange','II  优化目标 ℒ','改变误差之间的权衡'),(841,'teal','III  求解与时序 S','改变搜索路径与时间耦合'),(1242,'purple','IV  导出路径 Π','改变原生采样点之间的运动')]:
 text(x,292,title,24,c,True);text(x,322,sub,17,'gray')
for x in [420,821,1222]:line(x,276,x,916,'line',1)
# model: four independent groups; asset means source of centres, not absence of scaling.
text(31,360,'×  关节中心',17,bold=True)
option('center_asset',31,374,176,'缩放资产中心','blue');option('center_rig',217,374,178,'声明rig中心','blue')
text(31,444,'×  躯干长度 / 稳定性门控',17,bold=True)
option('trunk_prior',31,458,364,'模型先验','blue','长度由输入/资产标定；持续弯曲保留假设')
option('trunk_chain',31,520,176,'链长代理','blue','逐段稳定 ≤10%')
option('trunk_total',217,520,178,'总链门控','blue','前三段+总链稳定')
text(31,610,'×  足端固定几何',17,bold=True)
option('foot_asset',31,624,115,'缩放资产','blue');option('foot_rig',153,624,115,'rig优先¹','blue');option('foot_polar',275,624,120,'极角包络','blue')
# skeletal schematic showing fixed registration, no extra stage
q=human(95,746,.44,'blue',0,True)
for n in ['ls','rs','lf']:
 x,y=q[n];dot(x+9,y-5,3,'orange');line(x+9,y-5,x,y,'orange',1,arrow=True)
text(150,711,'×  坐标域',17,bold=True)
option('domain_base',150,724,245,'U0：共同K4独立运动学域','blue')
option('domain_foot',150,765,245,'U1：U0 + 足踝扩域','blue')
option('domain_arm',150,806,245,'U2：U1 + 上臂轴向扩域','blue')
text(31,883,'¹ 无rig的对应分支保留资产足几何。',15,'gray')
text(31,906,'总链分支沿用同一长度估计器。',15,'gray')
# Objective options: geometry vectors/hinge illustration as one conceptual unit
text(441,360,'×  基础目标',17,bold=True)
option('loss_balanced',441,374,355,'Balanced observation targets','orange','方向 + 位置 + 几何角 + 超额hinge')
option('loss_pelvis',441,437,355,'Pelvis emphasis','orange','骨盆线×2；关闭几何角与目标hinge')
# vector/cost motif
line(480,565,558,524,'blue',2.5,arrow=True);line(480,565,565,574,'orange',2.5,arrow=True);path('M526 542 Q538 557 530 569','gray',1.3)
text(446,602,'rdir ∝ u − uᴰ',19,bold=True)
line(645,578,780,578);line(645,578,645,515);path('M645 578 L704 578 L777 523','orange',2.4);text(651,602,'target hinge',17,'gray')
text(441,652,'共同项：位置在双髋中心坐标系计算',17)
text(441,681,'前态坐标正则 + 方向误差变化项',17)
text(441,730,'⊕  新增方向变化率超额惩罚',17,bold=True)
option('rate_off',441,746,104,'关闭','orange');option('rate_on',555,746,241,'开启：权重1 / margin 0.8','orange')
text(441,813,'Δe = (u−uᴰ)ₜ − (u−uᴰ)ₜ₋₁',19,bold=True)
text(441,851,'约束新增变化，不直接压低动作速度。',16,'gray')
text(441,883,'权重改变最优解；不是执行后端优化。',16,'gray')
# solve actual dependencies not conflating rechart prior to seed
text(841,360,'×  原生逐帧遍历',17,bold=True)
option('traverse_forward',841,374,173,'正向  q₀ → qT','teal');option('traverse_reverse',1024,374,173,'反向  qT → q₀','teal')
for j in range(4):
 x=872+j*94;dot(x,463,14,'teal');text(x,469,'q'+str(j),14,anchor='middle');rect(x-6,423,12,12,'orange','#fff7ec');line(x,435,x,449)
 if j<3:line(x+14,463,x+80,463,arrow=True)
text(842,506,'nominal / previous / 条件触发膝种子',16,'gray')
text(841,552,'⊕  整段联合优化：必须先有原生种子',17,bold=True)
option('traj_off',841,568,173,'无整段步骤','teal');option('traj_on',1024,568,173,'有界GN + LSMR','teal')
text(841,632,'⊕  整段前的等价坐标图',17,bold=True)
option('chart_off',841,648,173,'保留种子坐标','teal');option('chart_on',1024,648,173,'最大边界余量','teal')
for i in range(5):
 for j in range(5):rect(847+j*25,721+i*20,19,14,'line','#6fbab3' if i==j or i==j+1 else '#f5f8fa')
text(1001,748,'diag(Jφ)',19,'teal',True);text(1001,781,'相邻误差耦合 LJφ',17)
text(841,850,'整段前扩上臂域，检查种子FK不变。',16,'gray')
text(841,883,'反向/联合改变求解路径与时序依赖。',16,'gray')
# export alternatives shown once for all base configurations
text(1242,360,'共同起点：原生锚点 + SO(3)插值',17,bold=True)
for x,y in [(1281,445),(1392,417),(1565,451)]:dot(x,y,6,'blue')
path('M1281 445 Q1335 396 1392 417 T1565 451','purple',2.2);path('M1281 445 Q1405 494 1565 451','gray',1.5,True)
text(1242,501,'×  最终插值策略',17,bold=True)
option('path_so3',1242,518,365,'E0：SO(3) → FK','purple','不追加坐标协调')
option('path_leg',1242,581,365,'EL：固定锚点 + 腿部协调','purple','只优化12个腿部角坐标')
option('path_full',1242,644,365,'EA：固定锚点 + 全身协调','purple','所有可动角坐标；原生坐标逐值保持')
line(1280,732,1577,732,'gray',1.4)
for x in [1280,1428,1577]:dot(x,732,6,'blue')
for x in [1317,1354,1391,1465,1502,1539]:dot(x,732,3,'purple')
text(1242,779,'__coordinated：E0 或 EL → EA',20,'purple',True)
text(1242,819,'保持 θ、Qnative；只替换输出路径。',17)
text(1242,852,'已有EA的两条方案不再重复增加变体。',16,'gray')
text(1242,883,'插值假设不增加观测支持范围。',16,'gray')
# field and computation boundary legible at bottom
line(26,932,1614,932,'line',1.2)
text(30,966,'候选 k = (θ策略, 坐标域, 目标, 遍历, 整段/坐标图, 导出路径)',22,bold=True)
text(30,997,'图中是可解释的变化维度；实际只执行下方13个已登记原生组合及11个导出变体。',18,'gray')
rect(27,1017,1587,42,'line','#f3f7fa');text(43,1045,'独立验收：D−R / F−D / F−R · 所有候选完整分母       执行优化：FK/扰动批已接入；GPU求解控制、异构Batch仍待完成',18)
s.append('</svg>');svg=''.join(s);(O/'method_design_space.svg').write_text(svg)
# Exact candidate coordinates derived from the frozen configurations, not hand-matched by motion IDs.
native=[(name,t) for name,t in T.items() if t['kind']=='native'];records=[];mapping={}
def features(t):
 c=t['config'];r=c['refinement'];traj=c.get('trajectory_refinement');dom=c['kinematic_domain']['coordinate_ranges_deg']
 center='rig' if c.get('joint_centres')=='declared_rig' else 'asset'
 trunk={'model_prior':'prior','observed_chain_proxy_v1':'chain','observed_chain_total_proxy_v1':'total'}[c.get('position_trunk_strategy','model_prior')]
 foot='polar' if c.get('position_foot_geometry') else 'rig' if c.get('foot_geometry')=='declared_rig' else 'asset'
 domain='arm' if 'arm_rot_l' in dom else 'foot' if 'ankle_angle_l' in dom else 'base'
 loss='pelvis' if r.get('pelvis_line_weight')==2 else 'balanced';rate='on' if r.get('direction_rate_violation_weight',0) else 'off';traverse=r.get('traversal','forward');joint='on' if traj else 'off';chart='on' if traj and traj.get('equivalent_seed_chart') else 'off'
 policy=t.get('policy') if t['kind']=='export' else c.get('coordinated_path')
 path_='so3' if not policy else 'leg' if policy['refinement'].get('active_coordinates') else 'full'
 result=dict(center=center,trunk=trunk,foot=foot,domain=domain,loss=loss,rate=rate,traverse=traverse,traj=joint,chart=chart,path=path_)
 result['nodes']=[f'center_{center}',f'trunk_{trunk}',f'foot_{foot}',f'domain_{domain}',f'loss_{loss}',f'rate_{rate}',f'traverse_{traverse}',f'traj_{joint}',f'chart_{chart}',f'path_{path_}']
 result['name']=next(n for n,v in T.items() if v is t)
 return result
for n,t in T.items():mapping[n]=features(t)
assert all(all(k in nodes for k in v['nodes']) for v in mapping.values())
for i,(name,t) in enumerate(native,1):
 row=dict(mapping[name],id=original_ids[name],group=group_for[name],variant=name+'__coordinated' if name+'__coordinated' in T else None)
 row['eligible']=sum(r['cells'][name]['status']!='inapplicable' for r in protocol['routes']);records.append(row)
assert len(mapping)==24 and len(records)==13 and sum(bool(r['variant']) for r in records)==11
labels={'center':{'asset':'资产','rig':'rig'},'trunk':{'prior':'先验','chain':'链长','total':'总链门控'},'foot':{'asset':'资产','rig':'rig优先','polar':'包络估计'},'domain':{'base':'U0','foot':'U1','arm':'U2'},'loss':{'pelvis':'骨盆强调','balanced':'Balanced'},'traverse':{'forward':'→ 正向','reverse':'← 反向'},'path':{'so3':'E0','leg':'EL','full':'EA'}}
CSS='''*{box-sizing:border-box}body{margin:0;color:#173247;background:#f2f5f8;font:16px/1.75 system-ui,"Noto Sans CJK SC",sans-serif}header{background:#193549;color:white;padding:28px max(24px,calc((100vw - 1580px)/2))}h1{font-size:32px;margin:4px 0}header p{color:#d1e0eb;margin:10px 0;max-width:1280px}main{max-width:1640px;padding:20px;margin:auto}section{background:white;padding:24px;border:1px solid #dbe5ed;border-radius:8px;margin-bottom:22px;scroll-margin-top:78px}h2{font-size:24px;margin:0 0 14px}h3{font-size:18px;margin:8px 0}nav{display:flex;align-items:center;gap:14px;flex-wrap:wrap;background:white;position:sticky;top:0;z-index:5;padding:10px 24px;border-bottom:1px solid #dbe5ed}a{color:#2d6197}button,select{padding:7px 10px;color:#173247;background:white;border:1px solid #b4c7d5;border-radius:4px;font-size:14px;cursor:pointer}button.active{background:#193549;color:white}#figure svg{width:100%;display:block}#figure{overflow:auto}.muted,.caption{color:#6c8193;font-size:14px}.selection{padding:14px 18px;border-left:4px solid #167c77;background:#edf7f4;margin-top:14px}.legend{display:flex;gap:20px;flex-wrap:wrap}.legend strong{color:#2d6197}.table-scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th{background:#eef3f7;text-align:left}th,td{padding:9px 8px;border-bottom:1px solid #dde5ec;vertical-align:middle}td button{white-space:nowrap}.method-group th{padding-top:18px;color:#2d6197;font-size:15px}tr[id]{scroll-margin-top:90px}tr.selected{background:#edf7f4}.code{font-family:ui-monospace,monospace;font-size:12px;overflow-wrap:anywhere}.opt{font-size:12px;display:inline-block;border-radius:3px;padding:2px 5px;background:#edf3f9}.three{display:grid;grid-template-columns:repeat(3,1fr);gap:20px}.three>div{border-left:3px solid #2d6197;padding:0 16px}.callout{padding:17px;background:#fff7ec;border-left:3px solid #bd7830}details{padding:12px 0;border-top:1px solid #dde5ec}summary{cursor:pointer;font-weight:600}pre{padding:14px;background:#f4f7fa;font-size:12px;max-height:330px;overflow:auto;white-space:pre-wrap}.num{font-variant-numeric:tabular-nums}footer{padding:22px;color:#6c8193;font-size:13px}@media(max-width:800px){.three{grid-template-columns:1fr}nav{position:static}main{padding:10px}section{padding:14px}#figure svg{min-width:1050px}}@media print{nav,button,select,.tools{display:none}body{background:white}header{background:white;color:#173247}header p{color:#6c8193}section{break-inside:avoid;border:0}#figure svg{min-width:0}#figure,.table-scroll{overflow:visible}.three{display:block}}'''
p=[f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>K4备选方法 · 统一比较图</title><style>{CSS}</style></head><body><header id="method"><h1>K4备选方法 · 统一比较图</h1><p>共同计算只画一次；在模型表示、目标、时序求解和导出路径四个分歧位置展开设计。全部候选映射到同一张图，互斥与叠加关系可直接比较。</p></header><nav><a href="#design">方法设计空间</a><a href="#matrix">全部候选映射</a><a href="#interpretation">方法取舍</a><a href="#compute">加速与Batch</a><button id="reset">显示全部设计空间</button><select id="choose" aria-label="高亮候选方案"><option value="">高亮一个候选的路径</option>']
for group,names in DISPLAY_GROUPS:
 p.append(f'<optgroup label="{group}">')
 for base in names:
  for n in (base,base+'__coordinated'):
   if n in T:p.append(f'<option value="{n}">{"↳ 导出变体 · " if n!=base else ""}{n}</option>')
 p.append('</optgroup>')
p.append('</select></nav><main><section id="design"><div id="figure">'+svg+'</div><div class="selection" id="selection" aria-live="polite">当前显示全部设计空间。点击下方候选名，或使用上方选择器，高亮该候选采用的选项；其他路径会淡出。</div><p class="caption">骨架、向量、误差曲线和矩阵均为方法示意。选中路径来自冻结配置字段解析，不根据动作编号或实验得分决定。颜色区分方法维度，不表示性能排名。</p><div class="tools"><a href="assets/20261119_k4_design_space/method_design_space.svg" download>下载统一方法图 SVG</a> · <a href="assets/20261119_k4_design_space/method_design_space.png">查看完整PNG</a></div></section>')
p.append('<section id="matrix"><h2>13个原生组合 + 11个导出变体，落在同一组方法坐标上</h2><p>阅读顺序：共同基准 → 固定几何与可行域 → 目标权衡 → 时序求解 → 全身导出组合。每个导出变体紧随所属原生方案。C编号保留原登记标识；排序表达方法关系，不表示性能名次或严格单因素递进。表中每一行是一组原生参数。右侧的导出变体保持这一行的模型与原生轨迹，只将E0或EL改为EA。点击候选名可在上图中定位。</p><div class="table-scroll"><table><thead><tr><th>候选 / 适用路数</th><th>中心</th><th>躯干</th><th>足端</th><th>坐标域</th><th>目标</th><th>率超额</th><th>遍历</th><th>整段 / 坐标图</th><th>原配置导出</th><th>导出变体</th></tr></thead><tbody>')
last_group=None
for r in records:
 if r['group'] != last_group:
  p.append(f'<tr class="method-group"><th colspan="11">{r["group"]}</th></tr>');last_group=r['group']
 n=r['name'];p.append(f'<tr id="{n}" data-base="{n}"><td><button data-select="{n}">{r["id"]} · {n}</button><br><small>{r["eligible"]}/656路适用</small></td>')
 for key in ['center','trunk','foot','domain','loss']:p.append(f'<td>{labels[key][r[key]]}</td>')
 p.append(f'<td>{"⊕ 开" if r["rate"]=="on" else "—"}</td><td>{labels["traverse"][r["traverse"]]}</td><td>{("⊕ GN + 等价图" if r["chart"]=="on" else "⊕ GN") if r["traj"]=="on" else "—"}</td><td><span class="opt">{labels["path"][r["path"]]}</span></td>')
 p.append(f'<td><button id="{r["variant"]}" data-select="{r["variant"]}">{labels["path"][r["path"]]} → EA</button></td></tr>' if r['variant'] else '<td>已有EA</td></tr>')
p.append('</tbody></table></div><p class="caption">U0：共同K4独立运动学域。U1：在U0上扩展踝/距下范围至±90°。U2：U1再扩展上臂轴向范围至±180°。三条整段分支另在逐帧种子之后扩上臂域，并检查种子FK不变。</p><p class="caption">中心“资产”指中心位置策略，不代表完全未按输入缩放。足端“rig优先”在没有rig的分支保持资产几何。position_total_proxy与chart分支在160/164适用路上有已登记等价关系；仍保留独立方法定义与完整分母。</p><details><summary>查看当前选中方案的冻结配置</summary><pre id="config">请先选择一个候选。</pre></details></section>')
p.append('''<section id="interpretation"><h2>从分歧机制理解备选方案</h2><div class="three"><div><h3>几何可达性，与域内求解分开</h3><p>中心注册、躯干代理、固定足端和坐标域改变可表达的姿态集合。多初值与反向遍历改变域内搜索路径。前者若存在表示不足，后者再充分也无法消除模型残差。</p></div><div><h3>目标、前态和整段耦合是三种变化</h3><p>骨盆强调重分配损失；率超额约束拟合新增速度；反向遍历改变前态递推；整段GN把相邻误差联合求解。它们不应被统称为“连续性优化”。</p></div><div><h3>导出变体提供更干净的配对</h3><p>协调导出固定同一原生解，改变采样间FK路径。E0/EL→EA的比较可以定位导出因素；几何、域和目标同时改变的两行比较，需要进一步单因素消融才能归因。</p></div></div><div class="callout"><strong>已有组合是受输入契约约束的设计点。</strong> 这张图不建议将所有维度任意组合。例如声明中心依赖rig；位置链长策略受可辨识性和稳定性门控；等价坐标图在当前方案中用于整段求解前；导出协调不增加观测资格。</div><p>运行时候选选择只读D。独立评分对每个适用候选分别报告D−R、F−D、F−R，保留原始/统一骨长与原生/导出时钟。候选自身通过率、互补覆盖和选择器命中率分别记录。</p></section>''')
p.append('''<section id="compute"><h2>这套结构中，哪里可以加速或Batch？</h2><table><thead><tr><th>计算边界</th><th>可以保持算法的并行</th><th>已有实现 / 证据</th><th>剩余工程</th></tr></thead><tbody><tr><td>同一θ、同一次迭代</td><td>FK的多帧和有限差分扰动列</td><td>Torch批FK、扰动批、模型驻留、CUDA Graph已接入</td><td>完整r/J与接受决策尚未全部设备驻留</td></tr><tr><td>相互独立的候选 / 序列</td><td>各自保持前态，按就绪请求合批</td><td>CPU独立候选调度；相同模型跨序列仅有局部实验</td><td>异构θ的批FK、动态活跃样本和设备端TRF控制</td></tr><tr><td>整段GN</td><td>给定Q后全帧特征/导数；独立轨迹系统</td><td>批模型计算与CPU稀疏算子局部优化</td><td>有界线性子问题及稀疏结构的等价GPU实现</td></tr><tr><td>原生解 → 导出变体</td><td>复用原生拟合；独立区间/路径的模型评价</td><td>固定锚点、支持门控、原生结果复用</td><td>协调路径GPU批求解；现有前态初始化不能直接删除</td></tr><tr><td>固定几何 / 基础IK</td><td>独立模型/初值；哈希一致时复用</td><td>CPU校准、基础IK与回退、模型导出</td><td>基础IK和足端SLSQP没有完整GPU一致性路径</td></tr><tr><td>头颈 / root / 独立验收</td><td>身体姿态确定后的多帧计算与批评分</td><td>CPU完整语义；Torch批评分后端已有</td><td>头颈/导出整条GPU覆盖与独立回读资格仍待完成</td></tr></tbody></table><p><strong>必须保持的是依赖关系，不是CPU位置。</strong>单序列的连续前态与一次迭代的接受步骤有逻辑顺序，但可在设备上控制。删初值、删目标、换求解器或把耦合帧拆开属于算法变化，需独立验收。</p><p>本页解释方法结构与工程覆盖，不将局部加速解释为所有分支已替换CPU。当前CPU/GPU实测进展见<a href="20261119_k4_gpu_diagnostics.html">独立诊断报告</a>。</p></section>''')
p.append('<section><h2>证据与详细资料</h2><p>冻结模板来自dfe8239对应完整消融协议：24列、9512个适用单元。页面中的路径高亮与下表均由该协议解析。单方案原图仅作为附属材料保留：<a href="20261119_k4_candidate_cards.html">逐方案资料卡</a>。</p><details><summary>冻结配置全文与来源哈希</summary><pre>'+esc(json.dumps({'protocol_path':str(protocol_path),'sha256':hashlib.sha256(protocol_path.read_bytes()).hexdigest(),'templates':T},ensure_ascii=False,indent=2))+'</pre></details></section></main><footer>图示不代表实测运动或通过资格；未改变实验输入、模型、评分、数据划分或留出记录。SVG、映射数据与交互代码全部内嵌，可离线阅读。</footer>')
script='''const M=__MAP__,T=__TEMPLATES__; const root=document.querySelector('#figure svg');function show(name){const r=M[name];history.replaceState(null,'',r?'#'+name:location.pathname+location.search);document.getElementById('choose').value=r?name:'';root.classList.toggle('space-selected',!!r);document.querySelectorAll('[data-option]').forEach(g=>g.classList.toggle('active',!!r&&r.nodes.includes(g.dataset.option)));document.querySelectorAll('[data-base]').forEach(tr=>tr.classList.toggle('selected',!!r&&tr.dataset.base===T[name].base));document.querySelectorAll('[data-select]').forEach(b=>b.classList.toggle('active',b.dataset.select===name));document.getElementById('config').textContent=r?JSON.stringify(T[name],null,2):'请先选择一个候选。';document.getElementById('selection').textContent=r?name+'：'+r.nodes.map(k=>document.querySelector('[data-option="'+k+'"] text').textContent).join(' · ')+(T[name].kind==='export'?'。导出变体复用同一原生模型与轨迹，仅替换输出路径。':'。此为已登记的原生配置组合。'):'当前显示全部设计空间。点击下方候选名，或使用上方选择器，高亮该候选采用的选项；其他路径会淡出。';}document.getElementById('choose').addEventListener('change',e=>show(e.target.value));document.getElementById('reset').addEventListener('click',()=>show(''));document.querySelectorAll('[data-select]').forEach(b=>b.addEventListener('click',()=>{show(b.dataset.select);document.getElementById('design').scrollIntoView({block:'start'});}));function fromHash(){const n=decodeURIComponent(location.hash.slice(1));if(M[n])show(n);}window.addEventListener('hashchange',fromHash);fromHash();'''
p.append('<script>'+script.replace('__MAP__',json.dumps(mapping,ensure_ascii=False)).replace('__TEMPLATES__',json.dumps(T,ensure_ascii=False))+'</script></body></html>');page.write_text(''.join(p))
manifest={'protocol_sha256':hashlib.sha256(protocol_path.read_bytes()).hexdigest(),'mapping':mapping,'native_rows':records,'nodes':list(nodes),'display_groups':DISPLAY_GROUPS,'display_order':display_names,'stable_candidate_ids':original_ids,'native_count':13,'export_count':11}
(R/'develop_logs/test_runs/20261119_k4_method_atlas/design_space_mapping.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print('one comparative figure,',len(nodes),'choices,',len(mapping),'candidate mappings')
