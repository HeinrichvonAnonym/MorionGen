from pathlib import Path
import math,html,re
R=Path('/home/heinrich/MorionGen');O=R/'docs/assets/20261119_k4_method_atlas'
a=[]
COL={'ink':'#19344a','blue':'#285d94','teal':'#148b85','orange':'#d38b43','purple':'#8170a5','muted':'#718497','light':'#dae4eb','red':'#b06573'}
def esc(t):return html.escape(str(t))
def text(x,y,t,size=19,col='ink',bold=False,anchor='start'):
 a.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{COL.get(col,col)}" font-weight="{700 if bold else 400}" text-anchor="{anchor}">{esc(t)}</text>')
def line(x,y,X,Y,col='muted',w=1.5,dash=False,arrow=False):
 a.append(f'<path d="M{x},{y} L{X},{Y}" fill="none" stroke="{COL.get(col,col)}" stroke-width="{w}" '+('stroke-dasharray="5 5" ' if dash else '')+('marker-end="url(#head)" ' if arrow else '')+'/>')
def path(d,col='muted',w=1.5,dash=False,fill='none',arrow=False):
 a.append(f'<path d="{d}" fill="{fill}" stroke="{COL.get(col,col)}" stroke-width="{w}" '+('stroke-dasharray="5 5" ' if dash else '')+('marker-end="url(#head)" ' if arrow else '')+'/>')
def dot(x,y,r=4,col='blue',fill='white'):
 a.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{COL.get(col,col)}" stroke-width="1.6"/>')
def rect(x,y,w,h,col='light',fill='white',radius=5,dash=False):
 a.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{COL.get(col,col)}" '+('stroke-dasharray="5 5"' if dash else '')+'/>')
def axis(x,y,s=19):
 for dx,dy,c in [(s,2,'red'),(-s*.5,s*.4,'teal'),(0,-s,'blue')]:line(x,y,x+dx,y+dy,c,1.5,arrow=True)
def human(x,y,scale=1,t=0,col='blue',model=False,axes=False,ghost=False):
 # Fixed segment lengths, articulated 3D schematic with oblique orthographic projection.
 def add(p,v):return tuple(p[i]+v[i] for i in range(3))
 def limb(p,L,ang,side=0):return add(p,(L*math.sin(ang),side,-L*math.cos(ang)))
 b=.35*math.sin(t);p={'root':(0,0,0),'lh':(-15,0,0),'rh':(15,0,0)}
 p['chest']=(62*math.sin(b),0,62*math.cos(b));p['ls']=add(p['chest'],(-34,0,0));p['rs']=add(p['chest'],(34,0,0));p['neck']=add(p['chest'],(4,0,17));p['head']=add(p['neck'],(4*math.sin(t),0,21))
 p['le']=limb(p['ls'],42,-.3-.45*math.sin(t));p['lw']=limb(p['le'],40,-.15-.5*math.sin(t+.5));p['re']=limb(p['rs'],42,.6+.65*math.sin(t));p['rw']=limb(p['re'],40,.4+1.1*math.sin(t+.7))
 p['lk']=limb(p['lh'],57,.3+.4*math.sin(t));p['la']=limb(p['lk'],55,-.1-.35*math.sin(t));p['lf']=add(p['la'],(20,-6,0));p['rk']=limb(p['rh'],57,-.2+.3*math.sin(t));p['ra']=limb(p['rk'],55,-.1-.25*math.sin(t));p['rf']=add(p['ra'],(20,-6,0))
 q={n:(x+scale*(v[0]-.45*v[1]),y+scale*(-v[2]+.22*v[0]+.3*v[1])) for n,v in p.items()}
 edges=[('lh','rh'),('root','chest'),('chest','ls'),('chest','rs'),('chest','neck'),('neck','head'),('ls','le'),('le','lw'),('rs','re'),('re','rw'),('lh','lk'),('lk','la'),('la','lf'),('rh','rk'),('rk','ra'),('ra','rf')]
 for n,m in edges:
  if model:line(*q[n],*q[m],col,9*scale)
  else:line(*q[n],*q[m],col,2*scale,dash=ghost)
 for n in p:dot(*q[n],max(2,3.6*scale),col,'white')
 dot(*q['head'],10*scale,col,'#edf4f9' if model else 'white')
 if axes:
  for n in ['root','chest','rk']:axis(*q[n],20*scale)
 return q

a.append('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1680 1040" role="img" aria-labelledby="methodTitle methodDesc"><title id="methodTitle">K4 methodology: observation-conditioned multi-hypothesis kinematic reconstruction</title><desc id="methodDesc">方法总图：多格式动作观测经声明几何和可观测性约束，形成多候选整段关节轨迹；每个候选经模型FK、头颈及root处理、协调导出，分别保留结果。输入侧可选择整段候选，原动作在独立支路验收。GPU仅改变执行后端。</desc><defs><marker id="head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10Z" fill="#718497"/></marker></defs><rect width="1680" height="1040" fill="white"/><style>text{font-family:Arial,"Noto Sans CJK SC",sans-serif}path{stroke-linecap:round;stroke-linejoin:round}</style>')
text(32,43,'K4  |  Observation-conditioned multi-hypothesis kinematic reconstruction',29,bold=True)
text(32,74,'观测决定目标与支持范围；模型决定可达域；候选处理局部解与时序歧义；重建始终由模型FK生成。',20,col='muted')
# headings and aligned main strip
for x,num,name,sub in [(32,'A','Motion observations','异构输入 → 统一观测'),(350,'B','Articulated model','固定几何与可行域'),(660,'C','Hypothesis reconstruction','整段候选与输出路径'),(1160,'D','Input-side selection','候选保留 / 整段选择'),(1440,'E','Reconstructed motion','可追溯的重建结果')]:
 text(x,128,num,23,col='blue',bold=True);text(x,159,name,21,bold=True);text(x,187,sub,17,col='muted')
# A: a motion sequence, independent source channels + representation object
for i in range(3):human(75+83*i,325,.48,.25+i*.6,'orange')
line(43,400,281,400,'muted',1.5,arrow=True);text(253,425,'t',18,col='muted')
text(37,463,'P₁:T',23,col='orange',bold=True);text(111,462,'位置格式 × 2',17)
text(37,502,'P₁:T, R₁:T',22,col='orange',bold=True);text(182,501,'旋转格式 × 6',16)
for x in [51,89,127]:
 for j in range(3):dot(x,545+j*13,2.5,'orange')
text(151,560,'时钟 / 单位 / 轴 / 掩码',17)
rect(37,595,250,63,'orange',fill='#fff8f0');text(162,621,'D = {P, R可用, t, M, rig}',18,bold=True,anchor='middle');text(162,646,'已声明字段是拟合边界',16,col='muted',anchor='middle')
# B: model schematic pelvis/frames, fixed bones, geometry inferred sources
q=human(469,356,.89,0,'blue',True,True)
line(439,316,374,283,'blue',1);text(352,269,'关节中心',16,col='blue')
line(494,396,593,414,'blue',1);text(539,438,'固定骨长',16,col='blue')
path('M464 318 Q504 333 503 366','orange',2);text(527,357,'q ∈ Qθ',19,col='orange')
text(358,492,'θₖ = Calibrate(D, asset; cₖ)',20,bold=True)
text(358,523,'每个候选片段内固定 θₖ',17,col='muted')
text(358,552,'关节结构 + 范围 + 耦合',17)
text(358,583,'来源骨架 / 输入可辨识几何',16)
text(358,614,'独立头颈链附着胸段',17)
text(358,646,'欠观测分量保留未知',17,col='red')
line(291,625,341,625,'muted',2,arrow=True)
# C: candidate cards motion sequences + fit-prediction loop
for j,(label,col,phase) in enumerate([('c₁','blue',.15),('c₂','purple',.3),('cK','teal',.05)]):
 y=220+j*113
 rect(694,y,400,102,col,fill=['#f2f6fb','#f7f4fb','#f1f9f7'][j],radius=8)
 text(710,y+25,label,18,col=col,bold=True)
 for i in range(3):human(780+i*90,y+50,.31,phase+i*.65,col,True)
 line(756,y+89,1057,y+89,col,1,arrow=True)
 text(1098,y+58,'Fₖ',19,col=col,bold=True)
# Observation route directly supplies fitting targets, separately from geometry calibration.
path('M287 563 L317 563 L317 209 L671 209 L671 248','orange',1.4,dash=True,arrow=True)
text(401,202,'D : observation targets',14,col='orange')
# branching construction from fixed model
line(605,351,635,351,'muted',2);line(635,267,635,493,'muted',1.7)
for y in [271,384,497]:line(635,y,691,y,'muted',1.6,arrow=True)
text(684,575,'Qₖ = Solve(D; θₖ, ℒₖ, Sₖ)',23,bold=True)
text(684,604,'几何 / 可行域 / 目标 / 初值与遍历',17,col='muted')
# native to export mini FK path
for i,(x,y) in enumerate([(712,645),(819,630),(930,650),(1065,627)]):dot(x,y,5,'blue')
path('M712 645 C754 632 780 623 819 630 S890 655 930 650 S1017 630 1065 627','teal',2.5)
text(686,690,'FK + root / 头颈 → 原生锚点 → 协调导出',17)
text(686,716,'13原生配置 + 11导出变体；逐候选保留',16,col='muted')
# D: D-only selection structured quality matrix, no invented pass values
rect(1161,247,218,271,'light',fill='#fafbfd',radius=6)
text(1176,274,'qualityₖ(D, Fₖ)',18,bold=True)
for x,t in [(1190,'观测'),(1257,'位置'),(1324,'时序')]:text(x,312,t,15,anchor='middle')
for j,col in enumerate(['blue','purple','teal']):
 y=334+j*54
 for i in range(3):
  rect(1178+i*67,y,28,24,col,fill='white',radius=2)
  # symbolic greek quality, not fake measured scores
  text(1192+i*67,y+18,['ε','ε','δ'][i],16,col=col,anchor='middle')
text(1175,500,'整段排序 / 预算规则',16)
for y in [271,384,497]:line(1125,y,1158,y,'muted',1.5,arrow=True)
text(1162,564,'k* = Select{qualityₖ}',18,bold=True)
text(1162,594,'R不参与运行时选择',17,col='red')
text(1162,626,'候选分支仍可分别',17);text(1162,650,'进入后续Evalkit',17)
# E output teal ghost series and world trajectory
for i in range(3):human(1465+63*i,334,.4,.2+i*.6,'teal',True)
path('M1431 404 Q1480 417 1522 405 T1630 412','teal',2,dash=True)
text(1442,462,'Fₖ = FKθₖ(Qₖ)',23,bold=True,col='teal')
text(1442,501,'节点位置 / 段旋转',17);text(1442,533,'完整轨迹与实际文件',17)
text(1442,578,'来源 + 支持掩码',17);text(1442,611,'原生 / 导出 / 回退',17)
line(1378,373,1422,373,'muted',2,arrow=True)
# bottom independent evaluation and backend ribbon
line(30,749,1650,749,'light',1.5)
rect(32,780,1616,151,'orange',fill='#fffaf4',radius=6,dash=True)
text(51,810,'INDEPENDENT EVALUATION',18,bold=True,col='orange')
human(79,870,.25,.4,'orange');text(123,850,'原动作 R',20,bold=True);text(123,880,'仅受控构造与评分使用',16,col='muted')
line(333,854,398,854,'muted',1.5,arrow=True)
text(424,836,'逐候选：D−R  /  Fₖ−D  /  Fₖ−R',23,bold=True)
text(424,870,'原始 / 统一骨长 · native / export · D / R时钟',18)
text(424,901,'所有适用候选独立验收；准确度、覆盖与时序联合判断',17,col='muted')
line(1073,813,1073,906,'light',1.5)
text(1101,837,'候选能力 ≠ 选择器效果',21,bold=True)
text(1101,871,'候选通过率 / 互补覆盖 / 选择命中分列',17)
text(1101,901,'图中姿态与质量符号均为方法示意',16,col='muted')
# route from output to scoring, avoids solver feedback
path('M1609 650 L1640 650 L1640 765 L1023 765 L1023 797','orange',1.8,dash=True,arrow=True)
rect(32,953,1616,61,'light',fill='#f4f7fa',radius=6)
text(50,981,'EXECUTION BACKEND',17,bold=True,col='muted');text(302,981,'同一方法：CPU策略 ↔ Torch计算；批量FK / 导数已接入，设备端求解与异构Batch待完成。',18)
text(302,1004,'GPU改变执行位置与吞吐；算法变化单独成候选，保持原独立验收。',16,col='muted')
a.append('</svg>');svg=''.join(a);(O/'00_method_overview.svg').write_text(svg)
p=R/'docs/20261119_k4_method_atlas.html';s=p.read_text()
main='''<section id="method"><h2>方法总图：输入驱动的多候选运动学重建</h2><div class="figure">'''+svg+'''</div><p class="caption">主图呈现完整方法及信息边界。候选之间可以改变固定几何、可行域、目标和求解路径；每个候选沿整段时间保持自己的固定模型。原动作验收与GPU执行层均独立于运行时重建主线。</p><p><a download href="assets/20261119_k4_method_atlas/00_method_overview.svg">下载方法主图 SVG</a> · <a href="#matrix">加速完成度</a> · <a href="#catalogue">全部候选配置</a></p></section>'''
# One overview is primary, all six derivations moved into an explicit appendix.
s=re.sub(r'<section id="method">.*?</section>','',s,flags=re.S)
s=s.replace('<main>','<main>'+main,1)
start=s.index('<section id="pipeline">');end=s.index('<section id="matrix">',start)
chunk=s[start:end]
if '<details id="technicalAppendix"' not in s:
 s=s[:start]+'<details id="technicalAppendix"><summary style="padding:20px;background:#edf3f7">展开技术附图：残差、稀疏结构、执行后端、导出及性能证据</summary>'+chunk+'</details>'+s[end:]
# expert statements retained but secondary to overview

if '<nav><a href="#method">' not in s:s=s.replace('<nav>', '<nav><a href="#method">方法主图</a>',1)
s=s.replace('K4的优化问题与计算结构<br>从关节几何到批量求解','K4 · 多候选运动学重建方法')
s=s.replace('核心问题：观测目标在什么可行域内求解？哪些残差耦合了时间？哪些计算真正迁到GPU？用关节链、因子图和稀疏结构逐层回答。','统一异构动作观测，在输入定义的固定模型上构造整段运动学候选；通过模型FK生成可追溯结果，并分别评价候选能力与输入侧选择。')
s=s.replace('href="#pipeline">方法流程','href="#technicalAppendix">技术附图')
s=re.sub(r'<a href="#(?:candidates|batch|execution|export)">.*?</a>','',s)
# opening appendix via hash preserves detailed existing links
s=s.replace('<script>','<script>\nfunction revealAppendix(){const id=location.hash.slice(1), el=document.getElementById(id);if(el){const d=el.closest("details");if(d)d.open=true;}} window.addEventListener("hashchange",revealAppendix);revealAppendix();\n',1)
p.write_text(s)
print('master method figure and hierarchy updated')
