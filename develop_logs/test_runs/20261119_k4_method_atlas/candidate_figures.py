from pathlib import Path
import math,re,html,json,hashlib,datetime
R=Path('/home/heinrich/MorionGen');B=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k4_20261117');O=R/'docs/assets/20261119_k4_candidates';O.mkdir(exist_ok=True)
P=json.loads((B/'full_candidate_ablation_20261118_v1/protocol.json').read_text());T=P['templates'];BASE=T['balanced_targets']['config']
C={'ink':'#19344a','blue':'#285d94','teal':'#148b85','orange':'#ce8638','purple':'#8170a5','muted':'#718497','light':'#dbe4ec','red':'#b16572'}
def E(t):return html.escape(str(t))
class Fig:
 def __init__(self,i,name,title):
  self.i=i;self.s=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1440 760" role="img" aria-labelledby="f{i}"><title id="f{i}">{E(name)} — {E(title)}</title><defs><marker id="a{i}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10Z" fill="#718497"/></marker></defs><rect width="1440" height="760" fill="white"/><style>text{{font-family:Arial,"Noto Sans CJK SC",sans-serif}}path{{stroke-linecap:round;stroke-linejoin:round}}</style>'];self.txt(28,38,f'{i:02d}  {name}',28,b=True);self.txt(28,72,title,21,col='muted');self.line(28,92,1412,92,'light',1)
 def txt(self,x,y,t,sz=18,col='ink',b=False,anchor='start'):self.s.append(f'<text x="{x}" y="{y}" font-size="{sz}" fill="{C.get(col,col)}" font-weight="{700 if b else 400}" text-anchor="{anchor}">{E(t)}</text>')
 def line(self,x,y,X,Y,col='muted',w=1.5,dash=False,arrow=False):self.s.append(f'<path d="M{x} {y} L{X} {Y}" stroke="{C.get(col,col)}" stroke-width="{w}" fill="none" '+('stroke-dasharray="5 5" ' if dash else '')+(f'marker-end="url(#a{self.i})"' if arrow else '')+'/>')
 def path(self,d,col='muted',w=1.5,dash=False,fill='none'):self.s.append(f'<path d="{d}" stroke="{C.get(col,col)}" stroke-width="{w}" fill="{fill}" '+('stroke-dasharray="5 5"' if dash else '')+'/>')
 def dot(self,x,y,r=4,col='blue',fill='white'):self.s.append(f'<circle cx="{x}" cy="{y}" r="{r}" stroke="{C.get(col,col)}" stroke-width="1.6" fill="{fill}"/>')
 def rect(self,x,y,w,h,col='light',fill='white',dash=False):self.s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5" stroke="{C.get(col,col)}" fill="{fill}" '+('stroke-dasharray="5 5"' if dash else '')+'/>')
 def axes(self,x,y,s=16):
  for dx,dy,c in [(s,3,'red'),(-s*.5,s*.4,'teal'),(0,-s,'blue')]:self.line(x,y,x+dx,y+dy,c,1.3,arrow=True)
 def human(self,x,y,s=.7,t=.5,col='blue',model=False,axes=False,part=None):
  b=.35*math.sin(t);p={'hip':(0,0),'lh':(-17,0),'rh':(17,0),'chest':(62*math.sin(b),-62*math.cos(b))};p['ls']=(p['chest'][0]-34,p['chest'][1]);p['rs']=(p['chest'][0]+34,p['chest'][1]);p['neck']=(p['chest'][0]+2,p['chest'][1]-19);p['head']=(p['neck'][0]+5*math.sin(t),p['neck'][1]-22)
  def limb(k,n,L,a):p[n]=(p[k][0]+L*math.sin(a),p[k][1]+L*math.cos(a))
  limb('ls','le',43,-.3-.4*math.sin(t));limb('le','lw',40,-.1-.45*math.sin(t));limb('rs','re',43,.45+.65*math.sin(t));limb('re','rw',40,.4+1.1*math.sin(t+.6));limb('lh','lk',56,.2+.3*math.sin(t));limb('lk','la',55,-.2-.2*math.sin(t));limb('rh','rk',56,-.2+.4*math.sin(t));limb('rk','ra',55,-.15-.25*math.sin(t));p['lf']=(p['la'][0]+20,p['la'][1]+3);p['rf']=(p['ra'][0]+20,p['ra'][1]+3)
  q={k:(x+s*v[0],y+s*v[1]) for k,v in p.items()};edges=[('lh','rh'),('hip','chest'),('chest','ls'),('chest','rs'),('chest','neck'),('neck','head'),('ls','le'),('le','lw'),('rs','re'),('re','rw'),('lh','lk'),('lk','la'),('la','lf'),('rh','rk'),('rk','ra'),('ra','rf')]
  leg={'lh','rh','lk','rk','la','ra','lf','rf'}
  for a,b in edges:self.line(*q[a],*q[b],col if part!='legs' or (a in leg and b in leg) else 'light',(5 if model else 2)*s)
  for k in q:self.dot(*q[k],max(2,s*3.2),col if part!='legs' or k in leg else 'light')
  self.dot(*q['head'],s*10,col if part!='legs' else 'light')
  if axes:
   for k in ['hip','chest','rk']:self.axes(*q[k],s*20)
  return q
 def series(self,x,y,s=.35,col='blue',reverse=False):
  for j in range(3):self.human(x+j*72,y,s,.2+j*.6,col,True)
  if reverse:self.line(x+180,y+60,x-22,y+60,col,1.7,arrow=True)
  else:self.line(x-22,y+60,x+180,y+60,col,1.7,arrow=True)
 def factor(self,x,y,reverse=False,global_=False,rate=False):
  for j in range(4):
   self.dot(x+j*74,y,16,'blue');self.txt(x+j*74,y+5,'q'+str(j),14,anchor='middle')
   self.rect(x+j*74-7,y-61,14,14,'orange','#fbecd9');self.line(x+j*74,y-47,x+j*74,y-16)
   if j<3:
    if global_:
     self.rect(x+j*74+31,y-6,12,12,'teal','#dcf1ed');self.line(x+j*74+16,y,x+j*74+31,y);self.line(x+j*74+43,y,x+(j+1)*74-16,y)
    elif reverse:self.line(x+(j+1)*74-17,y,x+j*74+17,y,'muted',1.4,arrow=True)
    else:self.line(x+j*74+17,y,x+(j+1)*74-17,y,'muted',1.4,arrow=True)
  if rate:self.txt(x-10,y+41,'Δ(u−uᴰ) / Δt + rate hinge',17,col='teal')
 def output(self,x,y,kind):
  self.human(x+94,y,.6,.6,'teal',True,part='legs' if kind=='legs' else None)
  for a,b in [(x+15,y+102),(x+89,y+118),(x+187,y+104)]:self.dot(a,b,4,'blue')
  self.path(f'M{x+15} {y+102} Q{x+55} {y+103} {x+89} {y+118} T{x+187} {y+104}','teal',2,dash=(kind=='so3'))
  self.txt(x,y+158,{'legs':'Leg-coordinated FK','full':'Whole-body coordinated FK','so3':'SO(3) → model FK'}[kind],18,b=True,col='teal')
 def footer(self,scope,formula,limit):
  self.line(28,597,1412,597,'light',1);self.txt(30,632,formula,23,b=True);self.txt(30,669,limit,17,col='muted');self.rect(28,696,1384,44,'light','#f4f7fa');self.txt(45,724,scope,17)
 def save(self,name):
  svg=''.join(self.s)+'</svg>';(O/(name+'.svg')).write_text(svg);return svg

def diff(a,b,path=''):
 out={}
 if isinstance(a,dict) and isinstance(b,dict):
  for k in sorted(set(a)|set(b)):
   if k not in a:out[path+k]={'added':b[k]}
   elif k not in b:out[path+k]={'removed':a[k]}
   else:out.update(diff(a[k],b[k],path+k+'.'))
 elif a!=b:out[path.rstrip('.')]={'before':a,'after':b}
 return out
# Each is an explicit method explanation, not a generic placeholder summary.
SPEC={
'balanced_targets':('Balanced observation targets','共同基准：方向、位置、几何角与目标超额惩罚','baseline','forward','so3','ℒ = ‖rdir‖² + ‖rpos‖² + ‖rangle‖² + ‖rhinge‖² + ℒtemporal','方向5°、位置5cm作工程归一；margin=0.8、hinge幅值权重4。软惩罚不保证硬达标。'),
'pelvis_continuity':('Pelvis-line emphasis','骨盆线加权；保留连续目标，移除几何角与目标hinge项','pelvis','forward','so3','w(pelvis line): 1 → 2 ;   joint-angle / target-hinge terms: off','相对balanced_targets同时变化多个残差项，不能将结果全部归因于骨盆权重。'),
'rig_centres':('Declared-rig joint-centre registration','以声明源骨架关节中心注册固定模型','centres','forward','so3','θcentres ← declared rig ;   Q* = Refine(D; θcentres, ℒbalanced)','只增加joint_centres=declared_rig；几何片段内固定，节点中心与模型FK一致。'),
'rig_arm_domain_centres':('Registered geometry + limb-domain expansion','声明中心/足端，扩大上臂轴向及足踝坐标域','domain','forward','so3','θ ← rig centres / foot ;   Qθ ← Qθ ∪ {ankle, subtalar, arm axial ranges}','ankle/subtalar ±90°；arm_rot ±180°。关节结构保留，范围为独立运动学配置。'),
'rig_reverse_leg_path':('Reverse native fit + leg-path coordination','声明足几何与足踝范围；反向原生拟合，腿部协调导出','foot','reverse','legs','qT → qT−1 → … → q0 ;   inserted frames: optimize 12 leg coordinates','反向遍历改变前态递推；原生锚点固定，仅腿部坐标参与该方案的中间帧协调。'),
'rig_reverse_rate_leg_path':('Reverse fit + added-rate penalty + leg path','在反向腿部方案上增加方向误差变化率超额惩罚','rate','reverse','legs','ℒreverse + ‖r(rate-excess; weight=1, margin=0.8)‖²','惩罚拟合相对D新增的方向变化，不直接压低输入动作速度；导出仍为腿部协调。'),
'reverse_full_path':('Reverse native fit + whole-body coordination','反向重建与全身中间姿态协调','foot','reverse','full','Reverse Refine → native anchors → whole-body inserted-frame Refine','与rig_reverse_leg_path的主要区别为协调坐标范围：全部可动角坐标，含上肢。'),
'position_chain':('Position-derived fixed trunk proxy','从位置链估计固定有效躯干长度，再逐帧拟合','chain','forward','so3','Lproxy = medianₜ Σᵢ ‖pi+1,t − pi,t‖ ;   θtrunk fixed for the clip','逐段P5–P95相对跨度均≤10%；缺测或不稳定时保留模型先验。不是中立解剖长度识别。'),
'position_fixed_foot_chain':('Trunk proxy + fixed foot-envelope calibration','链长代理后，以输入足向量极角包络标定固定足端','footproxy','forward','so3','minδ ‖δ‖² / L²   s.t. input polar envelope covered, foot length within L ± 0.02 m','只调整固定MTP父框架平移；保持踝中心、轴、结构与耦合。包络可行是必要条件。'),
'position_trajectory':('Whole-window position-driven trajectory refinement','链长代理逐帧种子 → 整段有界Gauss–Newton','chain','global','so3','Q* = argmin ‖W(φθ(Q)−y)‖² + ‖L(φθ(Q)−y)‖² + hinge','整段前扩展arm_rot ±180°且核验种子FK不变；预算20次，线性迭代50，LSMR上限500。'),
'position_trajectory_chart':('Equivalent-chart seeded trajectory refinement','整段求解前选择边界余量更大的等价旋转坐标图','chart','global','so3','qseed ← argmax margin(q)   s.t. R(q)=R(qold) ;   then optimize Q jointly','仅审计过的三轴SO(3)坐标；保留物理姿态并核验FK，原表示平局优先。'),
'forward_full_path':('Forward native fit + whole-body coordination','共同基准的正向原生求解，追加全身协调导出','baseline','forward','full','Balanced forward Refine → fixed native anchors → whole-body path Refine','相对balanced_targets只增加全身coordinated_path；原生解与插值误差可配对隔离。'),
'position_total_proxy':('Total-chain stability gate + charted trajectory','总链长稳定性门控，配合等价坐标图与整段求解','total','global','so3','same Lproxy estimator ; gate: first 3 links stable AND total chain stable','总链长策略改变稳定性门控，并排除先验节点；并非更换长度估计器。160/164路与chart分支声明等价。')}

def native_fig(i,name):
 c=T[name]['config'];title,why,geom,solve,out,formula,limit=SPEC[name];f=Fig(i,name,title+' · '+why)
 pos=name.startswith('position_');rig=name.startswith('rig_');nformat=2 if pos else 6 if rig else 8
 for x,title2 in [(30,'Input observations'),(310,'Fixed geometry / domain'),(687,'Native reconstruction'),(1150,'Output path')]:f.txt(x,132,title2,20,b=True)
 for j in range(2):f.human(83+j*105,286,.5,.2+j*.7,'orange',False,not pos)
 f.line(38,364,254,364,'muted',1.4,arrow=True);f.txt(37,408,'P, t, M'+('' if pos else ' + R / declared rig'),17,col='orange',b=True);f.txt(37,441,f'{nformat}种格式适用',17)
 f.txt(37,480,'独立旋转按实际字段保留',16,col='muted');f.txt(37,508,'R仅用于独立验收',16,col='red')
 q=f.human(457,304,.8,.4,'blue',True,True)
 if geom in ('centres','domain'):
  for node in ['ls','rs','hip']:
   x,y=q[node];f.dot(x+13,y-11,4,'orange');f.line(x+13,y-11,x,y,'orange',1.5,arrow=True)
  f.txt(320,420,'declared joint centres → θ',18,col='orange',b=True)
 elif geom in ('chain','total'):
  xy=[(354,337),(362,299),(385,275),(408,242),(421,221)]
  for a,b in zip(xy,xy[1:]):f.line(*a,*b,'orange',3)
  for a in xy:f.dot(*a,4,'orange')
  f.line(354,337,354,221,'muted',1,dash=True);f.txt(318,420,'curved chain → fixed Lproxy',18,col='orange',b=True)
 elif geom=='chart':
  f.path('M356 231 A29 29 0 1 1 355 232','orange',2);f.path('M356 247 A13 13 0 1 1 355 248','blue',2);f.txt(320,420,'R(qold) = R(qnew)',20,col='orange',b=True)
 elif geom=='footproxy':
  f.path('M416 393 Q455 441 528 395','orange',2);f.path('M416 393 Q458 418 510 386','teal',2);f.dot(478,399,4,'orange');f.line(461,388,478,399,'orange',2,arrow=True);f.txt(320,444,'polar envelope → fixed δMTP',18,col='orange',b=True)
 elif geom in ('foot','rate'):
  for n in ['lf','rf']:x,y=q[n];f.dot(x+9,y-5,4,'orange');f.line(x,y,x+9,y-5,'orange',2)
  f.txt(320,430,'declared fixed foot geometry',18,col='orange',b=True)
 elif geom=='pelvis':
  f.line(*q['lh'],*q['rh'],'orange',6);f.txt(320,420,'pelvis-line residual × 2',18,col='orange',b=True)
 else:f.txt(320,420,'input-calibrated θ',19,col='blue',b=True)
 if geom=='domain':
  f.path('M513 251 Q569 271 548 318','orange',2);f.txt(320,456,'ankle / subtalar: ±90°',17);f.txt(320,486,'arm axial: ±180°',17)
 elif geom in ('foot','rate','chain','total','chart','footproxy'):
  f.txt(320,483,'ankle / subtalar: ±90°',17)
 else:f.txt(320,465,'固定长度、关节轴与耦合',17)
 f.line(264,300,300,300,'muted',1.6,arrow=True);f.line(621,300,675,300,'muted',1.6,arrow=True)
 if solve=='global':
  f.series(722,233,.29,'blue');f.txt(704,318,'逐帧种子 Q⁰',17)
  f.factor(722,405,global_=True)
  # explicit block Jacobian mini matrix
  for y in range(4):
   for x in range(4):f.rect(1018+x*20,355+y*22,14,16,'light','#b6dcd7' if x==y or x==y+1 else '#f4f6f8')
  f.txt(700,467,'bounded GN + sparse LSMR',19,col='teal',b=True)
  f.txt(700,499,'feature errors coupled across time',16,col='muted')
  if geom=='chart':f.txt(700,534,'equivalent chart → larger seed margin',16,col='orange')
  if geom=='total':f.txt(700,534,'gate changes; same median estimator',16,col='orange')
 else:
  f.series(737,237,.33,'blue',solve=='reverse');f.factor(727,406,reverse=solve=='reverse',rate=geom=='rate')
  f.txt(700,480,('reverse' if solve=='reverse' else 'forward')+' TRF · multistart',20,col='blue',b=True)
  f.txt(700,512,'nominal + previous + triggered knee',16,col='muted')
  if geom=='pelvis':f.txt(700,546,'angle / hinge OFF; pelvis weight 2',16,col='orange')
  elif geom=='rate':f.txt(700,546,'added-rate hinge ON',16,col='orange')
  else:f.txt(700,546,'balanced pose + temporal residuals',16,col='muted')
 f.line(1104,300,1134,300,'muted',1.6,arrow=True);f.output(1154,285,out)
 f.txt(1154,492,'root + attached head/neck',17);f.txt(1154,524,'native / export separately stored',15,col='muted')
 f.footer('加速：FK/扰动批已接入；'+('整段稀疏线性求解仍在CPU。' if solve=='global' else 'TRF控制仍在CPU；前态依赖保留。')+'  本图为方法定义，未声称该候选已通过全部窗口。',formula,limit)
 return f.save(name)

def export_fig(i,name):
 t=T[name];base=t['base'];c=t['config'];title,why,geom,solve,out,formula,limit=SPEC[base];f=Fig(i,name,'Whole-body coordinated export · 基于 '+base+' 的同一原生解')
 for x,title2 in [(30,'Reuse native reconstruction'),(400,'Construct supported path targets'),(805,'Joint-coupled inserted-frame fit'),(1167,'Export & independent check')]:f.txt(x,132,title2,19,b=True)
 f.series(83,267,.48,'blue',solve=='reverse');f.txt(32,381,'D + θbase + Qnative',23,b=True);f.txt(32,421,'原生求解：'+('整段联合' if solve=='global' else '反向逐帧' if solve=='reverse' else '正向逐帧'),18)
 f.txt(32,455,'继承：'+{'centres':'声明关节中心','domain':'声明中心与上肢/足踝范围','chain':'链长代理','chart':'等价坐标图 + 链长代理','total':'总链稳定门控 + 等价坐标图','footproxy':'固定足端包络 + 链长代理','foot':'声明足几何与足踝范围','rate':'反向新增变化率惩罚','pelvis':'骨盆权重2；角度/hinge关闭','baseline':'balanced目标'}[geom],16)
 f.txt(32,493,'复用原生拟合，不重新优化原生帧',17,col='blue',b=True)
 f.line(340,286,388,286,'muted',1.7,arrow=True)
 # path target with fixed native knots and inserted target dots
 p0=(441,339);p1=(708,228);f.line(*p0,*p1,'teal',2)
 for t2 in [0,.25,.5,.75,1]:
  x=p0[0]+t2*(p1[0]-p0[0]);y=p0[1]+t2*(p1[1]-p0[1]);f.dot(x,y,7 if t2 in [0,1] else 4,'blue' if t2 in [0,1] else 'teal')
 f.path('M441 339 Q620 391 708 228','muted',1.5,dash=True)
 f.txt(407,392,'p*(α)=(1−α)FK(q₀)+αFK(q₁)',20,b=True)
 f.txt(407,430,'M*(α) = M₀ ∩ M₁',20,col='orange');f.txt(407,470,'D的有效观测门控中间目标',17)
 f.txt(407,506,'线性路径是插值假设，非新增观测',16,col='muted')
 f.line(751,286,794,286,'muted',1.7,arrow=True)
 q=f.human(930,300,.82,.8,'teal',True,True)
 # locked native knots depicted at ends, unknown insertions around model
 for x in [834,1050]:f.rect(x,203,23,23,'blue','#e2ebf6');f.txt(x+11,220,'q',16,anchor='middle');f.line(x+11,226,930,256,'light',1)
 f.txt(810,435,'原生坐标精确保留',19,col='blue',b=True);f.txt(810,469,'全部可动角坐标参与协调',17)
 f.txt(810,501,'60次预算 · 坐标连续权重0',17);f.txt(810,535,'仍保留当前refine的初值语义',16,col='muted')
 f.line(1116,286,1155,286,'muted',1.7,arrow=True);f.output(1170,285,'full')
 f.txt(1170,492,'原生 → 协调 → 实际回读',16);f.txt(1170,524,'D/R时钟各自验收',17)
 special='从SO(3)导出增加全身协调。' if out=='so3' else '由原有腿部协调升级为全身协调；复用同一原生锚点。'
 f.footer('加速：可复用原生解；中间FK/残差可批。完整GPU协调求解尚未接入；不得未经验证移除前态。','q*α = argminq ‖r(FKθ(q), p*(α))‖² ;   q(tnative) = qnative',special+' 独立验收同时保留原生精度与插值新增误差。')
 return f.save(name)
allfigs=[]
for i,(name,t) in enumerate(T.items(),1):allfigs.append((name,native_fig(i,name) if t['kind']=='native' else export_fig(i,name)))
CSS='''*{box-sizing:border-box}html{scroll-behavior:auto}body{margin:0;background:#f1f4f7;color:#19344a;font:16px/1.75 system-ui,"Noto Sans CJK SC",sans-serif}header{background:#183248;color:white;padding:34px max(22px,calc((100vw - 1480px)/2))}h1{font-size:32px;margin:6px 0}header p{color:#d3e1eb;max-width:1180px}main{max-width:1520px;margin:auto;padding:20px}nav{position:sticky;top:0;background:#fff;z-index:10;padding:10px 22px;display:flex;gap:12px;flex-wrap:wrap;border-bottom:1px solid #dbe4ec}a{color:#285d94}button,select{border:1px solid #b8c9d6;background:white;color:#19344a;border-radius:5px;padding:8px 13px;font-size:14px;cursor:pointer}button.active{background:#19344a;color:white}section{background:white;margin:22px 0;border:1px solid #dbe4ec;border-radius:8px;padding:22px;scroll-margin-top:80px}h2{margin:0 0 12px;font-size:24px}.figure{overflow:auto;margin:12px -10px}svg{display:block;width:100%;height:auto}.caption{font-size:14px;color:#667f91}code{font-size:13px;overflow-wrap:anywhere}details{padding:12px 0;border-top:1px solid #e0e7ee}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;font-size:12px;max-height:360px;overflow:auto;padding:14px;background:#f5f7fa}.legend{display:flex;gap:20px;flex-wrap:wrap}.legend span{border-left:5px solid;padding-left:8px}.tools{display:flex;gap:14px;align-items:center;flex-wrap:wrap}.note{background:#fff7ea;padding:15px;border-left:3px solid #ce8638}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:15px}.card{padding:15px;background:#f3f7fa}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;text-align:left;border-bottom:1px solid #dbe4ec}th{background:#edf3f7}footer{padding:25px;color:#718497;font-size:13px}@media(max-width:750px){main{padding:10px}section{padding:12px}svg{min-width:1000px}.grid{grid-template-columns:1fr}nav{position:static}}@media print{nav,.tools,button{display:none}section{break-inside:avoid}body{background:white}.figure{overflow:visible}svg{min-width:0}header{background:white;color:#19344a}header p{color:#718497}details{display:block}pre{max-height:none}}'''
parts=[f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>K4 全部备选方案 · 逐方案方法图</title><style>{CSS}</style></head><body><header id="method"><h1>K4 全部备选方案 · 逐方案方法图</h1><p>13个原生重建配置 + 11个协调导出变体。每一张图对应冻结协议的一列；以人体几何、时序求解和输出路径呈现完整方案，并突出相对共同基准的变化。</p><div class="legend"><span style="border-color:#ce8638">橙：输入与变化因素</span><span style="border-color:#285d94">蓝：固定模型与原生解</span><span style="border-color:#148b85">绿：轨迹耦合与导出</span></div></header><nav><button class="active" data-filter="all">全部24方案</button><button data-filter="native">13个原生配置</button><button data-filter="export">11个导出变体</button><select id="jump" aria-label="跳转到方案"><option value="">选择具体方案</option>']
for name in T:parts.append(f'<option value="{name}">{name}</option>')
parts.append('</select><a href="#acceleration">加速与Batch状态</a></nav><main><section><h2>读图口径</h2><p>原生配置以<strong>balanced_targets</strong>为共同计算基准；导出变体以各自基础方案为基准，复用原生解。图中的骨架与路径均为方法示意，质量由现有独立实验判定。</p><p class="note">特别区分：rig_reverse_leg_path 与 rig_reverse_rate_leg_path 本身已经包含<strong>腿部</strong>协调；其 __coordinated 变体改为<strong>全身</strong>协调。forward_full_path 和 reverse_full_path 本身已有全身协调，因此没有额外同名变体。position_total_proxy 使用相同长度估计器，改变的是稳定性门控。</p></section>')
for i,(name,svg) in enumerate(allfigs,1):
 t=T[name];base=t['base'];isexp=t['kind']=='export';counts={}
 for route in P['routes']:
  state=route['cells'][name]['status'];counts[state]=counts.get(state,0)+1
 eligible=sum(v for k,v in counts.items() if k!='inapplicable')
 delta={'export_policy_before':t['config'].get('coordinated_path','whole_body_so3'),'export_policy_after':t['policy'],'native_fit_recomputed':False} if isexp else diff(BASE,t['config'])
 parts.append(f'<section class="method" id="{name}" data-kind="{t["kind"]}"><div class="figure">{svg}</div><div class="tools"><a download href="assets/20261119_k4_candidates/{name}.svg">下载SVG</a><a href="assets/20261119_k4_candidates/{name}.png">查看高清PNG</a><span class="caption">适用 {eligible}/656 路；{E(str(counts))}</span></div>')
 parts.append(f'<details><summary>该方案与基准的逐字段差异</summary><p class="caption">基准：{base if isexp else "balanced_targets"}；包含实现参数与说明字段，完整保留以便审阅。</p><pre>{E(json.dumps(delta,ensure_ascii=False,indent=2))}</pre></details><details><summary>完整冻结配置</summary><pre>{E(json.dumps(t,ensure_ascii=False,indent=2))}</pre></details></section>')
parts.append('''<section id="acceleration"><h2>对应这些方案的加速边界</h2><table><thead><tr><th>方法家族 / 环节</th><th>保持算法的Batch轴</th><th>已做</th><th>尚未完成</th></tr></thead><tbody><tr><td>全部13配置：基础模型与IK</td><td>独立模型/初值；相同身份缓存</td><td>固定模型导出、读取与结果复用</td><td>基础OpenSim IK完整GPU迁移</td></tr><tr><td>逐帧原生重建</td><td>同次有限差分列、独立初值、独立序列</td><td>GPU批量FK/扰动、CUDA Graph、模型驻留；同模型多序列局部实验</td><td>设备端TRF控制、异构固定几何批处理；连续前态要保留</td></tr><tr><td>position_trajectory / chart / total</td><td>给定轨迹后的全帧FK与导数；独立轨迹系统</td><td>批量计算后端；CPU稀疏算子局部优化</td><td>等价GPU有界线性求解、稀疏组装与迭代控制</td></tr><tr><td>位置几何代理 / 固定足端</td><td>输入几何统计、独立固定模型校准</td><td>CPU规则与回退实现；几何保持片段内固定</td><td>固定足端SLSQP的GPU一致性路径未实现</td></tr><tr><td>所有协调导出路径</td><td>原生解复用；区间/中间FK；独立路径</td><td>固定锚点、支持门控、独立回读</td><td>完整GPU协调求解；未经对照不能直接移除前态初始化</td></tr><tr><td>root / 头颈 / 评分</td><td>已有身体姿态后的逐帧计算；窗口与候选批</td><td>CPU完整语义；Torch批评分后端已有</td><td>头颈与导出验收的完整GPU覆盖尚未证明</td></tr></tbody></table><p>跨候选独立运行与跨序列批处理可增加吞吐；单序列迭代和连续初值存在依赖。GPU可以承载依赖控制，但当前完整设备端控制尚未实现。缩短预算、删候选或换求解器属于方法变化。</p><p>每个适用候选都单独对R验收；“任一候选通过”“选择器选中通过候选”“全部候选独立验收完成”分别记账。图示与方法描述不授予任何新实验资格。</p></section></main><footer>冻结方法协议：dfe8239；24列 / 9512适用单元。全部SVG、CSS、JavaScript内嵌，可离线阅读。所有插图由固定配置逐项生成；本次未修改拟合或评分。</footer><script>
function filter(kind){document.querySelectorAll('.method').forEach(s=>s.hidden=kind!=='all'&&s.dataset.kind!==kind);document.querySelectorAll('[data-filter]').forEach(b=>b.classList.toggle('active',b.dataset.filter===kind));}document.querySelectorAll('[data-filter]').forEach(b=>b.onclick=()=>filter(b.dataset.filter));document.getElementById('jump').onchange=e=>{if(e.target.value){filter('all');location.hash=e.target.value;}};window.addEventListener('hashchange',()=>{const el=document.getElementById(location.hash.slice(1));if(el&&el.classList.contains('method'))el.hidden=false;});
</script></body></html>''')
(R/'docs/20261119_k4_method_atlas.html').write_text(''.join(parts))
(R/'develop_logs/test_runs/20261119_k4_method_atlas/candidate_figure_manifest.json').write_text(json.dumps({'protocol_sha256':hashlib.sha256((B/'full_candidate_ablation_20261118_v1/protocol.json').read_bytes()).hexdigest(),'figures':[{'name':n,'kind':T[n]['kind'],'svg_sha256':hashlib.sha256(v.encode()).hexdigest()} for n,v in allfigs]},indent=2))
print('24 complete per-candidate figures; 13 native / 11 export')
