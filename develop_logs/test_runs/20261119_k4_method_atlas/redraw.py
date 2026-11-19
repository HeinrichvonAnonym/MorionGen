from pathlib import Path
import math,html,re,json,hashlib
R=Path('/home/heinrich/MorionGen');O=R/'docs/assets/20261119_k4_method_atlas'
P={'ink':'#172c43','blue':'#245a9b','cyan':'#0c8e9a','orange':'#d48333','red':'#b95669','gray':'#8494a4','light':'#dce4eb','purple':'#7762a5'}
def e(x):return html.escape(str(x))
class F:
 def __init__(self,n,title,subtitle,h=680):
  self.n=n;self.h=h;self.a=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 {h}" role="img" aria-labelledby="title{n}"><title id="title{n}">{e(title)}</title><defs><marker id="a{n}" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto-start-reverse"><path d="M0,0 L7,3.5 L0,7Z" fill="#8494a4"/></marker></defs><rect width="1200" height="{h}" fill="white"/><style>text{{font-family:Arial,"Noto Sans CJK SC",sans-serif;fill:#172c43}}.math{{font-family:"DejaVu Serif",Georgia,serif}}.bone{{stroke-linecap:round;stroke-linejoin:round}}</style>']
  self.txt(28,34,f'{n:02}  {title}',25,True);self.txt(28,63,subtitle,15,col='gray');self.line(28,82,1172,82,'light',1)
 def txt(self,x,y,t,sz=17,b=False,col='ink',math=False,anchor='start'):
  content=re.sub(r'\{([^}]+)\}',r'<tspan baseline-shift="sub" font-size="70%">\1</tspan>',e(t)) if math else e(t)
  self.a.append(f'<text x="{x}" y="{y}" font-size="{sz}" font-weight="{700 if b else 400}" style="fill:{P.get(col,col)}" text-anchor="{anchor}"'+(' class="math"' if math else '')+f'>{content}</text>')
 def line(self,x,y,X,Y,col='gray',w=1.5,dash=False,arrow=False):
  self.a.append(f'<path d="M{x} {y} L{X} {Y}" stroke="{P.get(col,col)}" stroke-width="{w}" fill="none" '+('stroke-dasharray="5 4" ' if dash else '')+(f'marker-end="url(#a{self.n})"' if arrow else '')+'/>')
 def path(self,d,col='blue',w=2,fill='none',dash=False):self.a.append(f'<path d="{d}" stroke="{P.get(col,col)}" stroke-width="{w}" fill="{fill}" '+('stroke-dasharray="5 4"' if dash else '')+'/>')
 def rect(self,x,y,w,h,col='light',fill='none',rx=0,dash=False):self.a.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" stroke="{P.get(col,col)}" fill="{fill}" '+('stroke-dasharray="5 4"' if dash else '')+'/>')
 def dot(self,x,y,r=5,col='blue',fill=None):self.a.append(f'<circle cx="{x}" cy="{y}" r="{r}" stroke="{P.get(col,col)}" stroke-width="1.6" fill="{fill or P.get(col,col)}"/>')
 def panel(self,x,y,label,title):self.txt(x,y,label,19,True,col='blue');self.txt(x+29,y,title,18,True)
 def axes(self,x,y,s=27):
  for dx,dy,c,l in [(s,5,'red','x'),(-s*.5,s*.5,'cyan','y'),(0,-s,'blue','z')]:self.line(x,y,x+dx,y+dy,c,1.5,arrow=True);self.txt(x+dx+4,y+dy,l,12,col=c)
 def skeleton(self,x,y,s=1,pose=0,col='blue',dots=True,frames=False):
  # Schematic articulated chain, not measured motion. Coordinates centered at hip midpoint.
  pts={'p':(0,0),'lh':(-22,0),'rh':(22,0),'t':(8 if pose else 0,-90),'ls':(-49+(12 if pose else 0),-100),'rs':(49+(12 if pose else 0),-100),'n':(8 if pose else 0,-128),'h':(12 if pose else 0,-156),'le':(-70,-45),'re':(80,-56 if pose else -40),'lw':(-68,9),'rw':(103,-95 if pose else 12),'lk':(-34+(35 if pose else 0),72),'rk':(35+(24 if pose else 0),72),'la':(-29,145),'ra':(36,145),'lf':(-49,151),'rf':(59,151)}
  edges=[('lh','rh'),('p','t'),('t','ls'),('t','rs'),('ls','rs'),('t','n'),('n','h'),('ls','le'),('le','lw'),('rs','re'),('re','rw'),('lh','lk'),('lk','la'),('la','lf'),('rh','rk'),('rk','ra'),('ra','rf')]
  xy={k:(x+s*a,y+s*b) for k,(a,b) in pts.items()}
  for a,b in edges:self.line(*xy[a],*xy[b],col,4*s)
  if dots:
   for k,(a,b) in xy.items():self.dot(a,b,max(2.5,4*s),col,'white')
  self.dot(*xy['h'],11*s,col,'white')
  if frames:
   for k in ['p','t','rk']:self.axes(*xy[k],20*s)
  return xy
 def finish(self,name):
  text=''.join(self.a)+'</svg>';(O/name).write_text(text);return text
fig=[]
# Figure 1 -- actual mathematical representation and geometric operator.
f=F(1,'Observation-conditioned articulated reconstruction','定义变量、坐标和观测支持；所有骨架及误差箭头均为几何示意。',700)
f.panel(30,118,'a','D：观测与支持掩码')
obs=f.skeleton(154,325,.88,1,'orange',True)
f.axes(55,438,32)
f.txt(35,500,'pᴰₜ,  Rᴰₜ（若独立可观测）,  Mₜ',18,math=True)
f.txt(35,528,'位置格式保留缺失扭转自由度',16,col='gray')
f.line(294,275,354,275,arrow=True);f.txt(302,256,'fit',17,math=True)
f.panel(367,118,'b','θ固定，qₜ求解')
fit=f.skeleton(493,325,.88,0,'blue',True,True)
for k in ['rw','rk','t']:
 a=fit[k];target=(a[0]+17,a[1]-11);f.dot(*target,4,'orange');f.line(*a,*target,'orange',1.3,arrow=True)
f.txt(380,500,'pₜ = fθ(qₜ),  Rₜ = gθ(qₜ)',19,math=True)
f.txt(380,528,'θ：关节中心 / 固定长度 / 附着框架',16,col='gray')
f.line(652,275,704,275,arrow=True);f.txt(659,256,'FK',17,math=True)
f.panel(721,118,'c','输出与可评价变量')
f.skeleton(825,325,.85,0,'blue',True)
f.line(836,192,1085,192,'gray',1.2);f.txt(940,169,'头—胸相对旋转',17,True);f.txt(940,217,'仅在独立支持时评价',15,col='gray')
f.line(887,241,1085,260,'gray',1.2);f.txt(939,281,'整体躯干方向 u',17,True);f.txt(939,304,'≠ 内部腰椎角',15,col='gray')
f.line(825,325,1085,362,'gray',1.2);f.txt(939,386,'root：双髋中点定位',16,True);f.txt(939,409,'补偿后重新FK',15,col='gray')
f.txt(724,500,'位置 / 方向 / 几何角 / 段旋转',18);f.txt(724,528,'来源、掩码、候选与回退均保留',16,col='gray')
f.line(28,555,1172,555,'light',1)
f.txt(35,587,'qₜ ∈ [lθ, uθ] ;    qdep = cθ(qind) ;    θ 在整个窗口内固定',22,math=True)
f.txt(35,624,'身体坐标只优化可动角坐标；从属坐标按模型耦合同步。头颈链与身体模型胸段组合FK。',17)
f.txt(35,660,'R只进入独立评分：D−R、F−D、F−R。观测对齐误差、模型表示误差与数值误差分别追踪。',17,col='gray')
fig.append(f.finish('01_pipeline.svg'))
# Figure 2 factor graph plus exact implemented objective.
f=F(2,'Residual geometry & candidate interventions','方向残差采用单位向量弦长；逐帧目标含坐标连续项，整段目标在观测特征误差上耦合。',760)
f.panel(30,119,'a','单帧残差：现有 refinement.py')
f.txt(40,164,'p̄ = p − mean(pLH, pRH),     u = v / ‖v‖',22,math=True)
f.txt(40,209,'rₚ = wₚ (p̄ − p̄ᴰ) / σₚ',23,math=True)
f.txt(40,253,'rᵤ = (u − uᴰ) / [2 sin(σₐ / 2)]',23,math=True)
f.txt(40,297,'rα = wₐ (α − αᴰ) / σₐ',23,math=True)
f.txt(40,341,'r{q} = w{q} (qₜ − qₜ₋₁) / [2 sin(σₐ / 2)]',21,math=True)
f.txt(40,385,'rΔe = wΔ [(u−uᴰ)ₜ − (u−uᴰ)ₜ₋₁] / (σv Δt)',20,math=True)
f.txt(40,426,'另加：归一目标超额 hinge 惩罚；可选段旋转目标。',16,col='gray')
f.txt(40,454,'σp = 0.05 m；σa = 5°；σv = 10°/s。',17)
f.line(620,104,620,484,'light',1)
f.panel(652,119,'b','状态因子与候选差别')
for x,t in [(724,'qₜ₋₁'),(1003,'qₜ')]:
 f.dot(x,267,28,'blue','white');f.txt(x,274,t,20,True,math=True,anchor='middle')
 f.rect(x-14,170,28,28,'orange',fill='#fff2e5');f.line(x,198,x,239);f.txt(x,153,'Dₜ₋₁' if x==724 else 'Dₜ',19,math=True,anchor='middle')
f.line(752,267,851,267);f.rect(851,253,28,28,'cyan',fill='#e7f6f4');f.line(879,267,975,267)
f.txt(866,229,'Δe / Δt',19,math=True,anchor='middle');f.txt(867,314,'前态同时影响残差与初值',16,anchor='middle')
f.rect(676,361,470,102,'light',fill='#f6f8fa');f.txt(693,389,'单帧：min ‖r(qₜ ; stateₜ₋₁)‖²',22,math=True);f.txt(693,421,'nominal / continuity → TRF → 条件重启',16);f.txt(693,446,'选择读取D；从原动作R挑最优属于事后上界。',15,col='gray')
f.line(28,490,1172,490,'light',1);f.panel(30,526,'c','13个原生候选改变的四类因素')
for x,label,main,sub in [(36,'θ','固定几何','中心、肩宽、足端、链长代理'),(329,'Qθ','可行域','坐标范围与已声明耦合延拓'),(622,'r','目标/权重','骨盆线、hinge、方向误差变化'),(915,'S','求解路径','遍历、初值、等价图、整段')]:
 f.dot(x+31,582,26,'blue','white');f.txt(x+31,591,label,23,True,math=True,anchor='middle');f.txt(x+68,578,main,18,True);f.txt(x,640,sub,15);f.line(x+3,659,x+246,659,'light',2)
f.txt(36,705,'这些是多因素配置，不是13种独立优化算法。11个导出变体另改变插值路径，复用原生解。',18,True)
f.txt(36,735,'重要限制：坐标连续残差仍是坐标差；它与SO(3)导出插值、独立旋转测地评分是不同层次。',17,col='red')
fig.append(f.finish('02_candidates.svg'))
# Figure 3 structural algebra for trajectories and batch.
f=F(3,'Temporal coupling determines the parallelization axis','批处理应沿独立系统和同次迭代的模型评价展开；迭代与前态依赖保持原算法语义。',700)
f.panel(30,119,'a','逐帧：沿序列传播状态')
for j in range(3):
 y=176+j*86;f.txt(35,y+6,f's{j+1}',19,True,math=True)
 for i in range(4):
  x=110+i*114;f.dot(x,y,19,'blue','white');f.txt(x,y+6,f'q{i}',15,math=True,anchor='middle')
  if i<3:f.line(x+20,y,x+94,y,arrow=True)
f.rect(79,148,62,230,'cyan',dash=True);f.txt(45,415,'可合批：每条序列当前就绪的帧',17,True,col='cyan')
f.txt(45,445,'同一序列仍维护独立previous / error。',15,col='gray')
f.panel(575,119,'b','整段：块稀疏 Jacobian')
f.txt(595,157,'eₜ = φθ(qₜ) − yₜ ;   r = [W e ; L e ; hinge]',19,math=True)
# Jacobian explicit block diag local plus first difference
for row in range(9):
 for col in range(5):
  active=(row<5 and row==col) or (row>=5 and col in (row-5,row-4))
  f.rect(626+col*45,196+row*22,39,17,'light',fill=('#356a9d' if row<5 else '#19999c') if active else '#f3f6f8')
f.txt(868,248,'Jφ：逐帧对角块',17,col='blue');f.txt(868,353,'LJφ：相邻帧耦合',17,col='cyan')
f.txt(596,437,'JᵀJ呈块带状；边界与活跃集仍影响每轮求解。',17)
f.line(28,476,1172,476,'light',1);f.panel(30,512,'c','保持算法的Batch与改变算法的并行')
f.txt(35,550,'可保持语义',18,True,col='cyan');f.txt(205,550,'扰动列；独立初值；独立序列/候选；给定Q后的全帧FK。',18)
f.txt(35,592,'需控制依赖',18,True,col='orange');f.txt(205,592,'线搜索接受顺序、条件重启、收敛掩码、预算、下一轮更新。',18)
f.txt(35,634,'另设算法条件',18,True,col='red');f.txt(205,634,'移除前态；独立拟合所有帧；更换线性求解器；减少初值。',18)
f.txt(35,676,'当前缺口：异构固定几何的batched model，以及设备端完整求解状态机。',18,True)
fig.append(f.finish('03_batch.svg'))
# Figure 4 hardware structure no text panels.
f=F(4,'Where the solver actually runs','当前CPU-policy路径：CPU持有算法状态，GPU持有固定模型并执行批量FK；箭头为数值交换。',690)
f.panel(30,118,'a','实际执行分区')
f.rect(52,160,398,280,'blue',fill='#f6f8fc',rx=12);f.rect(80,186,342,80,'blue',fill='white');f.txt(98,214,'CPU · SciPy / NumPy',22,True);f.txt(98,244,'TRF / bounded GN / LSMR',20,math=True)
for i in range(4):f.rect(88+i*79,295,62,60,'blue',fill='#e6edf6',rx=3);f.txt(119+i*79,331,['seed','J,r','step','stop'][i],17,math=True,anchor='middle')
f.txt(86,400,'残差组装 · 稀疏算子 · 接受/回退',17)
f.rect(735,160,398,280,'cyan',fill='#f1faf9',rx=12);f.rect(761,185,346,80,'cyan',fill='white');f.txt(780,214,'GPU · resident θ / float64',22,True);f.txt(780,244,'CUDA Graph + batched FK',19,math=True)
for i in range(8):
 for j in range(3):f.rect(774+i*40,294+j*24,31,15,'cyan',fill='#b6e0da')
f.txt(777,400,'q、q±h列 → 点 / 所需段旋转',17)
f.line(450,215,733,215,arrow=True);f.txt(487,193,'批量坐标 / 扰动列',17)
f.line(735,322,452,322,arrow=True);f.txt(490,301,'FK数值；按需回读',17)
f.path('M85 439 L85 457 L32 457 L32 147 L85 147 L85 159','gray',1.5);f.txt(115,474,'每轮重复；结果仍依赖CPU决策',17,col='gray')
f.line(28,497,1172,497,'light',1);f.panel(30,533,'b','已做与未做，对齐到同一计算图')
f.dot(47,572,6,'cyan');f.txt(65,578,'已接入：有限差分FK批、模型驻留、图捕获、轻量协议。',18)
f.dot(47,612,6,'orange');f.txt(65,618,'局部：同模型跨序列微批、并行初值、CPU转置CSR缓存。',18)
f.dot(47,652,6,'red');f.txt(65,658,'待完成：设备端r/J与线性子问题、迭代状态机、异构几何Batch。',18)
fig.append(f.finish('04_execution.svg'))
# Figure 5 path geometry with FK leg chains native at left/right and curves.
f=F(5,'Interpolation on SO(3) does not constrain task-space paths','原生锚点固定；末端轨迹由多关节组合FK决定。曲线为反例示意，非当前样本测量。',700)
f.panel(30,120,'a','关节独立插值 → 末端可偏离')
# hip fixed, knee/foot positions for 3 poses in schematic kinematic chain
hip=(207,185)
def chain(alpha):
 a=math.radians(125-100*alpha);b=math.radians(120-170*alpha)
 knee=(hip[0]+100*math.cos(a),hip[1]+100*math.sin(a));foot=(knee[0]+120*math.cos(b),knee[1]+120*math.sin(b));return knee,foot
poses=[chain(a) for a in [0,.5,1]]
for j,(knee,foot) in enumerate(poses):
 col=['blue','orange','blue'][j];f.line(*hip,*knee,col,5 if j!=1 else 3);f.line(*knee,*foot,col,5 if j!=1 else 3)
 for pt in [hip,knee,foot]:f.dot(*pt,6,col,'white')
curve=[chain(i/40)[1] for i in range(41)];f.path('M'+' L'.join(f'{x:.3f} {y:.3f}' for x,y in curve),'orange',2,dash=True)
f.line(*curve[0],*curve[-1],'cyan',2,dash=True);f.txt(44,431,'R(α) = R₀ exp[α log(R₀ᵀR₁)]',21,math=True)
f.txt(44,465,'每个关节的SO(3)路径不固定多关节末端轨迹。',16)
f.panel(590,120,'b','固定原生锚点的任务空间协调')
for x,y in [(668,348),(803,275),(946,202)]:f.dot(x,y,8,'blue','white')
f.line(668,348,946,202,'cyan',2)
for t in [.2,.4,.6,.8]:
 x=668+(946-668)*t;y=348+(202-348)*t;f.dot(x,y,5,'cyan','white');f.line(x,y,x+14,y-20,'gray',1,arrow=True)
f.txt(1010,208,'原生锚点',16,col='blue');f.txt(855,344,'中间坐标联合求解',17,col='cyan')
f.txt(594,431,'p*(α) = (1−α) fθ(q₀) + α fθ(q₁)',20,math=True)
f.txt(594,466,'q*(α) = argmin ‖r(fθ(q), p*(α))‖²',20,math=True)
f.line(28,498,1172,498,'light',1)
f.txt(35,536,'原生拟合 Fnative',20,True,col='blue');f.line(250,529,352,529,arrow=True);f.txt(370,536,'协调前 / 协调后 Fexport',20,True,col='cyan');f.line(674,529,764,529,arrow=True);f.txt(782,536,'实际文件回读',20,True,col='orange')
f.txt(35,581,'锚点坐标逐值保持；中间目标是插值假设，由D的观测支持门控。',18)
f.txt(35,619,'不同区间有批处理空间；当前refine仍携带前态，不能未经验证便把全部中间帧拆成独立任务。',17)
f.txt(35,661,'SO(3)路径、坐标连续正则、旋转测地验收各司其职；三者不能互相代替。',18,True)
fig.append(f.finish('05_export.svg'))
# Figure6 dense empirical panel with units/axes and cost ratio.
f=F(6,'Kernel speedup vs. solver speedup','两组配对实验的计时边界不同；内核加速倍数与完整循环收益分别呈现。',660)
f.panel(30,118,'a','固定32帧FK请求 / ms')
f.panel(620,118,'b','150帧姿态细化 / s')
for x,y,width,ticks,scale in [(205,370,300,[0,10,20,30,40],7.0),(810,370,280,[0,20,40,60,80],3.4)]:
 f.line(x,153,x,370,'gray',1);f.line(x,370,x+width,370,'gray',1)
 for v in ticks:f.line(x+v*scale,370,x+v*scale,377);f.txt(x+v*scale,398,str(v),14,anchor='middle')
for i,(label,val) in enumerate([('逐算子',34.20),('CUDA Graph',7.30),('+坐标批变换',5.07)]):
 y=175+i*65;f.txt(36,y+23,label,16);f.rect(205,y,val*7,30,'cyan',fill='#b6e0da');f.txt(212+val*7,y+23,f'{val:.2f}',16,True)
for i,(label,val) in enumerate([('控制实现',76.567),('轻量传输',69.059),('组合优化',66.039)]):
 y=175+i*65;f.txt(630,y+23,label,16);f.rect(810,y,val*3.4,30,'blue',fill='#d6e3f3');f.txt(817+val*3.4,y+23,f'{val:.2f}',16,True)
f.txt(205,436,'≈ 6.8×',25,True,col='cyan');f.txt(810,436,'−13.7% time',25,True,col='blue')
f.line(28,465,1172,465,'light',1);f.panel(30,504,'c','9个已剖析整段任务的合计求解时间')
f.rect(35,528,915,39,'light',fill='#f1f4f8');f.rect(35,528,915*.83166,39,'orange',fill='#f2d6b4');f.txt(55,554,'CPU bounded linear subproblem · 83.2%',20,True);f.txt(816,554,'其他16.8%',16)
f.txt(35,599,'A：同请求往返边界。B：复用固定模型与种子，三次预热中位数；CPU14 worker并行。',16)
f.txt(35,633,'C：仅覆盖九项整段任务；当前主要瓶颈仍未完成等价GPU迁移，局部FK收益因此被稀释。',17,True)
fig.append(f.finish('06_evidence.svg'))
page=R/'docs/20261119_k4_method_atlas.html';s=page.read_text();it=iter(fig);s=re.sub(r'<svg xmlns=.*?</svg>',lambda m:next(it),s,flags=re.S)
# Retain complete, source-derived catalogue. Replace verbose figure captions and prose with technical discussion.
sections={
'pipeline':'''<p><strong>问题是带固定几何与观测掩码的逆运动学。</strong>输入D决定目标及可观测性；θ定义节点映射、固定长度、关节坐标架和耦合。身体细化优化未锁定、非从属的角坐标，位置残差在双髋中点坐标系计算。root随后按输入髋中心定位并重新FK，所以求解目标与世界位置验收需分别看。</p><p><strong>表示能力与数值收敛是两层问题。</strong>固定模型可达域不足时，多初值只能改善域内解；几何或坐标范围候选改变的正是可达域。头胸整体方向、胸段旋转和内部腰椎坐标具有不同定义。头颈位置格式中欠观测的扭转自由度可用于构造一致姿态，但不能获授旋转准确度。</p>''',
'candidates':'''<p><strong>主姿态目标不是旋转测地最小二乘。</strong>方向项采用单位向量差，尺度为2sin(5°/2)；位置使用骨盆相对位置；肘膝采用几何屈曲角；启用时的段旋转目标使用旋转矩阵差。独立验收中的旋转测地距离与这些优化残差分别定义。5°/5 cm在这里是工程归一尺度，不是噪声协方差；hinge项只增加超额目标代价，不构成达标的硬约束。</p><p><strong>时序项同时含坐标正则与观测误差变化。</strong>后者约束Δ(u−uᴰ)/Δt，目标是减少拟合新增速度，而不是压低输入动作速度。但rq仍直接使用模型角坐标差，对坐标图和周期表示敏感；全身SO(3)导出不会自动修复这一层。等价坐标图与整段候选针对的是不同的求解条件。</p><p><strong>多初值影响轨迹递推。</strong>常规种子和前态种子经TRF求解后，按固定D侧目标比较；目标超标才触发膝几何种子。微小数值差异可能改变当帧选择，再改变下一帧残差和初始化。诊断必须锁定完整求解前状态，单看初值名称不足以归因。</p>''',
'batch':'''<p><strong>整段目标有可利用的代数结构。</strong>令e=φθ(Q)−y，Jφ按帧分块对角，L是带物理Δt的相邻差分算子；总Jacobian包含WJφ和LJφ。因而不能把整段问题简单拆成逐帧独立最小二乘，但可以批量计算φ与导数，并利用块稀疏结构求步长。</p><p><strong>现阶段最重要的Batch轴是独立系统。</strong>单条连续序列保持自身前态；多个序列或候选可组成活动批。当前SequenceKernel要求固定模型完全一致，而40项有34个不同模型，因此同拓扑、不同θ的批FK及每样本状态管理是扩展吞吐的关键。将冻结CPU算法的决策移入设备可以保留逻辑依赖；改变停止规则、初值数或线性求解策略则需要新的算法条件。</p>''',
'execution':'''<p><strong>目前的GPU路径是模型评价后端，还不是完整GPU求解器。</strong>它复用CPU策略的模型、输入和求解前状态，把扰动坐标的FK批量发送给常驻GPU进程。CPU继续组装残差/稀疏结构、解线性子问题、执行接受和停止决策。只扩大FK批量，无法消除这段主机循环。</p><p><strong>单帧与整段应采用不同优化重点。</strong>单帧的小问题需要跨任务合批、减少同步和设备控制；整段首先需要处理有界稀疏线性求解。现有独立Torch GN是另一算法条件，不能等同于CPU TRF/LSMR的迁移。CPU线性转置缓存的5.6%～7.9%收益只是算子层局部证据。</p>''',
'export':'''<p><strong>SO(3)插值保证关节旋转路径，未直接约束末端路径。</strong>多关节组合后，踝足或手部可能偏离原生端点之间的任务空间路径。协调导出固定所有原生坐标，以相邻原生FK点的线性插值为伪目标，再求中间关节坐标。它改变导出路径，不改变原生解。</p><p><strong>协调路径也有方法假设。</strong>线性末端路径未必处处可达；需要D支持门控、固定模型和回退。其质量应由中间FK及实际回读独立检验。中间帧虽然有并行潜力，当前实现仍复用带前态的refine；去掉前态必须单独对照。</p>'''}
for ident,prose in sections.items():
 pattern=r'(<section id="'+ident+r'">.*?</div><div class="bar">.*?</div>).*?(</section>)'
 s,n=re.subn(pattern,lambda m:m[1]+prose+m[2],s,count=1,flags=re.S);assert n==1,ident
s=s.replace('从动作观测到可信重建<br>方法图谱与批处理空间','K4的优化问题与计算结构<br>从关节几何到批量求解')
s=s.replace('当前收益有限，是因为加速主要覆盖了模型计算，整体求解仍保留CPU控制、线性子问题和连续状态依赖。下一步的重点应从单次FK转向完整迭代与跨任务批处理。','核心问题：观测目标在什么可行域内求解？哪些残差耦合了时间？哪些计算真正迁到GPU？用关节链、因子图和稀疏结构逐层回答。')
s=s.replace('收益已经出现，工程覆盖还未到主要瓶颈','三个决定方法与性能的判断')
a=s.index('<p>这段开发已经建立') if '<p>这段开发已经建立' in s else s.index('<p><strong>① 几何');b=s.index('<div class="callout warn">',a)
s=s[:a]+'''<p><strong>① 几何、目标与求解器相互独立。</strong>当前13个原生候选主要改变θ、可行域、残差权重和求解路径；24列不是24种算法，组合候选比较也不自动提供单因素因果证据。</p><p><strong>② 连续性存在两个层次。</strong>原生求解包含前态与坐标正则；SO(3)插值与协调导出处理采样间路径。一个层次通过，不授予另一个层次资格。</p><p><strong>③ GPU加速尚未覆盖主要控制循环。</strong>当前批量FK已实现；CPU端的线性子问题、决策和前态递推仍限制整体收益。后续优先级应按单帧与整段两类问题分别制定。</p>'''+s[b:]
s=s.replace('论文方法图风格的矢量示意','关节链 / 因子图 / 块稀疏结构')
page.write_text(s)
print('redrawn six figures; expert sections replaced; catalogue preserved')
