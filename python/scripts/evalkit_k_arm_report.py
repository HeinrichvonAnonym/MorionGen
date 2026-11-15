#!/usr/bin/env python3
"""Summarize the development-only K2 arm audit and render source/fit overlays."""
import argparse,json,sys,hashlib,shutil,datetime
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from motiogen.evalkit.k.arms import geometry_quality


def read(path):return json.loads(Path(path).read_text())
def p95(v):return float(np.nanpercentile(v,95))

def overlay_metrics(folder):
    """All final 100 Hz frames, including brief failures between plot samples."""
    from motiogen.evalkit.k.arms import assess
    from motiogen.evalkit.k import load
    import biomech
    work=folder/'work'
    if (work/'arm_quality.npz').exists():
        with np.load(work/'arm_quality.npz') as z:
            quality={s:{k:z[s+'_'+k] for k in ['elbow_bias_deg','position_error_m','valid','direction_error_deg']} for s in 'lr'}
    else:
        source=read(folder/'regression.json')['source']
        source_file=Path(source) if Path(source).is_file() else Path(source)/'motion.npz'
        fingerprint=hashlib.sha256((str(folder)+hashlib.sha256((work/'coordinates.mot').read_bytes()).hexdigest()+hashlib.sha256((work/'model.osim').read_bytes()).hexdigest()+hashlib.sha256(source_file.read_bytes()).hexdigest()+hashlib.sha256((ROOT/'utils/motiogen/evalkit/k/arms.py').read_bytes()).hexdigest()).encode()).hexdigest()
        cached=CACHE/(fingerprint+'.json')
        if cached.exists():return read(cached)
        motion=load(source);points,names,fps=motion.legacy()
        times,labels,coordinates=biomech.read_storage(work/'coordinates.mot')
        quality=assess(work/'model.osim',points,names,fps,times,labels,coordinates,{'quality_angle_deg':10,'quality_position_m':.1})
    result={}
    for side,s in [('left','l'),('right','r')]:
        q=quality[s];error=abs(q['elbow_bias_deg'])
        result[side]={'elbow_p95_deg':p95(error),'elbow_median_deg':float(np.nanmedian(error)),
                      'elbow_bias_deg':float(np.nanmean(q['elbow_bias_deg'])),
                      'within_deg':{str(t):float(np.mean(error<=t)) for t in [5,10,20]},'frames':len(error),
                      'wrist_p95_m':p95(q['position_error_m'][:,2]),'source_fidelity_share':float(q['valid'].mean()),
                      'direction_p95_deg':np.nanpercentile(q['direction_error_deg'],95,axis=0).tolist()}
    if not (work/'arm_quality.npz').exists():cached.write_text(json.dumps(result))
    return result


def plots(data,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    old=Path(read(ROOT/'config/evalkit_k_validation_20261116/final_usage.json')['format_output']).parent/'regression_candidate'
    fig=plt.figure(figsize=(13,13),layout='constrained')
    colors=['#2563a6','#d48228','#23836b']
    for row,method in enumerate(['motionlcm','omnicontrol','kimodo']):
        with np.load(old/method/'overlay.npz') as z:times=z['times'];names=list(z['names']);source=z['source'];before=z['fitted']
        with np.load(data/'regression_k2_adaptive'/method/'overlay.npz') as z:after=z['fitted']
        curve=fig.add_subplot(3,2,row*2+2);errors=[]
        for side,line in [('left','-'),('right','--')]:
            inds=[names.index(side+'_'+j) for j in ('shoulder','elbow','wrist')]
            for fitted,color,label in [(before,colors[1],'K1'),(after,colors[2],'K2')]:
                error=geometry_quality(source[:,inds],fitted[:,inds],10,.1)['elbow_bias_deg'];errors.append(abs(error))
                curve.plot(times,error,line,color=color,label=label+' '+side)
        curve.axhline(0,color='#9ca3af',lw=.7);curve.set(xlabel='Time (s)',ylabel='Signed elbow error (deg)',title=method+' / source-fidelity error');curve.legend(ncol=2,fontsize=8);curve.grid(alpha=.2)
        worst=int(np.argmax(np.maximum(errors[0],errors[2])))
        ax=fig.add_subplot(3,2,row*2+1,projection='3d')
        for points,color,label in [(source,colors[0],'Source'),(before,colors[1],'K1'),(after,colors[2],'K2')]:
            for side in ('left','right'):
                index=[names.index(side+'_'+j) for j in ('shoulder','elbow','wrist')]
                xyz=points[worst,index];ax.plot(xyz[:,0],xyz[:,2],xyz[:,1],'-o',color=color,label=label if side=='left' else None,ms=4)
        arm_index=[names.index(side+'_'+j) for side in ('left','right') for j in ('shoulder','elbow','wrist')]
        combined=np.concatenate([p[worst,arm_index] for p in (source,before,after)])
        mid=(combined.max(0)+combined.min(0))/2;span=max(np.ptp(combined,axis=0))*.55
        ax.set(xlim=(mid[0]-span,mid[0]+span),ylim=(mid[2]-span,mid[2]+span),zlim=(mid[1]-span,mid[1]+span),
               xlabel='X (m)',ylabel='Z (m)',zlabel='Y (m)',title=f'{method} / K1 worst elbow frame, t={times[worst]:.1f}s')
        ax.set_box_aspect((1,1,1));ax.view_init(elev=18,azim=-65);ax.legend(fontsize=8)
    out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(out,dpi=150);plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--data',type=Path,required=True);ap.add_argument('--evidence',type=Path,required=True);ap.add_argument('--report',type=Path,required=True)
    a=ap.parse_args();a.evidence.mkdir(parents=True,exist_ok=True)
    global CACHE
    CACHE=a.data/'arm_metric_cache';CACHE.mkdir(exist_ok=True)
    old={r['method']:r for r in read(ROOT/'config/evalkit_k_validation_20261116/regression_summary.json')}
    new={r['method']:r for r in read(a.data/'regression_k2_adaptive/summary.json')}
    oldroot=Path(read(ROOT/'config/evalkit_k_validation_20261116/final_usage.json')['format_output']).parent/'regression_candidate'
    pairs=[];cohorts={}
    for cohort in ['development','development_first']:
        left={r['key']:r for r in read(a.data/cohort/'k1_summary.json')};right={r['key']:r for r in read(a.data/cohort/'k2_adaptive_summary.json')}
        assert set(left)==set(right)
        for key,r in right.items():
            assert r['partition']=='development' and r['status']==left[key]['status']=='computed'
            pairs.append({'cohort':cohort,'id':r['id'],'group_id':r['group_id'],'method':r['method'],
                          'k1':{k:v for k,v in left[key].items() if k!='retarget_provenance'},'k2':{k:v for k,v in r.items() if k!='retarget_provenance'},'k1_geometry':overlay_metrics(a.data/cohort/'k1'/key),
                          'k2_geometry':overlay_metrics(a.data/cohort/'k2_adaptive'/key)})
        cohorts[cohort]={'exports':len(right),'recordings':len(set(r['id'] for r in right.values())),
                        'k1_qualified':sum(r['calibration']['qualified'] for r in left.values()),
                        'k2_qualified':sum(r['calibration']['qualified'] for r in right.values()),
                        'selection':read(a.data/cohort/'manifest.json')['selection']}
    actual={m:{'k1':old[m],'k2':new[m],'k1_geometry':overlay_metrics(oldroot/m),'k2_geometry':overlay_metrics(a.data/'regression_k2_adaptive'/m)} for m in new}
    output={'scope':'development-only, no new final holdout evaluation','cohorts':cohorts,'actual':actual,'pairs':pairs}
    (a.evidence/'comparison.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    figure=a.report.parent/'20261116_evalkit_k_arm_figures'/'arm_comparison.png';plots(a.data,figure)
    lines=['# K2：手臂拟合误差归因与开发修正','',
      '本轮围绕 MotionLCM、OmniControl、Kimodo 的手臂拟合误差，比较输入关节几何与肌骨模型输出，目标是减少转换及拟合损失，并把仍受限的部位明确标出。K1 提交 `acb48ac`、原冻结记录及最终留出使用记录均保留。本轮使用开发资料，Andy 和电视最终留出没有重跑。','',
      '**结论：肩关节表示范围、手臂标记注册和标定帧筛选共同影响误差。K2 对前两项采用格式配置，并修正了标定帧筛选顺序；输入角度与末端位置分别验收。OmniControl 的部分手臂仍受骨长波动及模型表达限制。**','',
      '## 误差从哪里产生','',
      '逆运动学（IK）将输入关节位置拟合到肌骨模型。源骨架的肩—肘—腕夹角与模型关节中心夹角提供直接几何对照；模型的单个肘屈曲坐标、注册标记夹角和关节中心夹角分别记录，避免混淆定义。','',
      '- **MotionLCM：肩屈曲达到模型旧上限。** 真实样例右肩在约 34% 原始 IK 帧接近 90° 上限；仅扩展该坐标表示范围，较大一侧肘角 P95 从约 26° 降至 11°。原始与滤波后误差接近，主要损失来自拟合约束。',
      '- **OmniControl：约束与非刚性骨架同时作用。** 双肩多次到达旧范围边缘；左上臂长度的 P5–P95 约为 21.6–28.4 cm。固定长度模型需要在肩、肘、腕位置间折中。解除范围限制、保留关节中心后仍有高误差帧，逐侧掩码保留这些限制。',
      '- **Kimodo：注册偏移改变了角度含义。** 原拟合左肘的关节中心误差 P95 约 21°，注册标记夹角误差约 9°；骨长几乎恒定。注册把初次 IK 的偏差部分吸收进标记偏移，导致较好的标记误差与较大的关节中心偏差并存。',
      '- **求解初值：扩展范围需要连续初始化。** 接近 T 姿态的开发记录在直接扩展范围后，IK 可能转入接近 ±180° 的肩部欧拉角分支，肘角误差增至约 80°。扩展范围候选先以原肩部范围求首个观测帧，再以该解初始化全段求解；原范围候选保留原求解初始化；校准和正式 IK 使用同一规则，交付模型的默认姿态保持原值。部分直臂姿态仍可能跳入较差分支，因此同时保留原范围及注册标记候选，以源几何保真度选择。',
      '- **标定筛选：合格帧被过窄的候选窗截断。** 旧实现先按距最小倾斜分数 5° 选帧，再检查 0.5 秒时长。有足够绝对姿态合格帧的序列，也可能只留下 3 帧。K2 先筛躯干 ≤20°、大腿 ≤25° 的全部帧，再优先选较直立者，并在合格池内补足最低时长。姿态阈值保持原值。','',
      '## 实际调整与处理边界','',
      '配置改由 `config/evalkit_k_v2.json` 管理。三个目标格式将关节中心原 ±90°、关节中心扩展 ±180°、注册标记原 ±90° 三条路径作为候选。候选均完成标定、IK 和滤波；优先选择角度与位置共同合格帧比例较高的结果，再以手臂关节位置 RMSE 区分。各候选的误差、失败原因和选择记录一并保存，其他五种格式保留原手臂策略。首次拟合需要计算三条候选，后续相同输入和配置复用缓存。新增范围服务于运动学表示，肩部肌肉在该范围内的受力准确性仍需专项验证。标定筛选修正作用于共享 K 和复用该函数的 FBLS 入口。','',
      '`arms.py` 检查最终滤波坐标的上臂方向、前臂方向、肘角及肩肘腕位置。10° 角度门沿用工程误差档，0.1 m 位置门沿用现有 IK 位置门；两者为工程约定。`arm_quality.npz` 保存两类掩码；对外 `arm_valid` 还与标定有效性相交。工效评分在失效侧保留未知手臂角度，输出评分区间。','',
      '另外试验了固定躯干后的骨方向优化：MotionLCM、Kimodo 的肘角可贴近输入，OmniControl 多起点优化后左肘 P95 约 3.7°。但相同样例的手腕 P95 可增至约 3–7 cm，且更依赖角度坐标的局部解选择。本轮采用有明确消融证据的范围与关节中心修正，骨方向优化保留为实验资料，未进入运行分支。','',
      '## 开发验证结果','',
      '八份真实基线样例用于接口回归；下表肘角和手腕误差使用完整 100 Hz 输出，取左右侧中较大的 P95；躯干沿用 10 Hz 检查网格。角度表示输入保真度，独立解剖精度需要带人体标定的观测。','',
      '| 格式 | K1 肘 P95° | K2 肘 P95° | K1 躯干 P95° | K2 躯干 P95° | 手腕较大 P95 cm（K1→K2） |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for m in ['motionlcm','omnicontrol','kimodo']:
        r=actual[m];ep=lambda version:max(r[version+'_geometry'][s]['elbow_p95_deg'] for s in ['left','right'])
        lines.append(f"| {m} | {ep('k1'):.2f} | {ep('k2'):.2f} | {r['k1']['geometric_fit_errors']['trunk_incline']['p95_deg']:.2f} | {r['k2']['geometric_fit_errors']['trunk_incline']['p95_deg']:.2f} | {max(r['k1_geometry'][s]['wrist_p95_m'] for s in ['left','right'])*100:.2f} → {max(r['k2_geometry'][s]['wrist_p95_m'] for s in ['left','right'])*100:.2f} |")
    lines+=['','角度与位置共同满足工程门的左右手臂帧覆盖如下；三份真实样例均具有合格标定。它表示对源骨架的拟合支持范围。', '', '| 格式 | 左臂有效帧 | 右臂有效帧 |', '| --- | ---: | ---: |']
    for m in ['motionlcm','omnicontrol','kimodo']:
        q=actual[m]['k2_geometry']
        lines.append(f"| {m} | {q['left']['source_fidelity_share']:.1%} | {q['right']['source_fidelity_share']:.1%} |")
    lines+=['','其他格式回归中，ARDY、CHOIS、CLoSD、PRIMAL 的上述几何误差保持原值。DART 的标定由受限转为合格，躯干误差接近原值，左肘 P95 从 6.57° 增至约 9.30°；这项局部退化单列保留，头胸原生旋转路径保持独立。', '', '开发对照分两组：第一组按记录 ID 的 SHA256 在 LAB 每人选一条，再各选一条飞机铆接和玻璃作业，取中间 5 秒；第二组为每位 LAB 参与者按文件名排序的首条记录，取起始 5 秒，补查标定覆盖。选取不依赖拟合误差。两组共 22 条记录、每种格式 22 次导出；共 66 对 K1/K2 拟合。原始 BVH、固定目标骨架和格式读回的转换范围沿用 K1。','',
        '| 开发组 | 记录数 / 导出数 | K1 标定合格导出 | K2 标定合格导出 |','| --- | ---: | ---: | ---: |']
    for name,c in cohorts.items():lines.append(f"| {name} | {c['recordings']} / {c['exports']} | {c['k1_qualified']} | {c['k2_qualified']} |")
    lines+=['','以下按原记录等权汇总，每条记录先取左右肘 P95 的较大值，再求均值。诊断拟合计入误差分布；对外有效覆盖另受标定门约束。此表只描述所抽开发动作的拟合损失。','',
            '| 格式 | K1 较大肘 P95 均值° | K2 较大肘 P95 均值° | K2 退化超过 1° 的记录数 |','| --- | ---: | ---: | ---: |']
    for m in ['motionlcm','omnicontrol','kimodo']:
        rr=[r for r in pairs if r['method']==m];values={v:np.array([max(r[v+'_geometry'][s]['elbow_p95_deg'] for s in ['left','right']) for r in rr]) for v in ['k1','k2']}
        lines.append(f"| {m} | {values['k1'].mean():.2f} | {values['k2'].mean():.2f} | {int((values['k2']>values['k1']+1).sum())} / {len(rr)} |")
    lines+=['','三份高误差真实样例改善明显，跨人开发平均改善较小，仍存在局部退化。最大退化来自 LAB 的 `S8P05R1`：MotionLCM/OmniControl 的较大肘 P95 从约 11.2° 增至约 18.5°。该片段的 K1 全身标定原为受限，K2 能建立标定，但其关节中心几何仍有拟合损失；对应失效帧保留在逐侧掩码中。Kimodo 最大退化为 `S6P01R1`，约 0.7°→3.0°。这些记录已保留，后续优化需要同时看覆盖、尾部误差与手部作用点。','',
            '全部 5°/10°/20° 比例、中位数、P95、有符号偏差和帧数均在逐记录对照中。开发表计入诊断拟合以展示完整误差分布；可用的拟合输出仍须通过标定和逐侧保真门。']
    lines+=['',f'![输入、K1 与 K2 手臂叠加及肘角误差]({figure.resolve()})','',
      '图中左列选择 K1 肘角误差最大的帧，展示源肩肘腕与两版拟合；右列展示样例的有符号肘角误差。绘图使用 10 Hz 抽样，统计使用全部 100 Hz 输出，以保留短时失效。选帧规则固定，避免只展示改进最大的时刻。','',
      '## FBLS 路径、记录与复现','',
      '`config/biomechanics_models.json` 统一声明 FBLS/HYOID 模型位置，`motiogen.model_resources` 复用既有 `MORIONGEN_DATA_ROOT` 覆盖机制。upper_body 与 evalkit 共用解析结果；默认路径已指向当前数据盘。使用连续弯腰的合成开发输入验证了模型直接加载及 `unreliable_kinematics` 返回，未运行腰部批量肌肉实验。','',
      f'完整逐记录对照见 [comparison.json]({(a.evidence/"comparison.json").resolve()})；版本、测试、缓存、原冻结记录不变性及空间审计见同目录 `verification.json`。逐帧及模型产物留在 `{a.data}`。','',
      '```bash',
      f'/home/heinrich/.local/share/opensim-api/bin/python python/scripts/evalkit_k_arm_audit.py --phase prepare --out {a.data}/development',
      f'/home/heinrich/.local/share/opensim-api/bin/python python/scripts/evalkit_k_arm_audit.py --phase fit --out {a.data}/development --version k2_adaptive --workers 3',
      '# 复现 K1 时指定 --repo 到 acb48ac 的独立检出目录；补充组 prepare 加 --cohort first_records。',
      '```','',
      '完成 38 项自动检查、74 份当前版本拟合记录核验和八份缓存复用检查。38 项包含 K 及新增手臂/路径检查 30 项、工效规则 8 项；实际接口回归另含外力、工效及 11 帧下肢肌肉求解。全部试跑产物约 5.15 GB，数据盘核查时剩余约 922 GB。原冻结配置与留出使用记录通过逐字节核对。', '', '本轮结果将几何拟合改善与有效覆盖提升分别记录。下一步优先研究 OmniControl 骨长变化与剩余肘部误差之间的关系，以及稳定保留手部作用点的方向/位置联合拟合；肩部肌肉负荷继续使用独立的验证与适用范围。']
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
