#!/usr/bin/env python3
"""Original/degraded/projected/fitted geometry and fixed-window error overlays."""
import argparse,json,sys,re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from motiogen.evalkit.k.bones import normalize,NAMES,PARENTS
from motiogen.evalkit.k.scoring import angles
from evalkit_k3_report import record_score
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':10})
COLORS={'R':'#15293c','D':'#2279b5','P':'#9c68bb','K2':'#8a949d','K3':'#cf5629'}
STYLES={'R':'-','D':'--','P':':','K2':'-','K3':'-'}


def overlay(row,variant,path,reason):
    with np.load(row['arrays']) as z:a={k:z[k] for k in z.files}
    t=a['times'];curves={'R':a['R'],'D':a['D'],'P':a['P:'+variant],'K2':a['F:legacy_raw'],'K3':a['F:'+variant]}
    values={k:angles(p) for k,p in curves.items()}
    error=sum(abs(values['K2'][n]-values['R'][n]) for n in ['trunk_incline','left_elbow_flexion','right_elbow_flexion','left_knee_flexion','right_knee_flexion'])
    index=int(np.nanargmax(error))
    fig,axes=plt.subplots(2,3,figsize=(15,9),layout='constrained')
    for ax,normalized in zip(axes[0,:2],[False,True]):
        data={k:normalize(p,a['reference_lengths']) if normalized else p for k,p in curves.items()}
        if normalized:data={k:p-p[:,:1] for k,p in data.items()}
        base=data['R'][index];_,_,vh=np.linalg.svd(base[:,[0,2]]-base[:,[0,2]].mean(0));horizontal=vh[0]
        for key,p in data.items():
            xy=np.stack([p[index,:, [0,2]].T@horizontal,p[index,:,1]],1)
            for j,parent in enumerate(PARENTS):
                if parent>=0:ax.plot(xy[[parent,j],0],xy[[parent,j],1],color=COLORS[key],ls=STYLES[key],lw=2 if key in ['R','K3'] else 1,alpha=.85)
            ax.scatter(xy[:,0],xy[:,1],s=9,color=COLORS[key],label=key)
        ax.set_aspect('equal');ax.grid(alpha=.2);ax.set_xlabel('水平主投影 / m');ax.set_ylabel('高度 / m')
        ax.set_title(('统一骨长、各自去根平移' if normalized else '原始几何、世界位置')+f'\nt={t[index]:.2f} s')
    for ax,metric,title in [(axes[0,2],'trunk_incline','躯干倾斜'),(axes[1,0],'left_elbow_flexion','左肘几何屈曲'),(axes[1,1],'left_knee_flexion','左膝几何屈曲')]:
        for key,v in values.items():ax.plot(t,v[metric],color=COLORS[key],ls=STYLES[key],label=key,lw=1.5,alpha=.9)
        ax.set_title(title);ax.set_xlabel('相对时间 / s');ax.set_ylabel('角度 / °');ax.grid(alpha=.2)
    ax=axes[1,2];j=NAMES.index('left_wrist')
    for key in ['D','P','K2','K3']:
        p=normalize(curves[key],a['reference_lengths']);r=normalize(curves['R'],a['reference_lengths'])
        delta=(p[:,j]-p[:,0])-(r[:,j]-r[:,0]);ax.plot(t,np.linalg.norm(delta,axis=1)*100,color=COLORS[key],ls=STYLES[key],label=key)
    ax.set_title('统一骨长后的左手位置误差');ax.set_xlabel('相对时间 / s');ax.set_ylabel('去根平移距离 / cm');ax.grid(alpha=.2)
    axes[0,2].legend(ncol=5,fontsize=8,loc='lower right')
    projection='开启' if variant.endswith('projected') else '关闭，P=D'
    fig.suptitle(f'{row["method"]} · {Path(row["id"]).stem} · {reason}\nR 原始参考；D 退化读回；P 投影视图（{projection}）；K2 基准拟合；K3 {variant}',fontsize=13)
    fig.savefig(path,dpi=140);plt.close(fig)
    return {'id':row['id'],'method':row['method'],'variant':variant,'reason':reason,'frame_s':float(t[index]),'path':str(path),'frame_rule':'largest baseline sum of trunk/left-right elbow/knee errors; independent of improvement'}


def overview(summary,path):
    methods=['motionlcm','omnicontrol','ardy','chois','closd','dart','kimodo','primal']
    fig,axes=plt.subplots(1,2,figsize=(14,6),layout='constrained');y=np.arange(len(methods))
    base=[summary[m]['variants']['legacy_raw']['body_pose_score'] for m in methods]
    selected=[summary[m]['variants'][summary[m]['selected']]['body_pose_score'] for m in methods]
    ax=axes[0]
    for i,(b,s) in enumerate(zip(base,selected)):
        ax.plot([b,s],[i,i],color='#b7c2cb',lw=3)
        ax.annotate(f'{b:.2f} → {s:.2f}',(max(b,s)+.2,i),va='center',fontsize=9)
    ax.scatter(base,y,color=COLORS['K2'],label='原拟合，同一批修正输入',s=40,zorder=3)
    ax.scatter(selected,y,color=COLORS['K3'],label='K3 选定设置',s=40,zorder=3)
    ax.set_yticks(y,methods);ax.invert_yaxis();ax.set_xlim(0,max(base+selected)*1.45)
    ax.set_xlabel('原始参考端到端角度 P95 汇总 / °');ax.grid(axis='x',alpha=.2);ax.legend(loc='lower right',fontsize=9)
    ax.set_title('同一参考与评分窗口上的优化变化')
    matrix=[]
    for m in methods:
        r=summary[m]['variants'][summary[m]['selected']]['regions']
        matrix.append([float(np.mean([v[stage] for v in r.values()])) for stage in ['degradation','fit_deviation','end_to_end']])
    matrix=np.asarray(matrix);ax=axes[1];im=ax.imshow(matrix,cmap='Blues',aspect='auto',vmin=0)
    for i in range(len(methods)):
        for j in range(3):ax.text(j,i,f'{matrix[i,j]:.2f}°',ha='center',va='center',color='white' if matrix[i,j]>.6*matrix.max() else '#162b40')
    ax.set_yticks(y,methods);ax.set_xticks(range(3),['降级 D−R','拟合 F−D','端到端 F−R']);ax.set_title('三种误差独立统计，P95 不作加法分解')
    fig.colorbar(im,ax=ax,label='角度 P95 汇总 / °',shrink=.85)
    fig.suptitle('八种格式：原始参考验收与误差来源\n各记录角度 P95 → 六部位等权 → 按人等权；开发资料，LAB 与工业迁移合并展示',fontsize=13)
    fig.savefig(path,dpi=150);plt.close(fig)


def fast_head(rows,path,criterion='p95'):
    """Select by original head angular speed, then inspect all native rotation paths."""
    from scipy.spatial.transform import Rotation,Slerp
    from motiogen.evalkit.k import load,evaluate
    from motiogen.evalkit.k.export import target_template
    from motiogen.evalkit.k.calibration import UPRIGHT
    from motiogen.evalkit.k.geometry import frame
    candidates=[];seen=set()
    for row in rows:
        if row['window_key'] in seen:continue
        seen.add(row['window_key'])
        with np.load(row['reference']) as z:
            names=list(z['names']);rot=z['rotations'][:,names.index('Head')];t=z['times']
        speed=np.degrees(Rotation.from_matrix(rot[:-1].swapaxes(-1,-2)@rot[1:]).magnitude())/np.diff(t)
        peak=int(np.argmax(speed))
        candidates.append({'p95':float(np.percentile(speed,95)),'peak':float(np.max(speed)),
            'peak_interval_index':peak,
            'peak_interval_s':[float(t[peak]-t[0]),float(t[peak+1]-t[0])],
            'window_key':row['window_key'],'row':row})
    selected=max(candidates,key=lambda v:(v[criterion],v['window_key']))
    window,chosen=selected['window_key'],selected['row']
    with np.load(chosen['reference']) as z:
        names=list(z['names']);t=z['times'];basis=z['segment_basis']
        head=z['rotations'][:,names.index('Head')]@basis
        chest_index=next(names.index(n) for n in ['Spine3','Chest','Spine2'] if n in names)
        chest=z['rotations'][:,chest_index]@basis
        peak_index=selected['peak_interval_index']
        root_step=float(np.linalg.norm(z['points'][peak_index+1,0]-z['points'][peak_index,0]))
    fig,axes=plt.subplots(2,2,figsize=(14,8),layout='constrained')
    original={'head':head,'neck':chest.swapaxes(-1,-2)@head};checks=[]
    for col,part in enumerate(['head','neck']):
        angle=np.degrees(Rotation.from_matrix(original[part][0].T@original[part]).magnitude())
        axes[0,col].plot(t-t[0],angle,color=COLORS['R'],lw=2.5,label='R 原始参考')
        axes[0,col].set_title(('头部世界旋转' if part=='head' else '头相对胸旋转')+'：相对首帧的旋转幅度')
        axes[1,col].set_title(('头' if part=='head' else '颈')+'原生 K 路径相对 R 的旋转距离')
    for row in sorted([r for r in rows if r['window_key']==window],key=lambda r:r['method']):
        method=row['method']
        if method in ['motionlcm','omnicontrol']:continue
        motion=load(row['source'],method,config='config/evalkit_k3_20261116/experiment_config.json')
        names,rest,_=target_template(method);index={n:i for i,n in enumerate(names)}
        b=frame(np.array([0.,1,0]),rest[index['right_hip']]-rest[index['left_hip']])
        b=UPRIGHT@b if method=='closd' else b
        result=evaluate(motion,{'head':b,'thorax':b},config='config/evalkit_k3_20261116/experiment_config.json')
        check={'method':method}
        for col,part in enumerate(['head','neck']):
            item=result['metrics'][part];valid=item.get('anatomical_valid',item['valid']).copy()
            valid&=np.isfinite(item['rotations']).all((1,2))&(motion.times>=t[0])&(motion.times<=t[-1])
            q=motion.times[valid];measured=item['rotations'][valid]
            if not len(q):check[part]={'valid':0};continue
            reference=Slerp(t,Rotation.from_matrix(original[part]))(q).as_matrix()
            magnitude=np.degrees(Rotation.from_matrix(original[part][0].T@measured).magnitude())
            error=np.degrees(Rotation.from_matrix(reference.swapaxes(-1,-2)@measured).magnitude())
            axes[0,col].plot(q-t[0],magnitude,ls='--',lw=1,label=method)
            axes[1,col].plot(q-t[0],error,lw=1,label=method)
            check[part]={'valid':len(q),'maximum_error_deg':float(np.max(error))}
        checks.append(check)
    for ax in axes.flat:ax.set_xlabel('窗口内时间 / s');ax.set_ylabel('角度 / °');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8,ncol=2);axes[1,1].legend(fontsize=8,ncol=2)
    label='高角速度窗口核查' if criterion=='p95' else '源旋转跳变核查'
    fig.suptitle(f'{label} · {Path(chosen["id"]).stem} · 原始角速度 P95 {selected["p95"]:.1f}°/s，峰值 {selected["peak"]:.1f}°/s\n按原始运动选取；六路原生旋转，D1/D2 独立头朝向保留未知；受控段约定，HYOID 保持原状',fontsize=12)
    fig.savefig(path,dpi=140,bbox_inches='tight',pad_inches=.15);plt.close(fig)
    return {'path':str(path),'window_key':window,'id':chosen['id'],'selection':f'maximum original native-window head angular speed {criterion}; independent of fitting error',
            'original_head_peak_deg_s':selected['peak'],'original_head_p95_deg_s':selected['p95'],'peak_interval_s':selected['peak_interval_s'],
            'source_window_s':chosen['source_window_s'],'root_displacement_in_peak_interval_m':root_step,
            'scope':'native K orientations under the controlled segment convention, not anatomical neck validation','checks':checks,
            'candidates':[{**{k:v for k,v in c.items() if k!='row'},'id':c['row']['id']} for c in candidates]}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--dest',type=Path,required=True);ap.add_argument('--preview',action='store_true');a=ap.parse_args();a.dest.mkdir(parents=True,exist_ok=True)
    aggregate={} if a.preview else json.loads((ROOT/'config/evalkit_k3_20261116/results_summary.json').read_text())
    paths=(a.out/'fits').glob('*/*/result.json') if a.preview else [Path(p) for p in aggregate['result_paths']]
    rows=[json.loads(p.read_text()) for p in paths];rows=[r for r in rows if r['status']=='computed' and len(r['scores'])==4]
    summary={} if a.preview else aggregate['methods']
    chosen=[]
    for method in sorted({r['method'] for r in rows}):
        pool=sorted([r for r in rows if r['method']==method],key=lambda r:record_score(r,'legacy_raw'))
        chosen.extend([(pool[len(pool)//2],'典型基准误差'),(pool[-1],'最高基准误差')])
        if not a.preview:
            variant=summary[method]['selected'];worst=max(pool,key=lambda r:record_score(r,variant)-record_score(r,'legacy_raw'))
            chosen.append((worst,'最大退化／最小改善'))
    # Cover source-labelled standing bend, sitting, overhead reach, and kneeling.
    for lo,hi,label in [(7,9,'深弯腰任务'),(15,20,'座椅任务'),(13,14,'举手过头任务'),(21,28,'跪姿任务')]:
        pool=[r for r in rows if r['subset']=='LAB' and r['method']=='motionlcm' and lo<=int(re.search(r'P(\d+)',Path(r['id']).name)[1])<=hi]
        if pool:chosen.append((sorted(pool,key=lambda r:r['id'])[0],label))
    receipts=[]
    for i,(row,reason) in enumerate(chosen):
        v='optimized_raw' if a.preview else summary[row['method']]['selected']
        receipts.append(overlay(row,v,a.dest/f'{i:02d}_{row["method"]}_{row["window_key"]}.png',reason))
    overview_path=None
    if not a.preview:
        overview_path=a.dest/'overview.png';overview(summary,overview_path)
    head=None if a.preview else fast_head(rows,a.dest/'head_fast_turn.png')
    jump=None if a.preview else fast_head(rows,a.dest/'head_source_jump.png','peak')
    (a.dest/'index.json').write_text(json.dumps({'preview':a.preview,'source_task_labels':'https://zenodo.org/records/6472155','overview':str(overview_path) if overview_path else None,'figures':receipts,'fast_head':head,'source_jump':jump},ensure_ascii=False,indent=2)+'\n')
    print('figures',len(receipts))

if __name__=='__main__':main()
