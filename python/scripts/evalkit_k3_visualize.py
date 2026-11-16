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
                if parent>=0:ax.plot(xy[[parent,j],0],xy[[parent,j],1],color=COLORS[key],lw=2 if key in ['R','K3'] else 1,alpha=.85)
            ax.scatter(xy[:,0],xy[:,1],s=9,color=COLORS[key],label=key)
        ax.set_aspect('equal');ax.grid(alpha=.2);ax.set_xlabel('水平主投影 / m');ax.set_ylabel('高度 / m')
        ax.set_title(('统一骨长、各自去根平移' if normalized else '原始几何、世界位置')+f'\nt={t[index]:.2f} s');ax.legend(ncol=3,fontsize=8)
    for ax,metric,title in [(axes[0,2],'trunk_incline','躯干倾斜'),(axes[1,0],'left_elbow_flexion','左肘几何屈曲'),(axes[1,1],'left_knee_flexion','左膝几何屈曲')]:
        for key,v in values.items():ax.plot(t,v[metric],color=COLORS[key],label=key,lw=1.5,alpha=.9)
        ax.set_title(title);ax.set_xlabel('相对时间 / s');ax.set_ylabel('角度 / °');ax.grid(alpha=.2)
    ax=axes[1,2];j=NAMES.index('left_wrist')
    for key in ['D','P','K2','K3']:
        p=normalize(curves[key],a['reference_lengths']);r=normalize(curves['R'],a['reference_lengths'])
        delta=(p[:,j]-p[:,0])-(r[:,j]-r[:,0]);ax.plot(t,np.linalg.norm(delta,axis=1)*100,color=COLORS[key],label=key)
    ax.set_title('统一骨长后的左手位置误差');ax.set_xlabel('相对时间 / s');ax.set_ylabel('去根平移距离 / cm');ax.grid(alpha=.2)
    fig.suptitle(f'{row["method"]} · {Path(row["id"]).stem} · {reason}\nR 原始参考；D 退化读回；P 拟合输入投影；K2 基准拟合；K3 {variant}',fontsize=13)
    fig.savefig(path,dpi=140);plt.close(fig)
    return {'id':row['id'],'method':row['method'],'variant':variant,'reason':reason,'frame_s':float(t[index]),'path':str(path),'frame_rule':'largest baseline sum of trunk/left-right elbow/knee errors; independent of improvement'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--dest',type=Path,required=True);ap.add_argument('--preview',action='store_true');a=ap.parse_args();a.dest.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(p.read_text()) for p in (a.out/'fits').glob('*/*/result.json')];rows=[r for r in rows if r['status']=='computed' and len(r['scores'])==4]
    summary={} if a.preview else json.loads((ROOT/'config/evalkit_k3_20261116/results_summary.json').read_text())['methods']
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
    (a.dest/'index.json').write_text(json.dumps({'preview':a.preview,'source_task_labels':'https://zenodo.org/records/6472155','figures':receipts},ensure_ascii=False,indent=2)+'\n')
    print('figures',len(receipts))

if __name__=='__main__':main()
