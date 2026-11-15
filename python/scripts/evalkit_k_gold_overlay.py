#!/usr/bin/env python3
"""Development BVH -> target rig -> full-body IK overlay; fixed typical/adverse records."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from motion_retarget.bvh import parse_bvh
from motiogen.evalkit.k.reference import archive
from motiogen.evalkit.k.motion import Motion
from motiogen.evalkit.k.export import retarget,serialize,ALIASES
from motiogen.evalkit.k.geometry import frame,distance
from evalkit_k_regression import check


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    roots=json.loads((ROOT/'config/gold_k_20261115/evidence.json').read_text())['local_roots']
    rows=[json.loads(l) for l in (ROOT/'config/gold_k_20261115/trials.jsonl').read_text().splitlines()]
    selected=[next(r for r in rows if r['status']=='included' and r['partition']=='development' and suffix in r['id']) for suffix in ('PLNS01P01R01.bvh','SWPCS01G01R34.bvh')]
    fig,axes=plt.subplots(2,2,figsize=(12,10),layout='constrained');receipts=[]
    for i,row in enumerate(selected):
        src=row['source'];b=parse_bvh(archive(str(Path(roots['collaborate'])/src['archive'])).read(src['member']))
        p,r=b.forward_kinematics(True);names=[j.name for j in b.joints];n=min(len(p),int(b.fps*10)+1)
        local=np.median(b.local_positions(True),0);rest=np.zeros_like(local)
        for k,joint in enumerate(b.joints):rest[k]=local[k] if joint.parent<0 else rest[joint.parent]+local[k]
        h,c,neck=(names.index(x) for x in ('Head','Spine3','Neck'))
        basis=frame(rest[neck]-rest[c],rest[names.index('RightShoulder')]-rest[names.index('LeftShoulder')])
        ref=Motion(p[:n]*.01,names,np.arange(n)/b.fps,'reference',r[:n],names,{'segment_basis':basis.tolist(),'scope':'full_body_published_bvh_cm'})
        proxy=frame(p[:n,neck]-p[:n,c],p[:n,names.index('RightShoulder')]-p[:n,names.index('LeftShoulder')])
        reference_discrepancy=float(np.nanpercentile(distance(r[:n,c]@basis,proxy),95))
        for j,method in enumerate(('ardy','dart')):
            if ref.provenance.get('coordinate_precision'):
                ax=axes[i,j];ax.set_axis_off()
                magnitude=float(np.max(np.abs(p[:n]))*.01)
                ax.text(.04,.85,f'{Path(src["member"]).stem} / {method}',transform=ax.transAxes,fontsize=12,weight='bold')
                ax.text(.04,.63,f'World coordinate magnitude: {magnitude:.2e} m\nLocal vertical differences lost in floating-point FK.',transform=ax.transAxes,fontsize=11,linespacing=1.6)
                ax.text(.04,.35,'Position-dependent export and IK: unavailable\nNative segment rotations: retained separately',transform=ax.transAxes,fontsize=11,linespacing=1.6,color='#bd3e2b')
                receipts.append({'id':row['id'],'method':method,'status':'world_positions_unreliable','magnitude_m':magnitude,'precision':ref.provenance['coordinate_precision']})
                continue
            folder=a.out/f'{i}_{method}';target,parents=retarget(ref,method);source=serialize(target,parents,folder/'input')
            ax=axes[i,j]
            if reference_discrepancy>20:
                import biomech
                from motiogen.evalkit.k import load
                native=load(source,method);points,native_names,_=native.legacy()
                tn=[v[2] for v in biomech.MARKERS.values()]
                tp=points[:,[native_names.index(x) for x in tn]];fp=None;index=0;t=float(native.times[0])
                receipt={'status':'reference_geometry_conflict','method':method,'reference_proxy_p95_deg':reference_discrepancy,
                         'reason':'native chest frame and published positions disagree; fit overlay withheld',
                         'calibration':{'qualified':False},'ik':{'max_m':None}}
            else:
                receipt=check((method,str(source),str(folder/'evaluation'),True))
                if receipt['status']!='computed':ax.text(.1,.5,receipt['reason']);continue
                with np.load(folder/'evaluation/overlay.npz') as z:times=z['times'];tn=list(z['names']);tp=z['source'];fp=z['fitted']
                index=int(np.argmax(np.linalg.norm(tp-fp,axis=-1).max(-1)));t=times[index]
            receipts.append({'id':row['id'],**receipt})
            sn=[ALIASES.get(name,name) for name in names]
            rp=np.array([[np.interp(t,ref.times,ref.positions[:,sn.index(name),k]) for k in range(3)] for name in tn])
            _,_,vh=np.linalg.svd(rp[:,[0,2]]-rp[:,[0,2]].mean(0));direction=vh[0]
            traces=[(rp,'#2a9d8f','published BVH'),(tp[index],'#247ba0','adapted target')]
            if fp is not None:traces.append((fp[index],'#e76f51','OpenSim fit'))
            else:ax.text(.03,.03,'Reference geometry conflict; fit withheld',transform=ax.transAxes,fontsize=8,color='#bd3e2b')
            for points,color,label in traces:
                xy=np.stack([points[:,[0,2]]@direction,points[:,1]],1)
                for side in ('left','right'):
                    for chain in [('hip','knee','ankle','foot'),('hip','shoulder','elbow','wrist')]:
                        ids=[tn.index(f'{side}_{x}') for x in chain];ax.plot(xy[ids,0],xy[ids,1],'-o',ms=3,lw=1,color=color)
                ax.scatter([],[],c=color,label=label)
            shown=np.concatenate([rp,tp[index]],0)
            horizontal=shown[:,[0,2]]@direction
            ax.set_xlim(float(horizontal.min())-.15,float(horizontal.max())+.15)
            ax.set_ylim(float(shown[:,1].min())-.15,float(shown[:,1].max())+.15)
            if receipt['ik']['max_m'] is not None and receipt['ik']['max_m']>1:
                ax.text(.03,.03,'IK outside geometric gate; distant points clipped',transform=ax.transAxes,fontsize=8,color='#bd3e2b')
            ax.set_aspect('equal');ax.legend(fontsize=8);ax.grid(alpha=.15)
            ax.set_title(f'{Path(src["member"]).stem} / {method}\nqualified calibration: {receipt["calibration"]["qualified"]}; t={t:.2f}s',fontsize=10)
            ax.set_xlabel('horizontal projection (m)');ax.set_ylabel('height (m)')
    fig.suptitle('Development reference -> adapted target rig -> OpenSim fit\nFixed target body shape changes joint positions; source BVH axes require separate quality checks')
    fig.savefig(a.out/'gold_overlays.png',dpi=150)
    (a.out/'summary.json').write_text(json.dumps(receipts,indent=2,allow_nan=False))


if __name__=='__main__':main()
