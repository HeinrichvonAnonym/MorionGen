#!/usr/bin/env python3
"""Reproducible development-only overlays of marker reconstruction and full-body fitting."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from motiogen.evalkit.k.reference import andy,marker_frames,template_fit
from motiogen.evalkit.k.geometry import distance


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--results',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--development-name',default='development_candidate');ap.add_argument('--regression-name',default='regression_candidate')
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    methods=['ardy','motionlcm','omnicontrol','kimodo','chois','closd','dart','primal']
    fig,axes=plt.subplots(2,4,figsize=(17,9),layout='constrained')
    for ax,m in zip(axes.flat,methods):
        with np.load(a.results/a.regression_name/m/'overlay.npz') as z:
            original,fitted,names,t=z['source'],z['fitted'],list(z['names']),z['times']
        i=int(np.argmax(np.linalg.norm(original-fitted,axis=-1).max(-1)))
        x=original[i][:,[0,2]];_,_,vh=np.linalg.svd(x-x.mean(0));direction=vh[0]
        for points,color,label in [(original[i],'#247ba0','source'),(fitted[i],'#e76f51','fitted joint centres')]:
            xy=np.stack([points[:,[0,2]]@direction,points[:,1]],1)
            for side in ('left','right'):
                for chain in [('hip','knee','ankle','foot'),('hip','shoulder','elbow','wrist')]:
                    idx=[names.index(f'{side}_{n}') for n in chain];ax.plot(xy[idx,0],xy[idx,1],'-o',color=color,ms=3,lw=1.5)
            ax.scatter([],[],c=color,label=label)
        ax.set_title(f'{m}: largest sampled position discrepancy\nt = {t[i]:.2f} s',fontsize=10);ax.set_aspect('equal');ax.grid(alpha=.15)
        ax.set_xlabel('horizontal projection (m)');ax.set_ylabel('height (m)')
    axes.flat[0].legend(fontsize=8,loc='upper right')
    fig.suptitle('Actual baseline outputs: source versus OpenSim fit\nGeometric diagnostic; fitted model joint centres and source rig points have distinct definitions',fontsize=14)
    fig.savefig(a.out/'body_overlays.png',dpi=150);plt.close(fig)
    rows=[json.loads(l) for l in (a.results/a.development_name/'records.jsonl').read_text().splitlines()]
    selected=sorted([r for r in rows if r['dataset']=='andy' and r.get('neck_template_comparison',{}).get('p95_deg') is not None],key=lambda r:r['neck_template_comparison']['p95_deg'])
    chosen=[selected[len(selected)//2],selected[-1]]
    manifest={r['id']:r for r in map(json.loads,(ROOT/'config/gold_k_20261115/trials.jsonl').read_text().splitlines())}
    roots=json.loads((ROOT/'config/gold_k_20261115/evidence.json').read_text())['local_roots']
    fig,axes=plt.subplots(2,2,figsize=(14,7),layout='constrained')
    for i,row in enumerate(chosen):
        t,names,p=andy(manifest[row['id']],roots['andy']);ref=marker_frames(p,names,'andy')
        for j,(region,(truth,points)) in enumerate(ref.items()):
            predicted,_,cal=template_fit(points,truth,int(round(1/np.median(np.diff(t)))))
            ax=axes[i,j];ax.plot(t,truth[:,1,1],label='four-marker reference',lw=.8,c='#247ba0')
            ax.plot(t,predicted[:,1,1],label='redundant-marker reconstruction',lw=.7,c='#e76f51',alpha=.8)
            ax.set_ylabel('segment up-axis, world vertical component');ax.set_xlabel('time (s)')
            err=distance(truth,predicted);err[cal]=np.nan
            other=ax.twinx();other.plot(t,err,c='#777777',alpha=.35,lw=.6);other.set_ylabel('orientation discrepancy (deg)')
            ax.set_title(f'{"Typical" if i==0 else "High discrepancy"}: {row["id"].split("/")[-1]}\n{region}',fontsize=10)
            ax.legend(fontsize=7,loc='lower left')
    fig.suptitle('Development optical reference/reconstruction overlays\nCalibration frames excluded from discrepancy scoring; natural missing reference remains blank',fontsize=13)
    fig.savefig(a.out/'optical_overlays.png',dpi=150);plt.close(fig)
    (a.out/'selection.json').write_text(json.dumps({'optical_records':[r['id'] for r in chosen],
        'selection':'Andy trial-P95 median and maximum; full-body highest sampled point discrepancy per actual baseline sample'},indent=2))


if __name__=='__main__':main()
