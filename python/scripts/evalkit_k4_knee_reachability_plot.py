"""Static reproducible model-range versus original-R evidence figure."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=json.loads((a.audit/'original_R_bound.json').read_text());fit=json.loads((a.audit/'result.json').read_text())
    with np.load(a.audit/'reachability.npz') as z:data={k:z[k] for k in z.files}
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
    for ax,side in zip(axes,['left','right']):
        lo,hi=r['sides'][side]['R_min_max_deg'];maximum=fit['sides'][side]['model_geometric_range_deg'][1]
        ax.axhspan(lo,hi,color='#7357a6',alpha=.18,label='Original R observed range')
        ax.plot(data[side+'_q_deg'],data[side+'_model_flex_deg'],color='#207791',lw=2.4,label='Fixed-model FK geometry')
        ax.axhline(maximum,color='#bb493a',ls='--',label=f'Model maximum {maximum:.2f}°')
        ax.axhline(maximum+5,color='#ad7c1d',ls=':',label='Model maximum + 5° tolerance')
        bound=r['sides'][side]['F-R_lower_bound_p95_deg']
        ax.text(.04,.53,f'F−R error lower bound\nP95 = {bound:.2f}°\n150 / 150 frames exceed 5°',transform=ax.transAxes,fontsize=11,
                bbox={'facecolor':'white','edgecolor':'#d1d5db','alpha':.95,'boxstyle':'round,pad=.5'})
        ax.set_title(side.capitalize()+' knee · window 074 / Kimodo');ax.set_xlabel('Model knee coordinate (degrees)')
        ax.set_xlim(0,140);ax.set_ylim(0,185);ax.grid(alpha=.16);ax.legend(loc='lower right',fontsize=8)
    axes[0].set_ylabel('Geometric hip–knee–ankle flexion (degrees)')
    fig.suptitle('Original-model reachability audit · fixed geometry · original R used only in scoring',fontsize=12)
    fig.tight_layout();a.output.parent.mkdir(parents=True,exist_ok=True)
    for ext in ['png','svg']:fig.savefig(a.output.with_suffix('.'+ext),dpi=170,bbox_inches='tight')


if __name__=='__main__':main()
