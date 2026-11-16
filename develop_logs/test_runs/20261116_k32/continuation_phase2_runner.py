"""Second ablation: retain an adequate previous-frame arm solution."""
import argparse,json,sys,hashlib,os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'python/scripts'))
from evalkit_k31_full import one,read,atomic
from motiogen.evalkit.k.config import implementation_key,digest
BASE=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pilot',action='store_true');a=ap.parse_args()
    for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']:assert os.environ.get(key)=='1'
    prior=read(BASE/'evalkit_k32_branch_20261116/freeze.json');cfg=prior['config']
    for s in cfg['fitting']['per_method'].values():s['accept_continued_arm']=True
    rows=[read(p) for p in read(ROOT/'config/evalkit_k3_20261116/results_summary.json')['result_paths']]
    chosen={(c['method'],c['window_key']) for c in prior['selection']};rows=[r for r in rows if (r['method'],r['window_key']) in chosen]
    evidence={'config':cfg,'implementation':implementation_key(),'runner':digest(__file__),'selection':prior['selection'],
        'scope':'same fixed fifteen development probes; accept continued arm below existing .01 squared objective threshold; no reference in fitting'}
    freeze=hashlib.sha256(json.dumps(evidence,sort_keys=True).encode()).hexdigest();out=BASE/'evalkit_k32_continue_20261116';out.mkdir(exist_ok=True)
    if (out/'freeze.json').exists():assert read(out/'freeze.json')==evidence
    else:atomic(out/'freeze.json',evidence)
    if a.pilot:rows=[r for r in rows if r['method']=='kimodo' and r['window_key']=='077']
    with ProcessPoolExecutor(max_workers=2) as pool:
        for r in pool.map(one,[(r,out,cfg,freeze) for r in rows]):print(r['method'],r['window_key'],r['status'],flush=True)
if __name__=='__main__':main()
