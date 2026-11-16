"""Development-only arm branch continuation ablation on frozen K3 inputs."""
import argparse, json, os, sys, hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'python/scripts'))
from evalkit_k31_full import one, read, atomic
from motiogen.evalkit.k.config import implementation_key, digest
BASE=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k32_branch_20261116')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pilot',action='store_true');a=ap.parse_args()
    for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']:assert os.environ.get(key)=='1'
    rows=[read(p) for p in read(ROOT/'config/evalkit_k3_20261116/results_summary.json')['result_paths']]
    # Fixed failure cases plus the previously registered typical cases; no new selection by fit error.
    cases={('kimodo','077'),('kimodo','070'),('kimodo','073'),('ardy','073'),('dart','073'),('kimodo','074'),('closd','081')}
    typical=read(ROOT/'develop_logs/notes/20261116_k3_figures/index.json')['figures']
    for c in typical:
        if c['reason']=='典型基准误差':
            r=next(r for r in rows if r['id']==c['id'] and r['method']==c['method']);cases.add((r['method'],r['window_key']))
    selected=[r for r in rows if (r['method'],r['window_key']) in cases]
    cfg=read(ROOT/'config/evalkit_k_v3_1.json')
    for setting in cfg['fitting']['per_method'].values():setting['continue_arm_branch']=True
    evidence={'config':cfg,'implementation':implementation_key(),'runner':digest(__file__),
              'selection':[{'id':r['id'],'method':r['method'],'window_key':r['window_key']} for r in selected],
              'scope':'development ablation; original R acceptance; unchanged K3.1 settings except arm seed ordering'}
    freeze=hashlib.sha256(json.dumps(evidence,sort_keys=True).encode()).hexdigest()
    BASE.mkdir(parents=True,exist_ok=True)
    if (BASE/'freeze.json').exists():assert read(BASE/'freeze.json')==evidence
    else:atomic(BASE/'freeze.json',evidence)
    if a.pilot:selected=[r for r in selected if r['method']=='kimodo' and r['window_key']=='077']
    with ProcessPoolExecutor(max_workers=3) as pool:
        for result in pool.map(one,[(r,BASE,cfg,freeze) for r in selected]):
            print(result['method'],result['id'],result['status'],flush=True)
if __name__=='__main__':main()
