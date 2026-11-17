"""Isolate hip coordinate cuts and validate the minimal unwrapped chart."""
from evalkit_k37_probe import *
DEST=BASE/'evalkit_k37_chart_20261117'
CHARTS={}
for name in ['flexion_chart','axial_chart','hip_unwrapped']:
 spec=deepcopy(PROFILES['combined'])
 if name in ['flexion_chart','hip_unwrapped']:spec['ranges']['hip_flexion']=(-360,360)
 if name in ['axial_chart','hip_unwrapped']:spec['ranges']['hip_rotation']=(-360,360)
 CHARTS[name]=spec
PROFILES.update(CHARTS)
def main():
 global DEST
 parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=DEST)
 DEST=parser.parse_args().out
 for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:assert os.environ.get(key)=='1'
 DEST.mkdir(exist_ok=True);row=next(r for r in inventory() if r['window']=='074')
 frozen={'row':row,'profiles':CHARTS,'parameters':PARAMETERS,'inputs':{row[k]:digest(row[k]) for k in ['model','comparison','kinematics']},'implementation':implementation_key(),'runner':digest(__file__),'scope':'development-only hip coordinate ablation; fixed reference and full window'}
 path=DEST/'freeze.json'
 if path.exists():assert read(path)==frozen,'Frozen experiment differs; select a new --out directory'
 else:atomic(path,frozen)
 with ProcessPoolExecutor(max_workers=3) as pool:results=list(pool.map(run_domain,[(row,p,{**PARAMETERS,'_output':str(DEST)}) for p in CHARTS]))
 atomic(DEST/'index.json',results)
if __name__=='__main__':main()
