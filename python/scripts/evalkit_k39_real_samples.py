"""Eight registered real outputs: input fidelity only, no original-motion claim."""
from evalkit_k39_prepare import *
from evalkit_k39_regression import CONFIG

def one(task):
 from motiogen.evalkit.k.selection import fit
 method,record,out=task;work=Path(out)/method;work.mkdir(parents=True,exist_ok=True)
 try:
  selection=fit(record['sample'],work,CONFIG)
  result={'method':method,'status':'computed','sample':record['sample'],'sample_sha256':digest(record['sample']),'selection':selection,'scope':'real generated output input fidelity; independent original motion unavailable'}
 except Exception as e:
  import traceback
  (work/'error.log').write_text(traceback.format_exc());result={'method':method,'status':'failed','sample':record['sample'],'reason':str(e)}
 atomic(work/'result.json',result);print(method,result['status'],flush=True);return str(work/'result.json')

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--workers',type=int,default=3);a=parser.parse_args();a.out.mkdir(exist_ok=True)
 records=read(ROOT/'config/evalkit_k3_20261116/source_contract_checks.json')['records']
 frozen={'implementation':implementation_key(),'config':read(ROOT/CONFIG),'samples':{m:digest(r['sample']) for m,r in records.items()},'runner':digest(__file__)}
 path=a.out/'freeze.json'
 if path.exists():assert read(path)==frozen
 else:atomic(path,frozen)
 with ProcessPoolExecutor(max_workers=min(a.workers,3)) as pool:results=list(pool.map(one,[(m,r,a.out) for m,r in records.items()]))
 atomic(a.out/'index.json',results);assert frozen['implementation']==implementation_key()
if __name__=='__main__':main()
