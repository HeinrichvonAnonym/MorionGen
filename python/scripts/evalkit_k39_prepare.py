"""Freeze registered development identities and build versioned common inputs."""
from evalkit_k35_legs import *
from evalkit_k33_conversion import reference,interp,source_hashes
from motiogen.evalkit.k.export import retarget,serialize,ALIASES
from motiogen.evalkit.k.motion import load
OUT=BASE/'evalkit_k39_20261117'
def main():
 OUT.mkdir(exist_ok=True);assert shutil.disk_usage(OUT).free>80*1024**3
 rows=[read(p) for p in read(ROOT/'config/evalkit_k3_20261116/results_summary.json')['result_paths']]
 selected={(r['method'],r['window_key']) for r in read(ROOT/'config/evalkit_k34_20261117/acceptance.json')['selected']}
 selected|={(r['method'],r['window']) for r in inventory()};assert len(selected)==18 and len(rows)==656
 frozen={'rows':rows,'selected':sorted(selected),'references':{r['reference']:digest(r['reference']) for r in rows},'protected':{str(p.relative_to(ROOT)):digest(p) for d in ['config','plans'] for p in (ROOT/d).rglob('*') if p.is_file() and 'k39' not in str(p)},'initial_commit':'59411e6','scope':'82 fixed development windows, eight formats; no holdout; selected 18 pairs unchanged','runner':digest(__file__)}
 path=OUT/'source_freeze.json'
 if path.exists():assert read(path)==frozen
 else:atomic(path,frozen)
 atomic(ROOT/'config/evalkit_k39_20261117/selection.json',{'selected':sorted(selected),'reference_files':frozen['references'],'scope':frozen['scope']})
 exports=[]
 for i,row in enumerate(rows):
  work=OUT/'inputs'/row['window_key']/row['method'];receipt=work/'export.json'
  if receipt.exists():exports.append(read(receipt));continue
  try:
   target,parents=retarget(reference(row),row['method'],policy='direction_v3',root_anchor='hip_midpoint');source=serialize(target,parents,work)
   motion=load(source,row['method'],config='config/evalkit_k34_20261117/runtime_config.json')
   with np.load(row['arrays']) as z:t=z['times'];R=z['R'];lengths=z['reference_lengths']
   D=common(interp(motion.times-motion.times[0],motion.positions,t),[ALIASES.get(n,n) for n in motion.names])
   assert np.isfinite(D).all(),'incomplete original clock'
   np.savez_compressed(work/'common.npz',times=t,R=R,D=D,reference_lengths=lengths)
   result={**row,'source':str(source),'selected':(row['method'],row['window_key']) in selected,'common':str(work/'common.npz'),'status':'exported','hashes':source_hashes(source),'root_anchor':target.provenance['root_anchor'],'policy':'direction_v3','clock':'original fixed comparison clock'}
  except Exception as e:result={**row,'status':'failed','reason':str(e),'selected':(row['method'],row['window_key']) in selected}
  work.mkdir(parents=True,exist_ok=True);atomic(receipt,result);exports.append(result)
  if (i+1)%32==0:print(i+1,len(rows),flush=True)
 atomic(OUT/'exports.json',exports);print('complete',len(exports),'failed',sum(r['status']!='exported' for r in exports),flush=True)
if __name__=='__main__':main()
