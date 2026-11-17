"""Audit native input contracts separately from end-to-end fitting."""
from evalkit_k39_prepare import *
from motiogen.evalkit.k.trustworthy import directions
from motiogen.evalkit.k.scoring import stats

def main():
 rows=read(OUT/'exports.json');results=[]
 for row in rows:
  if row['status']!='exported':results.append({'method':row['method'],'window':row['window_key'],'status':'failed','reason':row['reason']});continue
  z=np.load(row['common']);R=z['R'];D=z['D'];fields={}
  for n,j in [('hip_midpoint',0),('shoulder_midpoint',1),('left_shoulder',10),('right_shoulder',13)]:
   delta=D[:,j]-R[:,j];fields[n]={'p95_m':float(np.percentile(np.linalg.norm(delta,axis=1),95)),'bias_m':delta.mean(0).tolist()}
  for key in directions(R):fields[key]=stats(np.degrees(np.arccos(np.clip((directions(R)[key]*directions(D)[key]).sum(-1),-1,1))))
  results.append({'method':row['method'],'window':row['window_key'],'group':row['group_id'],'status':'computed','fields':fields,'source':row['source'],'hashes':source_hashes(row['source'])})
 atomic(OUT/'conversion_audit.json',results)
 summary={m:{'windows':sum(r['method']==m for r in results),'root_p95_median_m':float(np.median([r['fields']['hip_midpoint']['p95_m'] for r in results if r['method']==m and r['status']=='computed']))} for m in sorted({r['method'] for r in results})}
 contracts=read(ROOT/'config/evalkit_k3_20261116/source_contract_checks.json')
 atomic(ROOT/'config/evalkit_k39_20261117/source_audit.json',{'common_world':'right-handed Y-up, metres; original fixed clock','hip_midpoint':'geometric bilateral landmark; distinct from named rig root and anatomical pelvis origin','trunk_axis':'hip midpoint to shoulder midpoint; distinct from thorax native rotation and lumbar model coordinates','shoulder_line':'left-to-right shoulder endpoints; translation independent','neutral_torso':'format asset prior when no independently identifiable neutral geometry','definition_sources':{m:{'contract':r['native_definition'],'asset':r['controlled_fixture'].get('asset'),'sample':r['sample']} for m,r in contracts['records'].items()},'summary':summary,'full_data':str(OUT/'conversion_audit.json')})
 print(summary)
if __name__=='__main__':main()
