"""Summarize independent conversion and same-input solver comparisons."""
from evalkit_k34_development import *
from motiogen.evalkit.k.scoring import region

def metrics(row):
 z=np.load(Path(row['output'])/'comparison.npz');t=z['times'];a=angles(z['F:new']);ref=angles(z['R']);s=row['scores'];regions={}
 for reg in ['pelvis','torso','left_arm','right_arm','left_leg','right_leg']:
  names=[n for n in a if region(n)==reg]
  regions[reg]={stage:float(np.mean([s['raw']['angles'][n][stage]['p95_abs'] for n in names])) for stage in ['degradation','fit_deviation','end_to_end']}
  regions[reg].update(coverage=s['coverage'][reg]['share'],maximum_step_deg=max(float(abs(np.diff(a[n])).max()) for n in names),
   velocity_error_p95=float(np.mean([np.percentile(abs(np.gradient(a[n],t)-np.gradient(ref[n],t)),95) for n in names])))
 positions={space:{n:s[space]['positions'][n]['end_to_end']['world']['p95_abs'] for n in ['left_foot','right_foot','left_wrist','right_wrist']} for space in ['raw','normalized']}
 return {'regions':regions,'positions':positions}

def main():
 light=read(OUT/'light.json');old=read(BASE/'evalkit_k33_conversion_20261117/light.json');old={(r['method'],r['window_key']):r for r in old};agg={}
 for method in sorted({r['method'] for r in light}):
  rows=[r for r in light if r['method']==method];groups=sorted({r['group_id'] for r in rows})
  mean=lambda fun:float(np.mean([np.mean([fun(r) for r in rows if r['group_id']==g]) for g in groups]))
  agg[method]={'old_trunk':mean(lambda r:old[(method,r['window_key'])]['variants']['topology_v2']['trunk_incline']['p95_abs']),
   'new_trunk':mean(lambda r:r['angles']['trunk_incline']['p95_abs']),
   'infeasible_frames':sum(sum(not v for v in r['transfer']['corrected_frames']) for r in rows),
   'total_frames':sum(len(r['transfer']['corrected_frames']) for r in rows),
   'worse_gt1':sum(r['angles']['trunk_incline']['p95_abs']>old[(method,r['window_key'])]['variants']['topology_v2']['trunk_incline']['p95_abs']+1 for r in rows)}
 fits=[]
 for p in sorted((OUT/'fits').glob('*/*/*/result.json')):
  row=read(p)
  fits.append({k:row[k] for k in ['method','window_key','variant','status'] }|({'metrics':metrics(row),'output':row['output']} if row['status']=='computed' else {'reason':row['reason']}))
 pairs=[]
 for row in fits:
  if row['variant']!='solver' or row['status']!='computed':continue
  base=next((r for r in fits if r['method']==row['method'] and r['window_key']==row['window_key'] and r['variant']=='baseline'),None)
  if base:
   flags=[]
   for reg,v in row['metrics']['regions'].items():
    b=base['metrics']['regions'][reg]
    if v['end_to_end']>b['end_to_end']+1:flags.append(reg+':angle')
    if v['maximum_step_deg']>b['maximum_step_deg']+5:flags.append(reg+':step')
    if v['coverage']<b['coverage']-.01:flags.append(reg+':coverage')
   pairs.append({'method':row['method'],'window':row['window_key'],'flags':flags})
 summary={'light':agg,'fits':fits,'solver_pairs':pairs,'scope':'82x8 conversion; 16 paired complete fits and two frozen-input diagnostics; development only'}
 atomic(OUT/'summary.json',summary)
 print(json.dumps(agg,indent=2));print('fits',len(fits));print('flags',[r for r in pairs if r['flags']])
 for row in fits:
  if (row['method'],row['window_key']) in [('kimodo','010'),('kimodo','074'),('motionlcm','031'),('ardy','051')]:print(row['method'],row['window_key'],row['variant'],row.get('metrics',{}).get('regions'))
if __name__=='__main__':main()
