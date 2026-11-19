"""Summarize completed CPU candidate ablation and frozen-selector replay, read-only."""
import json,hashlib,gzip,itertools,collections,statistics,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
B=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k4_20261117');R=Path('/home/heinrich/MorionGen');OUT=B/'cpu_final_analysis_20261120_v1';OUT.mkdir(exist_ok=True)
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def quant(a,q):
 a=sorted(x for x in a if x is not None)
 if not a:return None
 x=(len(a)-1)*q;i=int(x);return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(x-i)
def stats(a):return dict(n=len(a),median=quant(a,.5),p95=quant(a,.95),max=max(a) if a else None)
raw=(B/'full_candidate_ablation_20261118_v1/progress.json').read_bytes();d=json.loads(raw);p=read(B/'full_candidate_ablation_20261118_v1/protocol.json');fp=read(Path(p['frozen_run'])/'protocol.json');routes=d['rows'];assert len(routes)==656 and len({(r['window'],r['method']) for r in routes})==656 and all(r['fully_evaluated'] for r in routes)
with gzip.open(OUT/'ablation_progress.json.gz','wb') as f:f.write(raw)
replay={f.parent.name:read(f) for f in (B/'selector_replay_20261120_v1/routes').glob('*/audit.json')};assert len(replay)==656
meta={r['window']+'_'+r['method']:r for r in fp['records']};res={'protocol_sha256':sha(B/'full_candidate_ablation_20261118_v1/protocol.json'),'progress_sha256':hashlib.sha256(raw).hexdigest(),'totals':{k:v for k,v in d.items() if not isinstance(v,(dict,list))},'by_format':{},'by_group':{},'paired_export':{},'unique_candidate_failures':collections.Counter(),'failure_metrics':collections.Counter(),'failure_regions':collections.Counter(),'selections':{},'format_candidate_counts':{},'zero_routes':[]}
for method in sorted({r['method'] for r in routes}):
 rr=[r for r in routes if r['method']==method];passing=[{n for n,c in r['candidates'].items() if c.get('passed')} for r in rr];names=sorted({n for r in rr for n,c in r['candidates'].items() if c['status']!='inapplicable' and not p['routes'][next(i for i,x in enumerate(p['routes']) if x['window']==r['window'] and x['method']==method)]['cells'][n].get('alias')});counts={n:sum(n in x for x in passing) for n in names};best=max(counts.values());masks={n:sum(1<<i for i,x in enumerate(passing) if n in x) for n in names};union=0
 for mask in masks.values():union|=mask
 covers=[]
 for size in range(1,len(names)+1):
  for comb in itertools.combinations(names,size):
   mask=0
   for n in comb:mask|=masks[n]
   if mask==union:covers.append(comb)
  if covers:break
 res['by_format'][method]={'n':len(rr),'any_pass':sum(bool(x) for x in passing),'all_pass':sum(r['all_candidates_passed'] for r in rr),'best_single_pass':best,'best_ties':[n for n,c in counts.items() if c==best],'posthoc_cover_size':size,'posthoc_cover_example':covers[0],'full_pass':sum(replay[r['window']+'_'+method]['full']['passed'] for r in rr),'budget_pass':sum(replay[r['window']+'_'+method]['budget']['passed'] for r in rr)};res['format_candidate_counts'][method]=counts
contracts={r['window']+'_'+r['method']:r for r in p['routes']}
for r in routes:
 key=r['window']+'_'+r['method']
 if not r['any_candidate_passed']:res['zero_routes'].append({'key':key,'metadata':meta[key],'audit':replay[key]})
 for n,c in r['candidates'].items():
  if c.get('passed') is not False or c['status']!='evaluated' or contracts[key]['cells'][n].get('alias'):continue
  failed=c.get('failures',[])
  def typ(x):
   metric=x.get('metric','')
   if any(v in metric for v in ('step_deg','rate_','velocity_','speed_')):return 'temporal'
   if x.get('status')=='target_failed':return 'pose'
   return 'coverage_or_other'
  res['unique_candidate_failures']['+'.join(sorted({typ(x) for x in failed}))]+=1
  res['unique_candidate_failures']['phase/native' if any(x['view']=='native' for x in failed) else 'phase/export_only']+=1
  for metric in {x['metric'] for x in failed}:res['failure_metrics'][metric]+=1
  for reg in {x['metric'].split('/')[0] for x in failed}:res['failure_regions'][reg]+=1
for n in p['templates']:
 if not n.endswith('__coordinated'):continue
 base=n.removesuffix('__coordinated');rr=[r for r in routes if r['candidates'][n]['status']=='evaluated' and r['candidates'][base]['status']=='evaluated' and not contracts[r['window']+'_'+r['method']]['cells'][n].get('alias')]
 res['paired_export'][n]={'n':len(rr),'rescued':sum(not r['candidates'][base]['passed'] and r['candidates'][n]['passed'] for r in rr),'regressed':sum(r['candidates'][base]['passed'] and not r['candidates'][n]['passed'] for r in rr)}
for group in sorted({r['group_id'] for r in meta.values()}):
 kk=[k for k,m in meta.items() if m['group_id']==group];res['by_group'][group]={'windows':len(kk)//8,'routes':len(kk),'full_pass':sum(replay[k]['full']['passed'] for k in kk),'budget_pass':sum(replay[k]['budget']['passed'] for k in kk)}
visits=[];available=[];early=0;selected_diff=0;selection_files={}
for key,a in replay.items():
 sf=B/'selector_replay_20261120_v1/routes'/key/'selection.json';s=read(sf);visits.append(a['visited']);available.append(a['available']);early+=s['early_accepted'] is not None;selected_diff+=a['full']['selected']!=a['budget']['selected'];selection_files[key]=sha(sf)
res['selector_summary']={'routes':656,'different_choices':selected_diff,'full_passed':sum(a['full']['passed'] for a in replay.values()),'budget_passed':sum(a['budget']['passed'] for a in replay.values()),'full_misses':sum(a['full']['missed_passing'] for a in replay.values()),'budget_misses':sum(a['budget']['missed_passing'] for a in replay.values()),'early_stop_loss':sum(a['early_stop_loss'] for a in replay.values()),'early_accepted':early,'full_pool_fallback':656-early,'visited_candidates':sum(visits),'available_unique_candidates':sum(available),'visited_distribution':stats(visits),'historical_paired':sum('historical_budget_match' in a for a in replay.values()),'historical_mismatches':sum(a.get('historical_budget_match') is False for a in replay.values())}
for label in ['full','budget']:res['selections'][label]=dict(collections.Counter(a[label]['selected'] for a in replay.values()))
# Independently read each selected export score once; no new fitting/scoring or candidate selection.
files={path:h for a in replay.values() for label in ['full','budget'] for path,h in a[label]['score_files'].items() if '/export/' in path}
def extract(item):
 path,h=item;raw=Path(path).read_bytes()
 if hashlib.sha256(raw).hexdigest()!=h:raise ValueError('Selected score hash changed')
 data=json.loads(raw)['original_clock'];out={}
 for comp in ('D-R','F-D','F-R'):
  v=data['comparisons'][comp]['full'];m={}
  for region,nodes in v['positions_cm'].items():
   for node,modes in nodes.items():
    for mode in ('world','common_length_world'):
     st=modes[mode];m['position/'+region+'/'+node+'/'+mode]={k:st.get(k) for k in ('median','p95','n','denominator','valid_fraction')}
  for kind in ('directions_deg','joint_angles_deg'):
   for node,st in v[kind].items():m[kind+'/'+node]={k:st.get(k) for k in ('median','p95','n','denominator','valid_fraction')}
  head=data.get('head_neck_rotation_comparisons',{}).get(comp,{}).get('full')
  if head:m['rotation/head_relative_thorax']={k:head.get(k) for k in ('median','p95','n','denominator','valid_fraction')}
  out[comp]=m
 return path,out
print('selected export files',len(files),flush=True)
with ThreadPoolExecutor(max_workers=2) as pool:metrics=dict(pool.map(extract,files.items()))
res['accuracy']={};res['accuracy_delta']={}
for method in res['by_format']:
 kk=[k for k in replay if k.split('_',1)[1]==method];res['accuracy'][method]={}
 for label in ['full','budget']:
  chosen=[metrics[next(p for p in replay[k][label]['score_files'] if '/export/' in p)] for k in kk];bins={}
  for comp in ('D-R','F-D','F-R'):
   for m in chosen:
    for metric,st in m[comp].items():
     if st['p95'] is not None and st['n']:bins.setdefault(comp+'/'+metric,[]).append(st)
  res['accuracy'][method][label]={k:{'window_median':stats([s['median'] for s in vals]),'window_p95':stats([s['p95'] for s in vals]),'frame_valid_fraction_min':min(s['valid_fraction'] for s in vals)} for k,vals in bins.items()}
 delta={}
 for k in kk:
  full,budget=[metrics[next(p for p in replay[k][label]['score_files'] if '/export/' in p)]['F-R'] for label in ['full','budget']]
  for metric in full.keys()&budget.keys():
   f,q=full[metric],budget[metric]
   if f['p95'] is not None and q['p95'] is not None and f['n'] and q['n']:delta.setdefault(metric,[]).append(q['p95']-f['p95'])
 res['accuracy_delta'][method]={m:dict(stats(v),worse=sum(x>1e-8 for x in v),better=sum(x<-1e-8 for x in v)) for m,v in delta.items()}
# Protected documents and actual runtime identities audited separately.
protected=read(R/'config/evalkit_k4_20261117/preservation.json')['files'];bad=[{'path':n,'expected':h,'actual':sha(R/n)} for n,h in protected.items() if sha(R/n)!=h]
actual={str(Path(p['code_root'])/'utils/motiogen/evalkit/k'/n):h for n,h in p['numerical_implementation'].items()}
for route in p['routes']:actual.update(route['expected_key']['inputs'])
actual[fp['fit_config']]=fp['fit_config_sha256'];actual_bad=[n for n,h in actual.items() if sha(n)!=h]
res['integrity']={'protected_checked':len(protected),'protected_mismatches':bad,'runtime_files_checked':len(actual),'runtime_mismatches':actual_bad,'ablation_completion_exists':(B/'full_candidate_ablation_20261118_v1/completion.json').exists(),'replay_completion_exists':(B/'selector_replay_20261120_v1/completion.json').exists()}
res['per_candidate']=d['per_candidate'];res['metadata']={'windows':82,'groups':len(res['by_group']),'subset_windows':dict(collections.Counter(m['subset'] for k,m in meta.items() if k.endswith('_ardy'))),'source_scope':'Fixed CoLLaboratE development windows; paired degradation formats; not independent final holdout'}
(OUT/'summary.json').write_text(json.dumps(res,indent=2));(OUT/'selection_source_hashes.json').write_text(json.dumps(selection_files,indent=2));(OUT/'selected_export_metrics.json').write_text(json.dumps(metrics));print('SUMMARY',OUT/'summary.json',flush=True);print(json.dumps({k:res[k] for k in ('by_format','selector_summary','integrity','unique_candidate_failures','paired_export')},indent=2))
