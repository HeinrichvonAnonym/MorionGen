"""Freeze conservative fixed-source replays; never select by measured outcome."""
import copy
import json
import random
import sys
from collections import defaultdict,Counter
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'utils'))
from motiogen.source_cohorts import SOURCES,archive_sources
from motiogen.runtime import checksum,json_write
base=ROOT/'develop_logs/test_runs'
out=base/'20261025_source_replays'
if (out/'plan.json').exists():raise ValueError('Frozen replay plan already exists')
analysis=json.loads((base/'20261024_native_matrix_analysis/analysis.json').read_text())
current={f:checksum(ROOT/f) for paths in SOURCES.values() for f in paths}
json_write(out/'source_archive.json',{'files':archive_sources(ROOT,out/'frozen_sources',current)})
selected=set();mismatches=[];inventories={}
for r in analysis['rows']:
    if r['run'] not in inventories:inventories[r['run']]=json.loads((ROOT/r['run']/'implementation.json').read_text())
    changed=[f for f in SOURCES[r['method']] if inventories[r['run']].get(f)!=current[f]]
    if changed:
        selected.add((r['method'],r['task'],r['seed']))
        mismatches.append({'run':r['run'],'method':r['method'],'task':r['task'],'seed':r['seed'],'clock':r['clock'],
                           'files':{f:{'recorded':inventories[r['run']].get(f),'frozen':current[f]} for f in changed}})
groups=defaultdict(list);seen=set()
for r in analysis['rows']:
    key=(r['method'],r['task'],r['seed'])
    if key not in selected or r['run'] in seen:continue
    seen.add(r['run']);path=ROOT/r['run']
    manifest=json.loads((path/'manifest.json').read_text());cfg=json.loads((path/'config.json').read_text())
    # Physical GPU is assigned by the scheduler, never restored from an old run.
    for field in ['gpu_index','_source','_data_root']:cfg.pop(field,None)
    job_id='source_replay_'+r['method']+'_task'+str(r['task'])+'_seed'+str(r['seed'])+'_'+r['clock']+'_'+path.name[-8:]
    native=r['method'] in ['dart','primal','closd']
    required=['{run_dir}/'+s for s in (['factorial/timing.json','factorial/initial_history.npz'] if r['method']=='primal' else ['motion.npz','metrics.json','intervention.json'])]
    job={'id':job_id,'method':r['method'],'seed':r['seed'],'task_id':r['task'],'scene':manifest['scenario'],
         'config':cfg,'method_spec':cfg['baselines'][r['method']],
         'command':[arg.replace(str(path),'{run_dir}') for arg in manifest['command']],
         'required_outputs':required,'require_protocol':not native,'timeout_seconds':1800,
         'estimated_seconds':max(60,min(float(manifest.get('elapsed_seconds') or 120),300)),
         'prior_run':r['run'],'replay_group':'/'.join(map(str,key))}
    groups[key].append(job)
keys=sorted(groups);random.Random(20261025).shuffle(keys)
jobs=[];replay_groups=[]
for key in keys:
    block=groups[key];random.Random('/'.join(map(str,key))).shuffle(block);jobs.extend(block)
    replay_groups.append({'key':'/'.join(map(str,key)),'method':key[0],'task':key[1],'seed':key[2],
                          'jobs':[j['id'] for j in block],'prior_runs':[j['prior_run'] for j in block]})
budget=json.loads((ROOT/'config/survey_revision.json').read_text())
json_write(out/'original_analysis.json',analysis)
json_write(out/'audit_before.json',{'analysis_cutoff':analysis['updated_at'],'selected_groups':len(groups),
    'mismatches':mismatches,'frozen_sources':current,'source_contract':SOURCES})
plan={'registered_at':datetime.now(timezone.utc).isoformat(),'devices':[0,1],
      'deadline':budget['experiment_deadline_at'],'report_reserve_seconds':14400,
      'jobs':jobs,'source_replay_groups':replay_groups,'frozen_sources':current,'identity_files':[],
      'scope':'Conservative same-source replays. Select solely by inference-module inventory mismatch, including instrumentation changes; do not assert those changes affected motion. Keep both clocks and all intervention conditions of each selected task/seed. Original evidence remains. Atomic whole-group replacement only after every replay succeeds; this does not increase independent sample counts.'}
json_write(out/'plan.json',plan)
print(json.dumps({'jobs':len(jobs),'groups':len(groups),'by_method':dict(Counter(j['method'] for j in jobs)),
                  'estimated_gpu_lease_hours':sum(j['estimated_seconds'] for j in jobs)/3600}))
