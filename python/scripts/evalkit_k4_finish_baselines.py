"""Wait for owned baseline jobs, score historical FK, and fix the dev panel.

This continuation never starts optimization or promotes an experimental fit.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]


def alive(pid):
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False


def aggregate(folder):
    windows={r['window_key']:r for r in json.loads((ROOT/'config/k_degradation_20261117/windows.json').read_text())['records']}
    records={};count=0
    def add(method,pair,path,item,key):
        name='/'.join([method,pair,*path]);row=records.setdefault(name,[])
        row.append({'window':key,'participant':windows[key]['group_id'],**item})
    for file in sorted(folder.glob('???_*.json')):
        d=json.loads(file.read_text());key=d['window'];method=d['method'];count+=1
        for pair in ('D-R','F-D','F-R'):
            metrics=d['comparisons'][pair]['full']
            for group in ('directions_deg','joint_angles_deg'):
                for name,item in metrics[group].items():add(method,pair,[group,name],item,key)
            for region,nodes in metrics['positions_cm'].items():
                for name,values in nodes.items():
                    for geometry in ('world','pelvis_relative','common_length_world','common_length_pelvis_relative'):
                        add(method,pair,['positions_cm',region,name,geometry],values[geometry],key)
    out={}
    for name,rows in records.items():
        valid=[r for r in rows if r['p95'] is not None];by_person={}
        for row in valid:by_person.setdefault(row['participant'],[]).append(row['p95'])
        out[name]={'scored_windows':len(valid),'available_records':len(rows),'expected_windows_per_format':82,
                   'median_window_p95':statistics.median([r['p95'] for r in valid]) if valid else None,
                   'worst_window_p95':max((r['p95'] for r in valid),default=None),
                   'participant_median_window_p95':{p:statistics.median(v) for p,v in by_person.items()},
                   'windows_at_5_unit_target':sum(r['p95']<=5 for r in valid),
                   'scorable_samples':sum(r['n'] for r in rows),'expected_samples_in_available_records':sum(r['denominator'] for r in rows)}
    return {'available_paths':count,'expected_paths':656,'metrics':out,
            'aggregation':'Median and worst per-window P95, with per-participant median window P95. These are not pooled-frame P95.',
            'clock':'Full actual D clock; complete original-clock results remain in each score file'}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--historical-pid',type=int,required=True);ap.add_argument('--common-pid',type=int,required=True)
    args=ap.parse_args();common=args.run/'common_v1';historical=args.run/'historical_k2'
    code_files=[ROOT/'python/scripts/evalkit_k4_score.py',ROOT/'python/scripts/evalkit_k4_panel.py',
                ROOT/'utils/motiogen/evalkit/k/scoring.py']
    code_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in code_files}
    def check_code():
        if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h for p,h in code_hashes.items()):
            raise RuntimeError('Review implementation changed during wait; restart with an explicit version')
    last=None
    while True:
        hs=list(historical.glob('*/*/status.json'));cs=list(common.glob('*/*/status.json'))
        count=(len(hs),len(cs))
        if count!=last:print('historical/common completed or failed:',*count,flush=True);last=count
        common_done=(common/'completion.json').exists();historical_done=len(hs)==656
        if common_done and historical_done:break
        if not common_done and not alive(args.common_pid):raise RuntimeError('Common baseline exited before complete accounting')
        if not historical_done and not alive(args.historical_pid):raise RuntimeError('Historical baseline exited before complete accounting')
        time.sleep(10)
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    check_code()
    score_dir=args.run/'historical_scores'
    with (args.run/'historical_scoring.log').open('w') as log:
        subprocess.run([sys.executable,str(ROOT/'python/scripts/evalkit_k4_score.py'),'--fits',str(historical),
                        '--output',str(score_dir)],check=True,stdout=log,stderr=subprocess.STDOUT,env=env)
    panel=ROOT/'config/evalkit_k4_20261117/fixed_panel.json'
    subprocess.run([sys.executable,str(ROOT/'python/scripts/evalkit_k4_panel.py'),
                    '--baseline',str(common),'--output',str(panel)],check=True,env=env)
    summaries={name:aggregate(folder) for name,folder in [('historical',score_dir),('common',common/'scores')]}
    check_code()
    protected=json.loads((ROOT/'config/evalkit_k4_20261117/preservation.json').read_text())['files']
    changed=[name for name,h in protected.items() if not (ROOT/name).exists() or hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=h]
    result={'baselines':summaries,'panel':str(panel),'protected_files':len(protected),'protected_changed':changed,
            'new_holdout_uses':0,'optimization_status':'pending fixed-panel ablation; experimental probe rejected',
            'review_code_sha256':code_hashes}
    (args.run/'baseline_review.json').write_text(json.dumps(result,indent=2)+'\n')
    if changed:raise RuntimeError('Protected files changed: '+str(changed))
    print('Baseline scoring and fixed panel complete. No candidate promoted. Review:',args.run/'baseline_review.json',flush=True)


if __name__=='__main__':main()
