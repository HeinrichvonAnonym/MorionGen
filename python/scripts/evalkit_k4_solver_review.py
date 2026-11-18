"""Compact paired numerical-solver evidence; never selects a runtime method."""
import argparse,hashlib,json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise ValueError('Preserve each review snapshot; choose a new output')
    source=a.run/'review.json';review=json.loads(source.read_text());rows=[];sources={str(source):hashlib.sha256(source.read_bytes()).hexdigest()}
    for r in review['records']:
        row={k:r[k] for k in ('window','method','condition','status')}
        if r['status']=='completed':
            fit=a.run/'fits'/r['condition']/r['window']/r['method']
            file=fit/'trajectory_refinement.json';q=json.loads(file.read_text());sources[str(file)]=hashlib.sha256(file.read_bytes()).hexdigest()
            row.update(passed=r['passed'],failures=r['failures'],optimizer={k:q[k] for k in ('initial_squared_residual','candidate_squared_residual','optimizer_success','nfev','njev','optimality')},
                linear_iterations=sum(x['linear_iterations'] for x in q['iterations']),linear_iteration_limit_hits=sum(x['linear_status']==0 for x in q['iterations']),
                backtracking_trials=sum(len(x['trials']) for x in q['iterations']),clock_metrics={})
            for view,clocks in r['acceptance'].items():
                row['clock_metrics'][view]={}
                for clock,result in clocks.items():
                    # Each number is the worst per-metric P95, not a pooled percentile.
                    supported=[x for x in result['required'] if x.get('p95') is not None]
                    pos=[x for x in supported if x['metric'].startswith('positions_cm/')]
                    rot=[x for x in supported if not x['metric'].startswith('positions_cm/')]
                    worst=lambda group:max(group,key=lambda x:x['p95']) if group else None
                    row['clock_metrics'][view][clock]={'worst_position_metric_cm':worst(pos),'worst_angular_metric_deg':worst(rot),
                        'temporal':result['temporal'],'full_window_coverage':result['full_window_coverage']}
            if r['condition']=='control':
                for name in ['control_reproduction.json']:
                    file=fit/name;row[name]=json.loads(file.read_text());sources[str(file)]=hashlib.sha256(file.read_bytes()).hexdigest()
        rows.append(row)
    result=dict(expected=review['expected'],completed=review['completed'],passed=review['passed'],records=rows,source_sha256=sources,
        interpretation='Compare same-source control and candidate separately for every view/clock. Objective and gradient diagnose numerical solve; independent original-R accuracy remains necessary. Linear iteration counts are not elapsed time; equal20 objective evaluations do not imply equal computation. No subtraction of P95s as error contributions.',promotion='none')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    for r in rows:
        if r['status']!='completed':continue
        m=r['clock_metrics']['export']['R_clock'];pos=m['worst_position_metric_cm'];rot=m['worst_angular_metric_deg']
        print(r['window'],r['condition'],'passed',r['passed'],'objective',round(r['optimizer']['candidate_squared_residual'],4),'worstR',round(rot['p95'],4),rot['metric'],round(pos['p95'],4),pos['metric'],'linear iterations',r['linear_iterations'])

if __name__=='__main__':main()
