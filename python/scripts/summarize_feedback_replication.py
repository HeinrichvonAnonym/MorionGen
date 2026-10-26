"""Recompute v4 outcomes; cluster scenarios, never treat clocks as replicates."""
import argparse
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'develop_logs/test_runs/20261024_feedback_replication_v4'


def sustained(error, radius, seconds, fps=30):
    width = max(1, round(seconds * fps))
    hits = error <= radius
    if len(hits) < width:
        return False, None
    starts = np.flatnonzero(np.convolve(hits.astype(int), np.ones(width, dtype=int), 'valid') == width)
    return bool(len(starts)), float(starts[0] / fps) if len(starts) else None


def summarize(require_complete=True):
    rows, arrays, sensitivity, clips = [], {}, [], []
    for method in ['dart', 'primal']:
        manifest = json.loads((RUN / method / 'manifest.json').read_text())
        for case in manifest['scenarios']:
            for seed in manifest['seeds']:
                for clock in manifest['modes']:
                    for flag in manifest['feedback']:
                        key = f"{case['id']}_seed{seed}_{clock}_feedback{flag}"
                        path = RUN / method / key
                        if not (path / 'trace.json').exists():
                            if require_complete:
                                raise AssertionError(f'Missing completed case: {path}')
                            continue
                        trace = json.loads((path / 'trace.json').read_text())
                        xyz = np.load(path / 'motion.npz')['joints']
                        assert xyz.shape == (450, 22, 3) and np.isfinite(xyz).all(), path
                        arrays[(method, case['id'], seed, clock, flag)] = xyz
                        target = np.array(trace['goal_events'][1]['position'])
                        errors = np.linalg.norm(xyz[150:, 0, :2] - target[:2], axis=1)
                        metrics = json.loads((path / 'metrics.json').read_text())['metrics']
                        requests = trace['requests']
                        new_requests = [r for r in requests if r['observed_frame'] >= 150 and np.allclose(r['goal'], target)]
                        ok, recovery = sustained(errors, .3, .5)
                        assert bool(trace['task'][1]['success']) == ok
                        assert abs(trace['task'][1]['final_error_m'] - float(errors[-1])) < 1e-6
                        row = dict(method=method, scenario=case['id'], initial=case['initial'], direction=case['direction'],
                                   seed=seed, clock=clock, feedback=flag, mean_error_m=float(errors.mean()),
                                   final_error_m=float(errors[-1]), success=int(ok), recovery_seconds=recovery,
                                   end_hold=int((errors[-15:] <= .3).all()), occupancy=float((errors <= .3).mean()),
                                   hold_fraction=trace['hold_fraction'], wall_seconds=trace['wall_seconds'],
                                   compute_p95_ms=float(np.quantile([r['compute_seconds'] for r in requests], .95)*1000),
                                   response_seconds=(max(new_requests[0]['request_frame'],new_requests[0]['apply_frame'])-150)/30 if new_requests else None,
                                   gpu_allocated_gib=trace['gpu_peak_bytes']/2**30,
                                   jerk_rms=metrics['jerk_rms']['value'], trunk_inclination=metrics['trunk_inclination']['value'],
                                   source=str(path.relative_to(ROOT)))
                        rows.append(row)
                        for window in [5, 10]:
                            for radius in [.2, .3, .5]:
                                for hold in [0, .5, 1]:
                                    hit, onset = sustained(errors[:window*30], radius, hold)
                                    sensitivity.append(dict(method=method, scenario=case['id'], seed=seed, clock=clock,
                                                            feedback=flag, window=window, radius=radius, hold=hold,
                                                            success=int(hit), recovery_seconds=onset))
                        if seed == 10 and clock == 'realtime':
                            origin = np.array(trace['goal_events'][0]['position']) - [1,0,0]
                            clips.append(dict(method=method, scenario=case['id'], feedback=flag, fps=10,
                                              joints=(xyz[::3]-origin).round(4).tolist(),
                                              goals=[(np.array(g['position'])-origin).tolist() for g in trace['goal_events']]))
    checks, groups, effects, scenario_effects = [], [], [], []
    metrics = ['mean_error_m','final_error_m','success','end_hold','occupancy','jerk_rms','trunk_inclination','hold_fraction','compute_p95_ms']
    rng = np.random.default_rng(20261024)
    for method in ['dart', 'primal']:
        for clock in ['paused', 'realtime']:
            selected = [r for r in rows if r['method']==method and r['clock']==clock]
            for flag in [0,1]:
                subset = [r for r in selected if r['feedback']==flag]
                if subset:
                    groups.append(dict(method=method,clock=clock,feedback=flag,n=len(subset),
                                       **{m:float(np.mean([r[m] for r in subset])) for m in metrics}))
            clusters = []
            paired_by_scenario = {}
            for case in sorted({r['scenario'] for r in selected}):
                pairs=[]
                for seed in [10,11,12]:
                    pair = [next((r for r in selected if r['scenario']==case and r['seed']==seed and r['feedback']==flag),None) for flag in [0,1]]
                    if None in pair:
                        continue
                    a,b = [arrays[(method,case,seed,clock,f)] for f in [0,1]]
                    prefix=float(np.max(np.abs(a[:150]-b[:150])))
                    checks.append(dict(method=method,clock=clock,scenario=case,seed=seed,pre_event_max_difference_m=prefix))
                    if require_complete:
                        assert prefix<1e-5, ('pre-event divergence',checks[-1])
                    pairs.append({m:pair[1][m]-pair[0][m] for m in metrics})
                if len(pairs)==3:
                    paired_by_scenario[case] = pairs
                    cluster=dict(method=method,clock=clock,scenario=case,n_pairs=3,
                                 **{m:float(np.mean([r[m] for r in pairs])) for m in metrics})
                    clusters.append(cluster)
                    scenario_effects.append(cluster)
            if len(clusters)==12:
                indexes=rng.integers(0,12,(20000,12))
                # Crossed factors share starting states and RNG seeds. Resample
                # entire factor levels, not 36 artificially independent episodes.
                initial_indices=rng.integers(0,3,(20000,3))
                direction_indices=rng.integers(0,4,(20000,4))
                seed_indices=rng.integers(0,3,(20000,3))
                for metric in metrics:
                    delta=np.array([r[metric] for r in clusters])
                    ci=np.quantile(delta[indexes].mean(1),[.025,.975])
                    tensor=np.array([[p[metric] for p in paired_by_scenario[case]]
                                     for case in sorted(paired_by_scenario)]).reshape(3,4,3)
                    crossed=tensor[initial_indices[:,:,None,None],direction_indices[:,None,:,None],seed_indices[:,None,None,:]].mean((1,2,3))
                    leave_out=[]
                    for prefix in ['standing','walk_warmstart','turn_warmstart']:
                        leave_out.append(float(np.mean([r[metric] for r in clusters if not r['scenario'].startswith(prefix)])))
                    for direction in ['continue','left','right','reverse']:
                        leave_out.append(float(np.mean([r[metric] for r in clusters if not r['scenario'].endswith(direction)])))
                    effects.append(dict(method=method,clock=clock,metric=metric,delta=float(delta.mean()),
                                        ci95=ci.tolist(),n_scenarios=12,n_pairs=36,
                                        crossed_factor_ci95=np.quantile(crossed,[.025,.975]).tolist(),
                                        negative_scenarios=int((delta<0).sum()),
                                        leave_factor_out_range=[min(leave_out),max(leave_out)]))
    clock_checks=[]
    for key,xyz in arrays.items():
        method,case,seed,clock,flag=key
        other=(method,case,seed,'realtime',flag)
        if clock=='paused' and other in arrays:
            clock_checks.append(dict(method=method,scenario=case,seed=seed,feedback=flag,
                                     max_difference_m=float(np.abs(xyz-arrays[other]).max())))
    completions={m:json.loads((RUN/m/'completion.json').read_text()) for m in ['dart','primal'] if (RUN/m/'completion.json').exists()}
    result=dict(n_runs=len(rows),expected_runs=288,rows=rows,groups=groups,effects=effects,
                scenario_effects=scenario_effects,prefix_checks=checks,clock_checks=clock_checks,
                sensitivity=sensitivity,clips=clips,completion=completions,
                inference='20,000 percentile bootstrap resamples of 12 scenario-mean paired effects; 3 seeds per scenario. Scenario grid is fixed and shares model-generated initial states: intervals describe this design, not a human-motion population. Clocks analyzed separately. Sensitivity endpoints exploratory, no multiplicity-adjusted claim.')
    if require_complete:
        assert len(rows)==288 and len(checks)==144 and len(completions)==2
    destination=RUN/('summary.json' if require_complete else 'partial_summary.json')
    destination.write_text(json.dumps(result,ensure_ascii=False,separators=(',',':')))
    print(json.dumps({k:result[k] for k in ['n_runs','groups','effects','completion']},indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--partial',action='store_true')
    summarize(not parser.parse_args().partial)
