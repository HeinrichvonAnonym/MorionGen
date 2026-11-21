"""Read existing TV test scores; no fitting, scoring, hash scan or selector change."""
import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
import json
from zoneinfo import ZoneInfo


def read(p):
    return json.loads(p.read_text())


def flat(a):
    out = {}
    for group in ('directions_deg', 'joint_angles_deg'):
        for k, v in a[group].items():
            out[f'{group}/{k}'] = v
    for region in a['positions_cm'].values():
        for k, v in region.items():
            for geometry in ('world', 'pelvis_relative', 'common_length_world', 'common_length_pelvis_relative'):
                out[f'positions_cm/{k}/{geometry}'] = v[geometry]
    for k, v in a['temporal'].items():
        for stat in ('reference_step_deg', 'output_step_deg', 'direction_rate_vector_error_deg_s'):
            if stat in v:
                out[f'{k}/{stat}'] = v[stat]
    return out


def brief(v):
    return {k: v.get(k) for k in ('median', 'p95', 'max', 'n', 'denominator', 'signed_bias') if k in v}


def activity(row):
    return next(x for x in ('Packaging', 'Board_Connection', 'Drilling') if x in row['recording_id'].split('/'))


def region(metric):
    if 'shoulder' in metric: return '肩/肩线'
    if 'trunk' in metric: return '躯干'
    if 'wrist' in metric: return '腕'
    if 'elbow' in metric: return '肘'
    if 'upper_arm' in metric or 'forearm' in metric: return '上臂/前臂'
    if 'thigh' in metric or 'shank' in metric or 'knee' in metric: return '腿/膝'
    if 'foot' in metric or 'ankle' in metric: return '踝/足'
    if 'neck' in metric or 'head' in metric: return '头颈'
    return '其他'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root = args.run
    progress = read(root / 'progress.json')
    rows = progress['rows']
    failures = [x for x in rows if x.get('passed') is False]
    assert len(rows) == 4256 and len(failures) == 88
    result = dict(created=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                  scope='Read-only test diagnosis. No new fitting, scoring, tuning, holdout use, or hash scan.',
                  source=str(root), completion=read(root/'completion.json'), records=[])
    counts = Counter(); body = Counter(); windows = {}; selector = Counter()
    for x in failures:
        task = root/'tasks'/f"{x['window']}_{x['method']}"
        fit = Path(x['fit'])
        selection = read(fit/'candidate_selection.json')
        fm = x['failed_metrics']
        spatial = any(z['status'] == 'target_failed' for z in fm)
        temporal = any('increase_limit' in z for z in fm)
        category = '姿态位置+时序' if spatial and temporal else '姿态位置' if spatial else '时序'
        counts[category] += 1
        layer = next((v+'/'+c for v,c in [('native','D_clock'),('native','R_clock'),('export','D_clock'),('export','R_clock')]
                      if any(z['view']==v and z['clock']==c for z in fm)), 'unknown')
        counts[layer] += 1
        for r in set(region(z['metric']) for z in fm): body[r] += 1
        scores = {}; degraded = {}
        for view in ('native','export'):
            s = read(task/'visited'/x['selected']/f'{view}_score.json')
            for clock, v in [('D_clock',s),('R_clock',s['original_clock'])]:
                scores[view+'/'+clock] = {
                    st: {pair: {k:brief(z) for k,z in flat(v['comparisons'][pair][st]).items()}
                         for pair in ('D-R','F-D','F-R')}
                    for st in ('full','continuous','source_jump_tolerance')}
                a = flat(v['comparisons']['D-R']['continuous'])
                degraded[clock] = {
                    'max_direction_p95_deg': max(z.get('p95') or 0 for k,z in a.items() if k.startswith('directions_deg/')),
                    'max_world_position_p95_cm': max(z.get('p95') or 0 for k,z in a.items() if k.startswith('positions_cm/') and k.endswith('/world'))}
        visited = []
        for f in sorted((task/'visited').glob('*/review.json')):
            rev = read(f)
            visited.append(dict(candidate=f.parent.name,passed=rev['passed'],failed_metrics=rev.get('failed_metrics',[]),review=str(f)))
        passing = [v['candidate'] for v in visited if v['passed']]
        assert len(passing) == x['visited_passed']
        exhausted = selection['budget'].get('full_pool_fallback',False)
        kind = '已访问通过候选漏选' if passing else '候选池耗尽且均未通过' if exhausted else '提前接受失败/未访问未知'
        selector[kind] += 1
        audit = read(task/'visited'/x['selected']/'roundtrip/audit.json')
        rec = dict(x, activity=activity(x), failure_category=category, first_failure_layer=layer,
                   regions=sorted(set(region(z['metric']) for z in fm)), selector_category=kind,
                   visited=visited,passing_visited=passing,budget=selection['budget'],scores=scores,
                   degradation_continuous=degraded,roundtrip_audit=audit,
                   provenance=dict(selection=str(fit/'candidate_selection.json'),task=str(task)))
        result['records'].append(rec)
        w=windows.setdefault(x['window'],dict(window=x['window'],source=x['recording_id'],activity=activity(x),
                source_window_s=x['source_window_s'],formats=[],regions=set(),metrics=set()))
        w['formats'].append(x['method']);w['regions'].update(rec['regions']);w['metrics'].update(z['metric'] for z in fm)
    for w in windows.values():
        w['regions']=sorted(w['regions']);w['metrics']=sorted(w['metrics']);w['formats']=sorted(w['formats'])
    result['windows']=list(windows.values())
    result['counts']=dict(counts);result['body_route_counts']=dict(body);result['selector_counts']=dict(selector)
    result['format_counts']={m:dict(failed=sum(x['method']==m for x in failures),total=sum(x['method']==m for x in rows)) for m in sorted(set(x['method'] for x in rows))}
    result['activity_counts']={a:dict(failed=sum(activity(x)==a for x in failures),total=sum(activity(x)==a for x in rows),
        affected_windows=len(set(x['window'] for x in failures if activity(x)==a)),total_windows=len(set(x['window'] for x in rows if activity(x)==a))) for a in ('Packaging','Board_Connection','Drilling')}
    result['roundtrip']=dict(passed=sum(r['roundtrip_audit']['passed'] for r in result['records']),total=len(failures))
    assert len(windows)==12 and result['roundtrip']['passed']==88
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps({k:result[k] for k in ('counts','selector_counts','body_route_counts','format_counts','activity_counts','roundtrip')},ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
