"""Read-only distributions of window statistics; no frame pooling or R selection."""
import hashlib
import json
import itertools
from pathlib import Path

POSITION_GROUPS={'neck':'头颈节点','head':'头颈节点','left_shoulder':'肩','right_shoulder':'肩',
 'left_wrist':'腕','right_wrist':'腕','left_knee':'膝','right_knee':'膝',
 'left_ankle':'踝','right_ankle':'踝','left_foot':'足端','right_foot':'足端'}
ANGLE_GROUPS={'head_relative_thorax':'头胸相对旋转','head_neck_axis':'头颈方向','trunk_axis':'躯干方向',
 'shoulder_line':'肩线方向','pelvis_line':'骨盆线方向'}
for side in ('left','right'):
 for key,label in [('upper_arm','上臂方向'),('forearm','前臂方向'),('thigh','大腿方向'),('shank','小腿方向'),('foot_axis','足方向'),('elbow','肘几何角'),('knee','膝几何角')]:ANGLE_GROUPS[side+'_'+key]=label


def complementary_methods(snapshot):
    """Post-hoc crossed verdicts on shared reviewed windows; no best-of-R output."""
    p=json.loads(Path(snapshot).read_text());result={}
    for method in sorted({r['method'] for r in p['rows']}):
        verdicts={}
        for row in p['rows']:
            if row['method']!=method or not row['fully_evaluated']:continue
            for name,c in row['candidates'].items():
                if c['status']=='evaluated' and not c.get('equivalent_to'):
                    verdicts.setdefault(name,{})[row['window']]=bool(c['passed'])
        pairs=[];members=set()
        for a,b in itertools.combinations(verdicts,2):
            common=verdicts[a].keys()&verdicts[b].keys()
            aw=sorted(w for w in common if verdicts[a][w] and not verdicts[b][w])
            bw=sorted(w for w in common if verdicts[b][w] and not verdicts[a][w])
            if aw and bw:
                pairs.append(dict(a=a,b=b,common_windows=sorted(common),a_only=aw,b_only=bw));members.update([a,b])
        result[method]=dict(members=sorted(members),pairs=pairs,excluded=sorted(set(verdicts)-members))
    return result


def collect(snapshot, include_failed=False, reuse=None):
    raw=Path(snapshot).read_bytes();p=json.loads(raw);points=[];sources=[];routes=set();groups={};candidates=0
    passed_routes=set();passed_candidates=0;candidate_records=[]
    cached={}
    if reuse is not None:
        if reuse['snapshot_sha256']!=hashlib.sha256(raw).hexdigest():raise ValueError('Reuse snapshot mismatch')
        for point in reuse['points']:
            cached.setdefault((point['window'],point['method'],point['candidate']),[]).append(point)
        hashes={x['path']:x['sha256'] for x in reuse['source_scores']}
    def add(kind,metric,stats,row,name,source):
        if stats.get('median') is None or stats.get('p95') is None:return
        points.append(dict(kind=kind,group=(POSITION_GROUPS if kind=='position' else ANGLE_GROUPS)[metric],
            metric=metric,window=row['window'],method=row['method'],candidate=name,
            median=stats['median'],p95=stats['p95'],n=stats['n'],denominator=stats['denominator'],source=source,
            candidate_passed=bool(row['candidates'][name]['passed'])))
    for row in p['rows']:
        if not row['fully_evaluated']:continue
        for name,c in row['candidates'].items():
            if c.get('status')!='evaluated' or c.get('equivalent_to'):continue
            if not include_failed and not c.get('passed'):continue
            path=next(k for k in c['score_sha256'] if '/export/' in k)
            candidates+=1;routes.add((row['window'],row['method']))
            if c['passed']:
                passed_candidates+=1;passed_routes.add((row['window'],row['method']))
            candidate_records.append(dict(window=row['window'],method=row['method'],candidate=name,passed=c['passed']))
            key=(row['window'],row['method'],name)
            if key in cached and hashes.get(path)==c['score_sha256'][path]:
                points.extend(dict(x,candidate_passed=bool(c['passed'])) for x in cached[key])
                sources.append(dict(path=path,sha256=hashes[path]));continue
            data=Path(path).read_bytes();h=hashlib.sha256(data).hexdigest()
            if h!=c['score_sha256'][path]:raise ValueError('Score hash changed: '+path)
            block=json.loads(data)['original_clock'];scores=block['comparisons']['F-R']['full']
            sources.append(dict(path=path,sha256=h))
            for region,items in scores['positions_cm'].items():
                for metric,values in items.items():
                    if metric in POSITION_GROUPS:add('position',metric,values['world'],row,name,path)
            for family in ('directions_deg','joint_angles_deg'):
                for metric,values in scores[family].items():
                    if metric in ANGLE_GROUPS:add('angle',metric,values,row,name,path)
            head=block.get('head_neck_rotation_comparisons',{}).get('F-R',{}).get('full',{})
            add('angle','head_relative_thorax',head,row,name,path)
    return dict(snapshot=str(Path(snapshot).resolve()),snapshot_sha256=hashlib.sha256(raw).hexdigest(),
        snapshot_time=p['updated_unix_s'],complete_routes=p['fully_evaluated_routes'],
        routes_with_pass=len(passed_routes),passed_candidates_without_aliases=passed_candidates,
        evaluated_candidates_without_aliases=candidates,routes_with_scores=len(routes),candidate_records=candidate_records,
        source_scores=sources,points=points,
        scope=('All evaluated candidates' if include_failed else 'Passing candidates')+' from fully reviewed routes; export F-R at original R clock; full window; raw world positions. Each point is one window/candidate/metric statistic; exact aliases excluded. Repeated candidates/formats are paired, not independent participants.')
