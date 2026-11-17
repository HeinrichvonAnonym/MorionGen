"""Per-window absolute K4 acceptance; complete denominators and pending rows retained."""
import argparse,json
from pathlib import Path
from motiogen.evalkit.k.strict_acceptance import assess
from motiogen.evalkit.k.config import ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('scores','baseline','panel','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--policy',type=Path,default=ROOT/'config/evalkit_k4_20261117/strict_v1/acceptance.json')
    a=p.parse_args()
    if a.output.exists():raise ValueError('Keep review snapshots immutable')
    panel=json.loads(a.panel.read_text());policy=json.loads(a.policy.read_text());rows=[]
    for item in panel['records']:
        w,m=item['window'],item['method'];f=a.scores/f'{w}_{m}.json';baseline=a.baseline/f'{w}_{m}.json'
        row={'window':w,'method':m,'status':'pending','score':str(f)}
        if f.exists():
            s=json.loads(f.read_text());b=json.loads(baseline.read_text()) if baseline.exists() else None
            result=assess(s,b,policy);row.update(status=result['status'],acceptance=result)
            row['comparison_metrics']={pair:s['comparisons'][pair] for pair in ['D-R','F-D','F-R']}
            row['head_neck_rotation_comparisons']=s.get('head_neck_rotation_comparisons')
        rows.append(row)
    result={'expected':len(rows),'full_regression_expected':policy['expected_full_paths'],
            'counts':{s:sum(r['status']==s for r in rows) for s in ['pending','failed','passed']},
            'rows':rows,'policy':policy,'promotion':'none; review alone never changes runtime defaults',
            'scope':'Per-window original-R targets; full-window errors/coverage retained, frozen continuous stratum target acceptance'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(result['counts'],'of',result['expected'])
    for row in rows:
        if row['status']=='pending':continue
        failures=[r for r in row['acceptance']['required'] if r['status'] not in ['passed','unobservable']]
        print(row['window'],row['method'],row['status'],[(r['metric'],round(r['p95'],2) if r['p95'] is not None else None,r['status']) for r in failures])


if __name__=='__main__':main()
