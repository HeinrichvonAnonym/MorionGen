#!/usr/bin/env python3
"""Describe full development-source coverage without selecting or scoring fit policies."""
import argparse,collections,hashlib,json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from motiogen.evalkit.k.bones import NAMES


def distribution(values):
    values=sorted(values)
    return {'count':len(values),'minimum':min(values),'median':statistics.median(values),'maximum':max(values)} if values else {'count':0}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    path=a.out/'light.jsonl';rows=[json.loads(line) for line in path.read_text().splitlines()]
    assert len({r['id'] for r in rows})==len(rows)
    data={}
    for dataset in sorted({r['dataset'] for r in rows}):
        pool=[r for r in rows if r['dataset']==dataset];good=[r for r in pool if r['status']=='computed']
        entry={'registered_rows':len(pool),'computed_rows':len(good),'participants':len({r['group_id'] for r in pool}),
            'original_recordings':len({r['recording_id'] for r in pool}),
            'failures':[{'id':r['id'],'reason':r.get('reason')} for r in pool if r['status']!='computed']}
        if dataset=='collaborate':
            entry['conversion_probe']={}
            for method in sorted(good[0]['conversions']):
                entry['conversion_probe'][method]={'status_counts':dict(collections.Counter(r['conversions'][method]['status'] for r in good)),
                    'rotation_valid_share':distribution([r['conversions'][method]['valid_rotation_share'] for r in good if r['conversions'][method]['status']=='computed'])}
            entry['conversion_scope']='at most 240 source poses; conversion probe uses three poses on a synthetic clock, verifying topology and proper rotations'
            entry['edge_length_m']={name:distribution([r['segment_lengths_m'][j] for r in good if r['segment_lengths_m'][j] is not None]) for j,name in enumerate(NAMES) if j}
            entry['sampled_source_frames']=sum(r['sampled_frames'] for r in good)
        else:
            entry['direct_marker_frames']={region:{key:sum(r['regions'][region][key] for r in good) for key in ['valid_frames','total_frames']} for region in ['head','thorax']}
            entry['coverage_scope']='direct geometric marker frames before redundant-marker recovery; each region counted separately; historical recovered-reference scores reported separately'
        data[dataset]=entry
    result={'source':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'datasets':data}
    dest=ROOT/'config/evalkit_k3_20261116/light_diagnostics.json';dest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:{n:v[n] for n in ['registered_rows','computed_rows','participants','original_recordings']} for k,v in data.items()},indent=2))


if __name__=='__main__':main()
