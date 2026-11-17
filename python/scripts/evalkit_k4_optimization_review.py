"""Read-only review of K4 multistart candidates against original-R and saved controls."""
import argparse
import datetime
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from motiogen.evalkit.k.acceptance import compare


def metrics(score):
    result={}
    for pair in ('D-R','F-D','F-R'):
        m=score['comparisons'][pair]['full']
        result[pair]={'directions_deg':m['directions_deg'],'joint_angles_deg':m['joint_angles_deg'],
                      'positions_cm':m['positions_cm'],
                      'temporal':m['temporal']}
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,required=True);ap.add_argument('--candidate',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if a.output.exists():raise ValueError('Preserve review snapshots; choose a new filename')
    protocol=json.loads((a.candidate/'protocol.json').read_text())
    panel=json.loads((a.candidate/'panel.json').read_text())['records'];rows=[];pending=[]
    for row in panel:
        key,method=row['window'],row['method'];name=f'{key}_{method}.json'
        oldroot=a.run/('supplement_022_ablation_v1' if key=='022' and method=='kimodo' else 'panel_ablation_v1')/'scores'
        basefile=oldroot/'legacy_common/before'/name
        if not basefile.exists():basefile=a.run/'common_v1/scores'/name
        baseline=json.loads(basefile.read_text())
        for condition in protocol['conditions']:
            for root in ('before','after'):
                file=a.candidate/'scores'/condition/root/name
                if not file.exists():pending.append({'window':key,'method':method,'condition':condition,'root':root});continue
                score=json.loads(file.read_text());record={**row,'condition':condition,'root':root,'score_file':str(file),
                    'baseline_file':str(basefile),'common_gate':compare(baseline,score),'metrics':metrics(score)}
                counterpart=protocol['old_counterparts'][condition]
                if counterpart in protocol['conditions']:controlroot=a.candidate/'scores'
                elif protocol.get('prior_candidate_folder'):controlroot=a.run/protocol['prior_candidate_folder']/'scores'
                else:controlroot=oldroot
                control=controlroot/counterpart/root/name
                if control.exists():
                    previous=json.loads(control.read_text())
                    record.update(counterpart_file=str(control),counterpart_gate=compare(previous,score),
                        paired_counterpart_gate=compare(previous['paired_baseline_clock'],score['paired_baseline_clock']))
                else:record['counterpart_status']='not available; no paired improvement claim'
                receipt=json.loads((a.candidate/'fits'/condition/key/method/'initialization/receipt.json').read_text())
                selected=receipt['selected'];record['initialization']={'selected':selected,'candidate_losses':{
                    k:v.get('input_loss',{'status':v['status']}) for k,v in receipt['candidates'].items()}}
                rows.append(record)
    result={'time':datetime.datetime.now().astimezone().isoformat(),'expected_views':len(panel)*len(protocol['conditions'])*2,
            'completed_views':len(rows),'pending':pending,'rows':rows,'promotion':'none',
            'scope':'Full D and fixed paired clocks. R scored after fit; no reference in runtime selection. No-regression alone is not target acceptance.'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print('reviewed views',len(rows),'pending',len(pending),'->',a.output)
    for row in rows:
        if row['root']!='after':continue
        m=row['metrics']['F-R'];angle=m['joint_angles_deg'];d=m['directions_deg']
        print(row['window'],row['method'],row['condition'],'seed',row['initialization']['selected'],
              'knee',*[round(angle[s+'_knee']['p95'],3) for s in ('left','right')],
              'trunk/shoulder',round(d['trunk_axis']['p95'],3),round(d['shoulder_line']['p95'],3),row['common_gate']['status'])


if __name__=='__main__':main()
