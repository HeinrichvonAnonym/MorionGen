"""Independent original-R review of a fixed D-only candidate bank experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from motiogen.evalkit.k.strict_acceptance import assess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Immutable review already exists')
    files={}
    def read(path):
        files[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    protocol=read(args.run/'protocol.json')
    panel=read(args.run/'panel.json')['records']
    review=read(args.run/'selected_replay/review.json')
    if review['completed']!=len(panel):raise ValueError('Selected replay is incomplete')
    expected={(r['window'],r['method']) for r in panel}
    if {(r['window'],r['method']) for r in review['rows']}!=expected:
        raise ValueError('Replay does not match registered panel')
    policy=read(ROOT/'config/evalkit_k4_20261117/strict_v1/acceptance.json')
    rows=[]
    for item in review['rows']:
        w,m=item['window'],item['method']
        folder=args.run/'fits/pool'/w/m
        selection=read(folder/'candidate_selection.json')
        receipt=read(folder/'receipt.json')
        audit=read(args.run/'export_audits'/w/m/'audit.json')
        candidates={}
        selected_score=None
        for name,quality in selection['qualities'].items():
            score_file=(args.run/'scores/pool/after'/f'{w}_{m}.json' if name==selection['selected']
                        else args.run/'unselected_scores'/name/f'{w}_{m}.json')
            score=read(score_file)
            candidates[name]={'runtime_input_fidelity':quality,
                              'export_original_R':assess(score,None,policy),
                              'original_clock_original_R':assess(score['original_clock'],None,policy),
                              'score_file':str(score_file)}
            if name==selection['selected']:selected_score=score
        if selected_score is None:raise ValueError('Selected candidate lacks score')
        # Same input and sampling must give exactly the same D-R, irrespective of F.
        for candidate in candidates.values():
            score=read(Path(candidate['score_file']))
            if score['comparisons']['D-R']!=selected_score['comparisons']['D-R']:
                raise ValueError('D-R changed between candidates')
        conditions={name:{key:value for key,value in data.items() if key in (
            'acceptance_relative_D','acceptance_original_clock_relative_D',
            'full_comparison_summaries','continuous_comparison_summaries','attribution')}
            for name,data in item['conditions'].items()}
        required=('native','whole_native_grid')
        passed=all(conditions[c][key]['status']=='passed' for c in required
                   for key in ('acceptance_relative_D','acceptance_original_clock_relative_D')) and audit['passed']
        rows.append(dict(window=w,method=m,selected=selection['selected'],candidates=candidates,
                         native_and_runtime_original_R_passed=passed,conditions=conditions,
                         export_audit=audit,failures=selection['failures'],
                         runtime_config=receipt['key']['config'],degradation_invariant=True))
    result={'version':'k4-candidate-review-1','registered':len(panel),
            'passed':sum(r['native_and_runtime_original_R_passed'] for r in rows),'rows':rows,
            'protocol':protocol,'source_sha256':files,'promotion':'none; local development evidence only',
            'new_holdout_uses':0,'scope':'D-only whole-clip selection; separate original-R native/export/time acceptance. Full and continuous strata retained; no subtraction of error norms.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('passed',result['passed'],'/',result['registered'])


if __name__=='__main__':main()
