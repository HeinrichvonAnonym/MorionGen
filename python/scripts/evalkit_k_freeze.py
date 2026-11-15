#!/usr/bin/env python3
"""Freeze completed development evidence before any final K evaluation."""
import argparse,datetime,hashlib,json,platform,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from motiogen.evalkit.k.config import implementation_key,digest
from evalkit_k_summarize import summarize,summarize_gaps
from check_gold_k_registration import check


def complete(folder,expected):
    receipt=json.loads((folder/'receipt.json').read_text())
    rows=[json.loads(s) for s in (folder/'records.jsonl').read_text().splitlines()]
    if len(rows)!=expected or len({r['id'] for r in rows})!=expected:raise ValueError(f'Incomplete evidence: {folder}')
    if any(r['status']!='computed' for r in rows):raise ValueError(f'Unresolved execution failure: {folder}')
    return receipt,rows


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('development','gaps','regression','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args()
    if a.out.exists():raise ValueError('Freeze is immutable; select a new version directory')
    reg=ROOT/'config/gold_k_20261115';registration=check(reg)
    key={**implementation_key(),'validation_script':digest(ROOT/'python/scripts/evalkit_k_validate.py'),'split':digest(reg/'split.json'),'manifest':digest(reg/'trials.jsonl')}
    gap_key={**implementation_key(),'gap_script':digest(ROOT/'python/scripts/evalkit_k_gap_validate.py')}
    dr,rows=complete(a.development,1670);gr,gaps=complete(a.gaps,1670)
    if dr['implementation']!=key or gr['implementation']!=gap_key:raise ValueError('Development does not match current implementation')
    regression=json.loads((a.regression/'summary.json').read_text())
    if len(regression)!=8 or any(r['status']!='computed' for r in regression):raise ValueError('Eight real-output regressions required')
    summary=summarize(rows,'development');summary['registered_gap_experiment']=summarize_gaps(gaps)
    summary['source']=str(a.development.resolve());summary['implementation']=key
    roots=json.loads((reg/'evidence.json').read_text())['local_roots']
    manifest=list(map(json.loads,(reg/'trials.jsonl').read_text().splitlines()))
    zips={};verified=0
    for row in manifest:
        if row['status']!='included' or row['dataset']=='han':continue
        src=next(x for x in row['sources'] if x['archive']=='qualisys_csv.zip') if row['dataset']=='andy' else row['source']
        path=Path(roots[row['dataset']])/src['archive']
        if path not in zips:zips[path]=zipfile.ZipFile(path)
        info=zips[path].getinfo(src['member'])
        if info.file_size!=src['bytes'] or f'{info.CRC:08x}'!=src['crc32']:raise ValueError(f'Source changed: {row["id"]}')
        verified+=1
    inventory={str(path):{'bytes':path.stat().st_size,'central_directory_sha256':hashlib.sha256('\n'.join(f'{i.filename}|{i.file_size}|{i.CRC}' for i in z.infolist()).encode()).hexdigest()} for path,z in zips.items()}
    for z in zips.values():z.close()
    han=Path(roots['han']).parents[2]/'outputs/evalkit_k_20261115/han_markers'
    marker_hashes={p.name:digest(p) for p in sorted(han.glob('*.npz'))}
    import opensim,numpy,scipy,biomech
    frozen={'version':'evalkit-k1-final-20261116','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'implementation':key,'gap_implementation':gap_key,'registration':registration,
            'development_source':str(a.development.resolve()),'gap_source':str(a.gaps.resolve()),'regression_source':str(a.regression.resolve()),
            'selected_reference_gates':{d:{r:v['selected_gate_m'] for r,v in regions.items()} for d,regions in summary['selected_reference_gates'].items()},
            'scoring':{'tiers_deg':[5,10,20],'precision_scope':'engineering consistency of declared reference frames; external anatomical accuracy separately required',
                'reference_window':'full record; optical calibration frames excluded from template comparison',
                'format_window':'first complete >=1 second interval, up to 10 seconds; PRIMAL first two frames reserved for history',
                'missing_windows':'retain in record denominator; no reference-based initialization',
                'recovery':'offline only; runtime max 0.1 s; 0.5/1/2 s stress tests separately labelled',
                'coordinate_precision_m':1e-4,'segment_axes':'anatomical right, superior, posterior; proper right-handed basis',
                'aggregation':'segments -> original recording -> person; no pooled-P95 claim'},
            'source_inventory':inventory,'manifest_zip_entries_verified':verified,'han_marker_sha256':marker_hashes,
            'integration':{str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'utils/motiogen/evalkit'/n for n in ('run.py','fit.py','ergo.py','neck_rotation.py','upper_muscle.py')]},
            'tools':{str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/'python/scripts').glob('evalkit_k*.py'))},
            'model':{'path':str(biomech.MODEL),'sha256':digest(biomech.MODEL)},
            'environment':{'python':platform.python_version(),'opensim':opensim.GetVersion(),'numpy':numpy.__version__,'scipy':scipy.__version__},
            'final_use':'Andy 75 trials / 5 people and Television 532 unique trials; one frozen evaluation, no retuning from results'}
    a.out.mkdir(parents=True)
    (a.out/'freeze.json').write_text(json.dumps(frozen,ensure_ascii=False,indent=2)+'\n')
    (a.out/'development_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (a.out/'regression_summary.json').write_text(json.dumps(regression,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'frozen','path':str(a.out),'records':len(rows),'folds':len(summary['folds']),'source_entries_verified':verified}))


if __name__=='__main__':main()
