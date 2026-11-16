#!/usr/bin/env python3
"""Verify the unchanged head/thorax evidence path before reusing historical development scores."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from motiogen.evalkit.k import load,evaluate
from motiogen.evalkit.k.config import digest


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    files=['reference.py','geometry.py','export.py','validation.py'];hashes={}
    for name in files:
        path=Path('utils/motiogen/evalkit/k')/name
        hashes[name]={'k2':digest(a.out/'k2_checkout'/path),'k3':digest(ROOT/path)}
    rows=json.loads((a.out/'exports.json').read_text())[:8];checks=[]
    for row in rows:
        result=evaluate(load(row['source'],row['method'],config='config/evalkit_k_v3.json'),config='config/evalkit_k_v3.json')
        with np.load(a.out/'parity'/row['method']/'native_k.npz') as old:
            regions={}
            for region,item in result['metrics'].items():
                np.testing.assert_allclose(item['rotations'],old[region+'_rotations'],atol=1e-12,equal_nan=True)
                np.testing.assert_array_equal(item['valid'],old[region+'_valid'])
                regions[region]={'identical_rotations_and_validity':True,'valid_frames':int(item['valid'].sum())}
        checks.append({'method':row['method'],'regions':regions})
    if any(v['k2']!=v['k3'] for v in hashes.values()):raise ValueError('Reference path changed; historical reuse requires revalidation')
    source=ROOT/'config/evalkit_k_validation_20261116/development_summary.json'
    result={'scope':'historical DEVELOPMENT head/thorax reference and gap scores reused; not rerun as new full-body evidence',
            'reference_code_identical':hashes,'native_path_regression':checks,'development_summary':str(source),'summary_sha256':digest(source),
            'k3_change_scope':'full-body fitting, normalization, receipts and explicit configuration; head reference construction retained'}
    (ROOT/'config/evalkit_k3_20261116/head_evidence_reuse.json').write_text(json.dumps(result,indent=2)+'\n');print('head evidence checks',len(checks))

if __name__=='__main__':main()
