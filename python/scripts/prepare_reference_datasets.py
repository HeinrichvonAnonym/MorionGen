#!/usr/bin/env python
"""Prepare bounded, reproducible dataset copies without modifying source archives."""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'utils'))

from motiogen.config import load_config
from motiogen.datasets import (bvh_members,benchmark_metadata_root,extract_kit,extract_members,
    materialize_subset,parquet_rows,prepared_root,select_stratified,selection_sha,source_receipts,stratified_root,
    validate_kit)
from motiogen.runtime import json_write


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=['all','bones-seed','kit-ml'],default='all')
    parser.add_argument('--per-group',type=int,default=32)
    parser.add_argument('--seed',type=int,default=20261008)
    parser.add_argument('--skip-extract',action='store_true')
    args=parser.parse_args(argv); cfg=load_config(); receipts=source_receipts()
    result={'status':'success','source_receipts':receipts}
    if args.dataset in ('all','bones-seed'):
        source=benchmark_metadata_root(cfg); root=stratified_root(cfg); subset=root/'testsuite'
        rows=select_stratified(source,args.per_group,args.seed)
        materialize_subset(source,subset,rows)
        members=bvh_members(subset); prepared=prepared_root(cfg)/'bones-seed-stratified-20261008'
        manifest={'schema_version':1,'seed':args.seed,'per_group':args.per_group,
                  'group_count':len({x['group'] for x in rows}),'case_count':len(rows),
                  'selection_sha256':selection_sha(rows),'source_testsuite':str(source),
                  'source_archive':receipts['bones_seed']['archive'],'prepared_dataset':str(prepared/'soma_uniform'),
                  'source_archive_format':receipts['bones_seed']['archive_format'],
                  'source_metadata_rows':parquet_rows(receipts['bones_seed']['metadata']),
                  'bvh_count':len(members),'cases':rows}
        if manifest['source_metadata_rows'] not in (None,142220):
            raise ValueError(f"expected 142220 BONES-SEED metadata rows, found {manifest['source_metadata_rows']}")
        json_write(root/'selection_manifest.json',manifest)
        if not args.skip_extract:
            extract_members(receipts['bones_seed']['archive'],prepared,members,root/'required_bvh_members.txt')
        missing=[name for name in members if not (prepared/name).is_file()]
        manifest['extraction_status']='ready' if not missing else 'missing_resources'; manifest['missing_bvh']=missing[:100]
        json_write(root/'selection_manifest.json',manifest); result['bones_seed']=manifest
    if args.dataset in ('all','kit-ml'):
        target=prepared_root(cfg)/'KIT-ML'
        for name in ('Mean.npy','Std.npy','test.txt','train.txt','val.txt','train_val.txt','all.txt'):
            src=Path(receipts['kit_ml']['source_root'])/name
            if src.is_file():
                target.mkdir(parents=True,exist_ok=True); shutil.copy2(src,target/name)
        if not args.skip_extract:
            extract_kit(receipts['kit_ml']['archives'],target)
        check=validate_kit(target,receipts['kit_ml']['split'])
        json_write(target/'prepared_receipt.json',{'schema_version':1,'source':receipts['kit_ml'],**check})
        result['kit_ml']={'prepared':str(target),**check}
    print(json.dumps(result,indent=2,ensure_ascii=False))
    return 0 if all(x.get('status','ready') in ('ready','success') for x in result.values() if isinstance(x,dict)) else 1


if __name__=='__main__': raise SystemExit(main())
