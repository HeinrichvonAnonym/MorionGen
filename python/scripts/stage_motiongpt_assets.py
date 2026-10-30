"""Download only official MotionGPT inference checkpoint with pinned LFS hash."""
import json
import argparse
import sys
from pathlib import Path
import requests
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import load_config,data_root
from motiogen.runtime import checksum,json_write
from motiogen.transfer import proxy_file
p=argparse.ArgumentParser();p.add_argument('--text-encoder',action='store_true');args=p.parse_args()
repo='google/flan-t5-base' if args.text_encoder else 'OpenMotionLab/MotionGPT-base'
dest=data_root(load_config())/'outputs/assets'/('motiongpt-flan-t5-base' if args.text_encoder else 'motiongpt-official');dest.mkdir(parents=True,exist_ok=True)
manifest=dest/'source_manifest.json'
if manifest.exists():d=json.loads(manifest.read_text())
else:
    r=requests.get(f'https://huggingface.co/api/models/{repo}?blobs=true',timeout=30)
    r.raise_for_status();d=r.json();json_write(manifest,d)
rows=[]
for row in d['siblings']:
    name=row['rfilename'];target=dest/name
    if args.text_encoder and name not in ['config.json','generation_config.json','model.safetensors','spiece.model','special_tokens_map.json','tokenizer.json','tokenizer_config.json']:continue
    if not target.exists():proxy_file(f'https://huggingface.co/{repo}/resolve/{d["sha"]}/{name}',target,row['size'])
    sha=checksum(target);expected=row.get('lfs',{}).get('sha256')
    if expected and sha!=expected:raise ValueError('Checkpoint hash mismatch')
    rows.append({'path':name,'bytes':target.stat().st_size,'sha256':sha,'lfs_match':sha==expected if expected else None})
    json_write(dest/'source_receipt.json',{'repository':repo,'revision':d['sha'],'files':rows})
    print(json.dumps(rows[-1]),flush=True)
