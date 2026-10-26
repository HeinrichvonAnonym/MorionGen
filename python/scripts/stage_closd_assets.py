"""Stage only official inference assets, pin commit and verify LFS content hashes."""
import concurrent.futures
import json
from pathlib import Path,PurePosixPath
import shutil
import sys
import requests
import argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import ROOT,load_config,data_root
from motiogen.runtime import json_write,checksum
from motiogen.transfer import proxy_file

parser=argparse.ArgumentParser()
parser.add_argument('--bert',action='store_true',help='Stage the official DiP DistilBERT text encoder')
args=parser.parse_args()
repository='distilbert/distilbert-base-uncased' if args.bert else 'guytevet/CLoSD'
cfg=load_config(); destination=data_root(cfg)/'outputs/assets'/('closd-distilbert' if args.bert else 'closd-official')
destination.mkdir(parents=True,exist_ok=True)
manifest=destination/'source_manifest.json'
if manifest.exists():
    metadata=json.loads(manifest.read_text())
else:
    response=requests.get(f'https://huggingface.co/api/models/{repository}?blobs=true',timeout=30)
    if response.status_code in [401,403]: raise RuntimeError('Official CLoSD assets require authentication; notify user')
    response.raise_for_status(); remote=response.json()
    selected=[]
    for row in remote['siblings']:
        name=row['rfilename']; relative=PurePosixPath(name)
        if args.bert and name not in ['README.md','config.json','tokenizer_config.json','tokenizer.json','vocab.txt','model.safetensors']: continue
        if relative.is_absolute() or '..' in relative.parts: raise ValueError('Unsafe remote path')
        if name.startswith('data/amass/') or name=='data/humanml3d/t2m_train.npy': continue
        if name.startswith('evaluation/'): continue
        if relative.name.startswith('opt') and relative.suffix=='.pt': continue
        selected.append(row)
    metadata={'repository':repository,'revision':remote['sha'],'files':selected,
              'exclusions':'Training AMASS/t2m caches, optimizer states and published evaluation outputs; no retraining required'}
    json_write(manifest,metadata)
required=sum(row.get('size',0) for row in metadata['files'])
free=shutil.disk_usage(destination).free
if free-required<80*1024**3: raise RuntimeError('Inference assets would cross data-disk reserve')
print(json.dumps({'files':len(metadata['files']),'bytes':required,'free_bytes':free,'revision':metadata['revision']}),flush=True)
def stage(row):
    name=row['rfilename']; target=destination/name
    expected=row.get('lfs',{}).get('sha256')
    if not target.is_file() or target.stat().st_size!=row['size']:
        proxy_file(f'https://huggingface.co/{repository}/resolve/{metadata["revision"]}/{name}',target,row['size'])
    actual=checksum(target)
    if expected and actual!=expected: raise RuntimeError(f'Hash mismatch: {name}')
    result={'path':name,'bytes':target.stat().st_size,'sha256':actual,'official_lfs_hash_matches':bool(expected)}
    json_write(target.with_name(target.name+'.verified.json'),result)
    print(json.dumps(result),flush=True)
    return result
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    records=list(pool.map(stage,metadata['files']))
json_write(destination/'source_receipt.json',{'revision':metadata['revision'],'files':records,'status':'content_verified'})
