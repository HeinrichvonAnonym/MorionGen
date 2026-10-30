"""Record released sources, disk checks and clone attempts without claiming inference."""
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import requests
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import ROOT, load_config, data_root
from motiogen.runtime import json_write, checksum

cfg=json.loads((ROOT/'config/supplement_20261028.json').read_text())
out=ROOT/cfg['output'];out.mkdir(parents=True,exist_ok=True)
data=data_root(load_config());dest=data/'baselines'
assert dest.is_dir(), 'Existing mounted data disk required; do not create fallback assets on system disk'
receipt={'started_at':datetime.now(timezone.utc).isoformat(),'disk_free_bytes':{str(p):shutil.disk_usage(p).free for p in [ROOT,data]},'repositories':[]}
if shutil.disk_usage(data).free<80*1024**3:raise RuntimeError('Data reserve below 80 GiB')
for name,url in cfg['repositories'].items():
    folder=dest/name;row={'name':name,'url':url,'path':str(folder)}
    start=datetime.now(timezone.utc).isoformat()
    if not folder.exists():
        command=['git','clone','--depth','1',url,str(folder)]
        try:
            result=subprocess.run(command,capture_output=True,text=True,timeout=300)
            row.update(clone_returncode=result.returncode,clone_stderr=result.stderr[-4000:])
        except subprocess.TimeoutExpired:row.update(clone_returncode=None,clone_stderr='clone timeout 300 seconds')
    if (folder/'.git').exists():
        head=subprocess.run(['git','-C',str(folder),'rev-parse','HEAD'],capture_output=True,text=True)
        row.update(commit=head.stdout.strip(),readmes=[{'path':str(p.relative_to(folder)),'sha256':checksum(p)} for p in folder.iterdir() if p.is_file() and p.name.lower().startswith('readme')],status='source_cloned' if head.returncode==0 else 'clone_incomplete')
    else:row['status']='clone_failed'
    row.update(started_at=start,finished_at=datetime.now(timezone.utc).isoformat())
    receipt['repositories'].append(row);json_write(out/'source_audit.json',receipt)
    print(json.dumps(row),flush=True)
for name,url in {'physdiff_project':'https://nvlabs.github.io/PhysDiff/','rula_worksheet':'https://ergo.human.cornell.edu/Pub/AHquest/CURULA.pdf','reba_worksheet':'https://ergo.human.cornell.edu/Pub/AHquest/REBAworksheetbw.pdf','reba_tables':'https://ergo.human.cornell.edu/CUErgoTools/REBA%206.xls'}.items():
    response=requests.get(url,timeout=60)
    extension='.pdf' if url.endswith('.pdf') else '.xls' if url.endswith('.xls') else '.html'
    p=out/'sources'/(name+extension);p.parent.mkdir(exist_ok=True)
    row={'url':url,'http_status':response.status_code,'retrieved_at':datetime.now(timezone.utc).isoformat()}
    if response.ok:p.write_bytes(response.content);row.update(path=str(p.relative_to(ROOT)),sha256=checksum(p),bytes=len(response.content))
    receipt[name]=row;json_write(out/'source_audit.json',receipt)
    print(json.dumps({name:row}),flush=True)
