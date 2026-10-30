"""Record dated official publication evidence, without fabricated run attempts."""
import hashlib,json,sys,subprocess
from datetime import datetime,timezone
from pathlib import Path
import requests
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import ROOT,load_config,data_root
from motiogen.runtime import json_write
out=ROOT/'develop_logs/test_runs/20261028_supplement';base=data_root(load_config())
rows=[]
for name,org in [('PHC_MJX','ZhengyiLuo'),('PhysDiff','NVlabs')]:
    item={'method':name,'observed_at':datetime.now(timezone.utc).isoformat(),'requests':[],'inference_started':False}
    urls=([f'https://api.github.com/repos/{org}/{name}/releases',f'https://api.github.com/repos/{org}/{name}/contents']
          if name=='PHC_MJX' else ['https://nvlabs.github.io/PhysDiff/','https://api.github.com/repos/NVlabs/PhysDiff'])
    for ordinal,url in enumerate(urls):
        r=requests.get(url,timeout=30);p=out/'sources'/f'{name}_audit_{ordinal}.txt';p.write_text(r.text)
        item['requests'].append({'url':url,'http_status':r.status_code,'saved_response':str(p.relative_to(ROOT)),
                                'sha256':hashlib.sha256(r.content).hexdigest()})
    if name=='PHC_MJX':
        repo=base/'baselines/PHC_MJX'
        files=subprocess.check_output(['git','-C',str(repo),'ls-files']).decode().splitlines()
        item.update(commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip(),tracked_files=files,
                    published_policy_files=[x for x in files if Path(x).suffix in ['.pth','.pt','.pkl','.onnx','.safetensors']],
                    status='No bundled policy; official release response and load path require review',
                    performance_conclusion=None,equivalent_to_closd=False)
    else:
        item.update(status='Project page exposes paper/video but no executable download; named NVlabs repository API response recorded',
                    inference_not_started_reason='No official executable/weight entry obtained from checked sources',
                    scope='This audit is limited to the recorded official sources; not proof that no implementation exists anywhere.',
                    independent_reimplementation=False,performance_conclusion=None)
    rows.append(item)
json_write(out/'candidate_release_audit.json',{'rows':rows})
print(json.dumps(rows,ensure_ascii=False))
