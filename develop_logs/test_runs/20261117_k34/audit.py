import json,hashlib
from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET
root=Path.cwd();out=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k34_20261117')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
f=json.loads((out/'freeze.json').read_text());bad=[p for p,h in f['protected'].items() if sha(root/p)!=h];badref=[p for p,h in f['references'].items() if sha(p)!=h]
assert not bad and not badref
records=[]
for path in (out/'fits').glob('*/*/*/fit/kinematics.npz'):
 z=np.load(path);labels=list(z['labels']);tree=ET.parse(path.parent/'model.osim');coords={c.attrib['name']:c for c in tree.findall('.//Coordinate')};bounds={}
 for n in labels:
  if not n.startswith(('knee_angle_','hip_','ankle_angle_')) or n.endswith('beta'):continue
  lo,hi=np.rad2deg(np.fromstring(coords[n].findtext('range'),sep=' '));v=z['coordinates'][:,labels.index(n)]
  bounds[n]={'min_deg':float(v.min()),'max_deg':float(v.max()),'lower_share':float(np.mean(abs(v-lo)<.1)),'upper_share':float(np.mean(abs(v-hi)<.1))}
 records.append({'path':str(path),'bounds':bounds})
report={'protected_files_unchanged':len(f['protected']),'reference_files_unchanged':len(f['references']),'new_holdout_runs':0,'bounds':records}
(root/'develop_logs/test_runs/20261117_k34/audit.json').write_text(json.dumps(report,indent=2))
print('Protected',len(f['protected']),'references',len(f['references']),'fit bounds',len(records))
