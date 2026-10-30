"""Freeze text/source selection before observing any generated effect."""
import json,re,sys,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import load_config,data_root,ROOT
from motiogen.runtime import json_write,checksum
import numpy as np
assets=data_root(load_config())/'outputs/assets'
data=assets/'closd-official/data/dataset/HumanML3D'
test=set((data/'test.txt').read_text().splitlines())
classes={'walk':r'\bwalk(?:s|ing)?\b','run':r'\brun(?:s|ning)?\b','jump':r'\bjump(?:s|ing)?\b','wave':r'\bwav(?:e|es|ing)\b'}
used=set();selected=[]
with zipfile.ZipFile(data/'texts.zip') as z:
    members={Path(name).stem:name for name in z.namelist() if name.endswith('.txt') and Path(name).stem in test and not Path(name).stem.startswith('M')}
    for family,pattern in classes.items():
        found=[]
        for source_id in sorted(members):
            if source_id in used:continue
            for line in z.read(members[source_id]).decode().splitlines():
                fields=line.split('#')
                if len(fields)!=4 or not re.search(pattern,fields[0],re.I):continue
                if fields[2:]!=['0.0','0.0']:continue
                candidates=[assets/'motionlcm/humanml3d/new_joint_vecs'/f'{source_id}.npy',data/'new_joint_vecs'/f'{source_id}.npy']
                reference=next((x for x in candidates if x.is_file()),None)
                frames=len(np.load(reference,mmap_mode='r')) if reference else None
                found.append({'source_id':source_id,'class':family,'text':fields[0],'tokens':fields[1],
                    'prompt':'Generate motion: '+fields[0],'requested_frames':frames or 196,
                    'requested_length_source':'reference' if reference else 'pilot request default; missing reference length',
                    'reference_path':str(reference) if reference else None,'reference_sha256':checksum(reference) if reference else None})
                used.add(source_id);break
            if len(found)==2:break
        if len(found)!=2:raise RuntimeError('Insufficient source texts for '+family)
        selected+=found
requests=[dict(row,seed=seed) for row in selected for seed in [0,1]]
json_write(ROOT/'develop_logs/test_runs/20261028_supplement/generation_inputs.json',{'selection':'First two distinct unmirrored test source IDs per prespecified lexical class, first full-clip annotation; selected before generation. Lexical strata are pilot coverage, not population representativeness.',
    'text_archive_sha256':checksum(data/'texts.zip'),'reference_gap':sum(r['reference_path'] is None for r in selected),'requests':requests})
print(json.dumps(selected,indent=2))
