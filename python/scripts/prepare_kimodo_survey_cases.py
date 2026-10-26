"""Deterministic illustrative pairs, not outcome-selected or new experiments."""
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import ROOT,load_config
from motiogen.motion import canonicalize,save_motion,SOMA77_NAMES
from motiogen.runtime import checksum,json_write

audit=ROOT/'develop_logs/test_runs/20261024_kimodo_pair_audit/audit.json'
rows=json.loads(audit.read_text())['rows']
base=Path(load_config()['data_root'])/'benchmarks/kimodo-stratified-20261008'
out=ROOT/'develop_logs/test_runs/20261024_kimodo_survey_cases'
entries=[]
families=['constraints_withtext/root/path_2dposrot','constraints_withtext/end-effectors/hands_feet_posrot',
          'constraints_withtext/fullbody/inbetweening','constraints_withtext/mixture/root_path_fullbody']
for family in families:
    row=min((r for r in rows if r['family']==family),key=lambda r:r['case'])
    for label,folder in [('raw','generated_raw'),('postprocess','generated_postprocess')]:
        source=base/folder/row['case']/'motion.npz'
        if checksum(source)!=row['outputs'][label]['sha256']:raise ValueError('Audited motion changed')
        with np.load(source,allow_pickle=False) as data:joints=data['posed_joints']
        if joints.ndim==4:joints=joints[0]
        xyz,_=canonicalize(joints,'Y-up')
        destination=out/family/Path(row['case']).name/label/'motion.npz'
        save_motion(destination,xyz,SOMA77_NAMES,30,{'units':'m','axes':'right-handed Z-up',
            'source':str(source),'source_sha256':checksum(source),'selection':'lexicographically first accepted case in prespecified family'})
        entries.append({'method':'kimodo','case':family+'/'+Path(row['case']).name,'seed':row['seed'],
            'condition':{'correction':int(label=='postprocess')},'scene':{'constraints':[]},
            'run':str(destination.parent.relative_to(ROOT)),'motion_path':str(destination.relative_to(ROOT)),
            'native_case':row['case'],'source':str(source)})
json_write(out/'entries.json',entries)
print(json.dumps({'pairs':len(entries)//2,'selection':'lexicographic, not by outcome','output':str(out)}))
