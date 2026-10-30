"""Convert native UHC reference frames to CLoSD's upright SMPL frame convention."""
import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import joblib
import numpy as np
from scipy.spatial.transform import Rotation
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import ROOT,data_root,load_config
from motiogen.runtime import checksum,json_write
p=argparse.ArgumentParser();p.add_argument('--export',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
data=data_root(load_config());repo=data/'baselines/CLoSD'
target_xml=repo/'closd/data/robot_cache/smpl_humanoid_0.xml'
def skeleton(path):
    tree=ET.parse(path).getroot();global_coordinates=tree.find('compiler').get('coordinate','local')=='global'
    names=[];parents=[];positions=[]
    def visit(node,parent):
        idx=len(names);names.append(node.get('name'));parents.append(parent)
        positions.append(np.fromstring(node.get('pos'),sep=' '))
        for child in node.findall('body'):visit(child,idx)
    visit(tree.find('worldbody').find('body'),-1)
    offsets=np.stack(positions)
    if global_coordinates:
        offsets=np.stack([v-positions[parents[i]] if parents[i]>=0 else v for i,v in enumerate(positions)])
    return names,parents,offsets

def fk(rot,root,parents,offsets):
    result=np.empty((len(root),len(parents),3))
    for i,parent in enumerate(parents):
        result[:,i]=root if parent<0 else result[:,parent]+np.einsum('tij,j->ti',rot[:,parent],offsets[i])
    return result

names,parents,offsets=skeleton(target_xml)
# Exact axis permutation corresponding to the official SMPL robot zero-pose
# rotvec [1.2091996,1.2091996,1.2091996] (120 degrees about [1,1,1]).
upright=np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])
export=json.loads(a.export.read_text());motions={};rows=[]
for row in export['rows']:
    path=Path(row['path']);source=np.load(path)
    source_names=source['body_names'].tolist();indices=[source_names.index(n) for n in names]
    source_rot=Rotation.from_quat(source['rotations_xyzw'].reshape(-1,4)).as_matrix().reshape(len(source['joints']),24,3,3)
    own_names,own_parents,own_offsets=skeleton(path.parent/'humanoid.xml')
    assert own_names==source_names
    source_fk=fk(source_rot,source['joints'][:,0],own_parents,own_offsets)
    source_error=float(np.max(np.abs(source_fk-source['joints'])))
    if source_error>2e-5:raise ValueError(f'UHC reference FK mismatch {source_error}')
    target_rot=source_rot[:,indices]@upright.T
    roundtrip=float(np.max(np.abs(target_rot@upright-source_rot[:,indices])))
    if roundtrip>1e-10:raise ValueError('Rotation convention roundtrip failed')
    target_fk=fk(target_rot,source['joints'][:,0],parents,offsets)
    displacement=np.linalg.norm(target_fk-source['joints'][:,indices],axis=-1)
    pose=np.concatenate([source['pose_aa'][:,:66],np.zeros((len(target_fk),6))],axis=1)
    quat=Rotation.from_matrix(target_rot.reshape(-1,3,3)).as_quat().reshape(len(target_fk),24,4)
    motions[row['source']]={'pose_quat_global':quat,'root_trans_offset':torch.tensor(source['joints'][:,0],dtype=torch.float32),
        'pose_aa':pose,'fps':30,'beta':np.zeros(16),'gender':'neutral'}
    target=a.output/f'{len(rows):02d}_retargeted.npz'
    np.savez_compressed(target,joints=target_fk,source_joints=source['joints'][:,indices],
        rotations_xyzw=quat,body_names=names,fps=30)
    rows.append({'source':row['source'],'frames':len(target_fk),'source_sha256':checksum(path),
        'source_fk_max_abs_error_m':source_error,'rotation_roundtrip_max_error':roundtrip,
        'retarget_mean_joint_error_m':float(displacement.mean()),'retarget_max_joint_error_m':float(displacement.max()),
        'source_native_fps':float(source['fps']),'target_fps':30,'retargeted_path':str(target)})
joblib.dump(motions,a.output/'common_references.pkl')
json_write(a.output/'retarget_audit.json',{'rows':rows,'target_xml':str(target_xml),'target_xml_sha256':checksum(target_xml),
    'source_convention':'CLoSD smpl_local_robot.py upright_start zero-pose rotation; UHC smpl_to_qpose SMPL-H conversion retains first 66 coordinates and appends six zero hand coordinates.',
    'scope':'Common source motion and world root trajectory; PHC neutral body geometry differs from UHC subject shape. Retargeting error is separate from execution error. Native MotionLib height correction remains to be recorded at execution.'})
print(json.dumps({'references':len(rows),'max_source_fk_error':max(r['source_fk_max_abs_error_m'] for r in rows),'output':str(a.output)}))
