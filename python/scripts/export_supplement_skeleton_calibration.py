"""Extract source-defined neutral joint frames, without loading motion weights."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import ROOT,load_config,data_root
from motiogen.runtime import checksum,json_write
from motiogen.adapters import joint_aliases
data=data_root(load_config());repo=data/'baselines/ardy';sys.path.insert(0,str(repo))
from ardy.skeleton import CoreSkeleton27,SOMASkeleton77
out=ROOT/'develop_logs/test_runs/20261028_supplement/skeleton_calibration.json';rows={}
for name,cls in [('ardy',CoreSkeleton27),('kimodo',SOMASkeleton77)]:
    skel=cls();aliases=joint_aliases(skel.name);path=Path(skel.folder)/'joints.p'
    rows[name]={'native_names':skel.bone_order_names,'names':[aliases.get(n,n) for n in skel.bone_order_names],
        'neutral_joints':skel.neutral_joints.cpu().numpy().tolist(),'parents':skel.joint_parents.cpu().numpy().tolist(),
        'source_path':str(path),'sha256':checksum(path),'axes':'Y-up','global_rotation_semantics':'Source FK global rotation acts on neutral offsets; canonical world conversion multiplies on left only.'}
json_write(out,rows);print(out)
