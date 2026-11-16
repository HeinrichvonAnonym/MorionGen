#!/usr/bin/env python3
"""Audit the BVH translation correction against source channels and explicit FK."""
import json,sys,hashlib,urllib.request
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from evalkit_k3_audit import archive,ROOTS,REG,parse_bvh
from motiogen.evalkit.k.bones import common
from motiogen.evalkit.k.export import ALIASES
from motiogen.evalkit.k.scoring import angles
from motiogen.evalkit.k.geometry import coordinate_precision_valid


def main():
    cfg=ROOT/'config/evalkit_k3_20261116';rows=[json.loads(l) for l in (REG/'trials.jsonl').read_text().splitlines()]
    excluded={r['id'] for r in json.loads((cfg/'superseded_reference_v1/light_summary.json').read_text())['failures']}
    examples=[r for r in rows if r['partition']=='development' and r['status']=='included' and (r['id'] in excluded or r['id']=='collaborate/LAB/EAWS/S1/S1P01R1.bvh')]
    results=[]
    for row in examples:
        s=row['source'];b=parse_bvh(archive(str(Path(ROOTS['collaborate'])/s['archive'])).read(s['member']));b.frames=b.frames[::max(1,int(np.ceil(len(b.frames)/240)))]
        old,_=b.forward_kinematics(True,dtype=np.float64);new,_=b.forward_kinematics(True,translation_mode='absolute_channels',dtype=np.float64)
        # Rotation-free local translations are independently read directly from channels.
        local=np.stack([j.offset for j in b.joints])[None].repeat(len(b.frames),0)
        for i,j in enumerate(b.joints):
            for k,c in enumerate(j.channels):
                if c in ['Xposition','Yposition','Zposition']:local[:,i,'XYZ'.index(c[0])]=b.frames[:,j.channel_start+k]
        np.testing.assert_allclose(local,b.local_positions(True,'absolute_channels'))
        results.append({'id':row['id'],'root_header_offset_m':(b.joints[0].offset*.01).tolist(),
            'animated_root_first_m':(new[0,0]*.01).tolist(),'animated_root_abs_max_m':float(np.max(abs(new[:,0]))*.01),
            'old_height_range_m':(np.array([np.ptp(old[:,:,1],axis=1).min(),np.ptp(old[:,:,1],axis=1).max()])*.01).tolist(),
            'corrected_height_range_m':(np.array([np.ptp(new[:,:,1],axis=1).min(),np.ptp(new[:,:,1],axis=1).max()])*.01).tolist(),
            'corrected_geometry_finite':bool(np.isfinite(new).all()),
            'shared_float32_world_position_eligible':bool(coordinate_precision_valid((new*.01).astype(np.float32)).all()),
            'previous_exclusion':row['id'] in excluded})
    url='https://raw.githubusercontent.com/blender/blender-addons/main/io_anim_bvh/import_bvh.py'
    payload=urllib.request.urlopen(url).read();folder=ROOT/'develop_logs/test_runs/20261116_k3/source_docs';folder.mkdir(exist_ok=True)
    (folder/'blender_import_bvh.py.txt').write_bytes(payload)
    result={'status':'confirmed_reader_contract_correction','classification':'deterministic channel interpretation supported by independent importers',
        'primary_sources':[{'url':url,'sha256':hashlib.sha256(payload).hexdigest(),'line_range':[519,526],
            'finding':'Blender armature location subtracts rest_head_local from animated bvh_loc; therefore animation supplies the local translation'},
            {'url':'https://bvhsdk.readthedocs.io/en/latest/_modules/bvhsdk/anim.html#Joints.getLocalTranslation',
             'finding':'getLocalTranslation returns OFFSET for three-channel joints, otherwise frame translation'}],
        'semantics':'animated local translation replaces OFFSET on declared position axes; unanimated axes retain OFFSET',
        'legacy_mode':'offset_plus_channel remains explicit default for old callers; K3 CoLLaboratE selects absolute_channels and float64',
        'effects':'old reference positions and body lengths were distorted; native rotations remain independently available. Seven extreme animated root origins still fail the shared float32 export precision contract; local pose can be inspected after removing the root before FK.',
        'superseded_experiment':'49ad7d0 at outputs/evalkit_k3_20261116; 166 result files retained, no configuration selection from this incomplete run',
        'historical_results':'preserved; old CoLLaboratE positional interpretations require this correction; Andy/Han optical reference unchanged',
        'records':results}
    (cfg/'reference_translation_correction.json').write_text(json.dumps(result,indent=2)+'\n');print('corrected examples',len(results))

if __name__=='__main__':main()
