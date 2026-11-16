#!/usr/bin/env python3
"""Freeze source evidence separately from project conventions; no model/data mutation."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
DATA=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen')

def evidence(path,needles):
    p=Path(path);p=p if p.is_absolute() else ROOT/p
    if not p.exists():return {'path':str(p),'status':'missing'}
    lines=p.read_text(errors='replace').splitlines()
    return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
            'lines':[{'line':i+1,'text':s.strip()[:260]} for i,s in enumerate(lines) if any(n in s for n in needles)][:16]}

def field(value,kind):return {'value':value,'classification':kind}

def main():
    definitions={
     'motionlcm':('MotionLCM/mld/data/humanml/scripts/motion_process.py',['Y-axis','r_pos[..., 1]','positions = qrot'],'Y-up','HumanML22 positions; reconstructed root quaternion wxyz; independent terminal head rotation absent'),
     'omnicontrol':('OmniControl/data_loaders/humanml/scripts/motion_process.py',['Y-axis','recover_from_ric','r_pos[..., 1]'],'Y-up','HumanML263 RIC decoding; root quaternion wxyz; rotation features derived from retained positions in controlled export'),
     'ardy':('ardy/ardy/postprocess.py',['fk(','global_rot_mats','neutral_joints','joint_parents'],'Y-up','SOMA/cskel27 local and global matrices, world acts on neutral offsets'),
     'chois':('chois_release/manip/data/long_cano_traj_dataset.py',['floor_z=True','local2global_pose','global_rot_mat'],'Z-up','SMPL body22 global matrices plus two extra hand positions'),
     'closd':('CLoSD/closd/utils/smpllib/smpllib/smpl_local_robot.py',['upright_start','1.2091996'],'Z-up','Isaac rigid-body world xyzw; upright-start rest frame differs from SMPL neutral'),
     'dart':('DART/utils/smpl_utils.py',['global_orient','body_pose','transform_local_points_to_global'],'Z-up canonical export','SMPL-X global root and 21 local rotation matrices; FK parent multiplication'),
     'kimodo':('kimodo/kimodo/exports/motion_io.py',['Y-up','meters','skeleton.fk','global_rot_mats'],'Y-up','SOMA77 local/global matrices with neutral-offset FK; per-asset names and parents'),
     'primal':('PRIMAL/primal/utils/data_io.py',['transl','glorot_aa','pose_body','concatenate'],'Y-up native; canonical archive Z-up','SMPL-X translation plus root and body axis-angle; two controlled history frames excluded')}
    records={}
    for m,(path,needles,axes,rotation) in definitions.items():
        sources=[evidence(DATA/'baselines'/path,needles),evidence('utils/motiogen/evalkit/k/motion.py',["method == '"+m,"method=='"+m,'native_y','Y_TO_Z'])]
        cal=ROOT/'config/evalkit_k_calibration'/f'{m}.json'
        if cal.exists():sources.append(evidence(cal,['sha256','source_path','axes','registration_note','global_rotation_semantics']))
        records[m]={'scope':'current registered native output; missing native sidecars remain legacy positions-only',
          'units':field('metres; canonical output schema','source_or_adapter_contract'),
          'world_axes':field(axes,'source_code_and_adapter_contract'),
          'rotation':field(rotation,'source_code'),
          'handedness':field('right-handed proper rotations; determinant checked','deterministic_conversion'),
          'neutral_topology_lengths':field('source asset rest joints/parents; input geometry for observed lengths','source_asset'),
          'head_thorax_anatomy':field('neutral model axes reused as segment convention; no independently supplied anatomical calibration found in registered assets','project_assumption'),
          'sources':sources}
    records['chois']['rest_recovery']=field('unknown sequence: median inverse FK offsets from current input, followed by bone-direction consistency gate','estimated_from_motion')
    for method in ['motionlcm','omnicontrol','chois']:
        records[method]['controlled_fixture_shape']=field('controlled exporter reuses the PRIMAL SMPL22 rest asset; this is an experimental fixture, not this generator\'s supplied subject shape','project_convention')
    for method in ['motionlcm','omnicontrol']:
        records[method]['neutral_topology_lengths']=field('HumanML22 hierarchy; observed positions determine lengths; no independent native segment rest rotations in the retained input','source_topology_and_input_geometry')
    records['closd']['sources'].append(evidence('utils/motiogen/closd_native.py',['rigid_body_rot','rotations_xyzw']))
    records['kimodo']['sources'].append(evidence(DATA/'baselines/kimodo/docs/source/user_guide/output_formats.md',['rotation','meter','Y-up','posed_joints']))
    records['andy']={'units':field('CSV explicitly [mm], timestamp [s]','source_header'),
      'world_axes':field('Z-up used by existing importer; checked against standing marker geometry, no per-record world-axis declaration in CSV header','geometry_inference'),
      'head_thorax_anatomy':field('four-marker head and C7/T10/CLAV/STRN thorax frames; right/superior/posterior convention defined by project','project_convention'),
      'marker_template':field('first complete observed frames estimate redundant marker template, excluded from reference score','estimated_from_motion'),
      'sources':[evidence('develop_logs/test_runs/20261114_152201_gold_data_audit/qualisys_csv.zip.sample.txt',['[mm]','MARKER_NAMES']),evidence('utils/motiogen/evalkit/k/reference.py',['z_to_y','right=','forward=','up=','front='])],
      'publisher_url':'https://doi.org/10.1177/0278364919882089','publisher_access':'DOI fetch unavailable in audit; no unsupported axis claim attributed to paper'}
    records['han']={'units':field('metres in B3D markerObservations','container_API_contract'),
      'world_axes':field('Y-up importer assumption; preserve B3D marker frame and verify marker/gravity geometry','container_convention_and_geometry_check'),
      'head_thorax_anatomy':field('head anterior/posterior and trunk landmarks form project geometric segment frames','project_convention'),
      'sources':[evidence('python/scripts/evalkit_k_han_export.py',['markerObservations','includeProcessingPasses=False']),evidence('utils/motiogen/evalkit/k/reference.py',['head_l_ant','trunk_c7','Right-handed'])],
      'publisher_url':'https://nimblephysics.org/docs/working-with-addbiomechanics-data.html',
      'reference_scope':'observed markers only; processed joint kinematics are separately labelled estimates'}
    records['collaborate']={'units':field('centimetres inferred from segment lengths; BVH header has no unit declaration','geometry_inference'),
      'world_axes':field('Y-up inferred from rest skeleton and standing motions; preserve arbitrary world yaw','geometry_inference'),
      'rotation':field('Euler rotation order and translations read from each joint CHANNELS; parent FK; non-root translation retained','source_header'),
      'neutral':field('median native local translation including non-root channels','estimated_from_motion'),
      'head_thorax_anatomy':field('native published segment rotations; geometric chest rest basis reused for both head and chest in existing controlled protocol','project_assumption'),
      'sources':[evidence('utils/motion_retarget/bvh.py',['rotation_channels','include_nonroot_translation','global_r[','local_p'])],
      'publisher_url':'https://zenodo.org/records/6472155','publisher_scope':'release confirms BVH and task labels; page does not declare anatomical head/thorax axes or units'}
    records['collaborate']['translation']={'value':'animated position channels replace OFFSET on their axes; unanimated axes use OFFSET',
        'classification':'deterministic_conversion','sources':['https://raw.githubusercontent.com/blender/blender-addons/main/io_anim_bvh/import_bvh.py#L523',
        'https://bvhsdk.readthedocs.io/en/latest/_modules/bvhsdk/anim.html#Joints.getLocalTranslation']}
    result={'version':'k3-conventions-2-bvh-absolute-channels','records':records,
      'decisions':['Retain documented world/rotation conversions; do not estimate known axes from evaluated motion.',
                   'Retain uncertain anatomical correspondence as project assumption, not promoted by low controlled round-trip error.',
                   'Source rest basis and marker geometry give representation references, not independent per-vertebra anatomy.',
                   'HYOID fitting and C2/T1 correspondence unchanged and outside this version.',
                   'Correct CoLLaboratE animated local translation; preserve additive legacy parser as an explicit compatibility mode.']}
    target=ROOT/'config/evalkit_k3_20261116/conventions.json';target.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print('sources',len(records),'missing',[m for m,r in records.items() if any(s.get('status')=='missing' for s in r['sources'])])
if __name__=='__main__':main()
