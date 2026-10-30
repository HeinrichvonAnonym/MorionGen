"""Forward-kinematic export of saved native qpos; retain cached-body discrepancy."""
import json
from pathlib import Path
import sys
import shutil
import re
import os
from types import SimpleNamespace

import joblib
import numpy as np
import mujoco_py

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'utils'))
from motiogen.runtime import checksum,json_write
BASE=ROOT/'develop_logs/test_runs/20261028_supplement'
OUT=BASE/'uhc_executed_pose';OUT.mkdir(exist_ok=True)
DATA=Path(json.loads((ROOT/'config/baselines.json').read_text())['data_root'])
repo=DATA/'baselines/UHC';sys.path.insert(0,str(repo))
import torch
from uhc.utils.config_utils.copycat_config import Config
from uhc.data_loaders.dataset_amass_single import DatasetAMASSSingle
from uhc.envs.humanoid_im import HumanoidEnv
torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
work=OUT/'work';(work/'data').mkdir(parents=True,exist_ok=True)
if not (work/'data/smpl').exists():(work/'data/smpl').symlink_to(DATA/'outputs/assets/closd-official/data/smpl')
if not (work/'assets').exists():(work/'assets').symlink_to(repo/'assets')
os.chdir(work)
cfg=Config(cfg_id='uhc_implicit',base_dir=str(repo),create_dirs=False);cfg.base_dir=str(work)
cfg.update(SimpleNamespace(test=True,num_threads=1,gpu_index=0,epoch=19000,show_noise=False,resume=None,no_log=True,debug=False,mode='stats',render_video=False,render_rfc=False,render=False,hide_expert=False,no_fail_safe=True,focus=False,output=str(work),shift_expert=False,smplx=False,hide_im=False))
cfg.data_specs['file_path']=str(repo/'sample_data/amass_copycat_take5_test_small.pkl');cfg.data_specs.pop('test_file_path',None)
cfg.data_specs['neutral_path']=str(repo/'sample_data/standing_neutral.pkl')
loader=DatasetAMASSSingle(cfg.data_specs,data_mode='train')
sample=loader.get_sample_from_key(take_key=sorted(loader.pickle_data)[0],full_sample=True,fr_start=-1)
env=HumanoidEnv(cfg,init_expert=sample,data_specs=cfg.data_specs,mode='test',no_root=cfg.no_root)
export=ROOT/'develop_logs/test_runs/20261029_003224_uhc_reference_export_38441486/reference_export.json'
references={r['source']:r for r in json.loads(export.read_text())['rows']}
rows=[]
for run_name in ['20261028_102935_uhc_tracking_3e35f23d','20261028_103019_uhc_tracking_96ffc470']:
    run=ROOT/'develop_logs/test_runs'/run_name
    source=joblib.load(run/'references.pkl')
    for row in json.loads((run/'tracking.json').read_text())['rows']:
        folder=Path(row['output']);result=joblib.load(folder/'native_result.pkl')
        reference=Path(references[row['source']]['path']);xml=reference.parent/'humanoid.xml'
        # The archived XML refers to temporary collision meshes removed by the
        # native robot destructor. Rebuild those with the same official loader,
        # verify all XML except the temporary directory, and preserve meshes.
        sample=loader.get_sample_from_key(take_key=row['source'],full_sample=True,fr_start=-1)
        env.load_expert(sample)
        generated=env.smpl_robot.export_xml_string().decode()
        normalize=lambda text:re.sub(r'/tmp/smpl/[^/]+/geom/','MESH_ROOT/',text)
        if normalize(generated)!=normalize(xml.read_text()):
            (OUT/'xml_generated_mismatch.txt').write_text(normalize(generated))
            (OUT/'xml_archived_mismatch.txt').write_text(normalize(xml.read_text()))
        import xml.etree.ElementTree as ET
        generated_tree=ET.fromstring(generated)
        meshes={Path(m.get('file')).stem:m.get('file') for m in generated_tree.findall('./asset/mesh')}
        tree=ET.fromstring(xml.read_text());modeldir=OUT/f'model_{len(rows):02d}';modeldir.mkdir(exist_ok=True)
        for mesh in tree.findall('./asset/mesh'):
            original=Path(meshes[Path(mesh.get('file')).stem]);destination=modeldir/original.name
            shutil.copyfile(original,destination);mesh.set('file',str(destination))
        preserved=modeldir/'humanoid.xml';ET.ElementTree(tree).write(preserved,encoding='unicode')
        model=mujoco_py.load_model_from_path(str(preserved));sim=mujoco_py.MjSim(model)
        ids=[i for i,n in enumerate(model.body_names) if n!='world']
        positions=[];rotations=[]
        for qpos in result['pred'][1:]:
            sim.data.qpos[:]=qpos;sim.data.qvel[:]=0;sim.forward()
            positions.append(sim.data.body_xpos[ids].copy())
            rotations.append(sim.data.body_xquat[ids][:,[1,2,3,0]].copy())
        xyz=np.asarray(positions)
        difference=float(np.max(np.abs(xyz-result['pred_jpos'][1:].reshape(-1,24,3))))
        if not np.isfinite(xyz).all():raise ValueError('Nonfinite native qpos FK')
        path=OUT/f'{len(rows):02d}.npz'
        np.savez_compressed(path,joints=xyz,rotations_xyzw=np.asarray(rotations),
            body_names=[model.body_names[i] for i in ids],beta=np.load(reference)['beta'][0],gender=source[row['source']]['gender'],fps=30.)
        rows.append(dict(source=row['source'],rfc=row['rfc_enabled'],path=str(path),
            source_result=str(folder/'native_result.pkl'),source_sha256=checksum(folder/'native_result.pkl'),
            xml=str(xml),xml_sha256=checksum(xml),saved_joint_max_abs_difference_m=difference))
        rows[-1].update(preserved_native_xml=str(preserved),archived_kinematic_xml_preserved=True,
            mesh_scope='Regenerated collision meshes only resolve absent temporary assets. No dynamics are evaluated; original XML body/joint geometry is preserved exactly. Generated XML may differ by 0.1mm rounding and is not substituted.',
            channel='Saved native executed qpos FK; distinct from archived MuJoCo cached body_xpos channel',
            cached_body_mean_distance_m=float(np.linalg.norm(xyz-result['pred_jpos'][1:].reshape(-1,24,3),axis=-1).mean()))
        print(json.dumps({'completed':len(rows),'max_difference_m':difference}),flush=True)
json_write(OUT/'export.json',dict(rows=rows,scope='Saved post-action qpos FK with original native XML and MuJoCo 2.1. This channel differs from archived cached body_xpos; differences are measured, not silently replaced. Native tracking errors retain original body_xpos. No new independent sample. Failed action replay is excluded and logs preserved.'))
