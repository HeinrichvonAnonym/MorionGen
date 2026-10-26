"""Reconstruct native object geometry for paired visualization, CPU only."""
import hashlib
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import trimesh
import torch
from pytorch3d import transforms

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'utils'))
from motiogen.survey_motion import accepted_pilot_entries
DATA=Path(json.loads((ROOT/'config/baselines.json').read_text())['data_root'])
entries=json.loads((ROOT/'develop_logs/test_runs/20261024_native_pilots/audited_pilots.json').read_text())['rows']
matrix_path=ROOT/'develop_logs/test_runs/20261024_native_matrix_analysis/analysis.json'
if matrix_path.exists():entries+=accepted_pilot_entries(json.loads(matrix_path.read_text())['rows'],list(entries))
p=argparse.ArgumentParser()
p.add_argument('--extra-entries',type=Path,action='append',default=[])
args=p.parse_args()
for path in args.extra_entries:entries.extend(json.loads(path.read_text()))
entries=list({entry['run']:entry for entry in entries}.values())
rows=[]
for entry in entries:
    if entry['method']!='chois':continue
    run=ROOT/entry['run']
    native_files=list((run/'raw/chois_results/res_npz_files').rglob('*.npz'))
    if len(native_files)!=1:raise ValueError('Expected one native sequence')
    with np.load(native_files[0],allow_pickle=False) as meta:sequence=str(meta['seq_name'])
    name=sequence.split('_')[1]
    mesh_path=DATA/'outputs/assets/chois/processed_data/rest_object_geo'/f'{name}.ply'
    mesh=trimesh.load_mesh(mesh_path)
    vertices=np.asarray(mesh.vertices,dtype=np.float32)
    # Deterministic farthest-point display sample; metrics below use ALL vertices.
    selected=[0];distance=np.full(len(vertices),np.inf)
    for _ in range(min(320,len(vertices))-1):
        distance=np.minimum(distance,((vertices-vertices[selected[-1]])**2).sum(axis=1))
        selected.append(int(distance.argmax()))
    closest=[];errors=[]
    with np.load(run/'raw/native_prediction.npz',allow_pickle=False) as raw:
        rotations=transforms.quaternion_to_matrix(transforms.matrix_to_quaternion(torch.from_numpy(raw['object_rotations']))).numpy()
        for joints,rotation,center,expected in zip(raw['joints'],rotations,raw['object_positions'],raw['hand_object_distances_m']):
            world=(rotation@vertices.T).T+center
            distances=np.linalg.norm(joints[22:24,None,:]-world[None,:,:],axis=-1)
            indices=distances.argmin(axis=1)
            closest.append(world[indices].tolist())
            errors.append(float(np.max(np.abs(distances[np.arange(2),indices]-expected))))
        row={'run':entry['run'],'object_name':name,'mesh_path':str(mesh_path),
             'mesh_sha256':hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
             'full_mesh_vertices':len(vertices),'display_sample_indices':selected,
             'rest_points':vertices[selected].tolist(),'positions':raw['object_positions'].tolist(),
             'rotations':rotations.tolist(),'nearest_surface_points':closest,
             'native_rotation_projection':'trainer_chois.py:2618-2620 matrix_to_quaternion then quaternion_to_matrix, before geometry reconstruction',
             'raw_rotation_max_orthogonality_error':float(np.max(np.abs(raw['object_rotations']@raw['object_rotations'].transpose(0,2,1)-np.eye(3)))),
             'hand_distance_max_reconstruction_error_m':max(errors)}
    if max(errors)>2e-6:raise ValueError(f'Object reconstruction differs from native saved distances: {max(errors)}')
    rows.append(row)
output=ROOT/'develop_logs/test_runs/20261024_chois_survey_geometry/geometry.json'
output.parent.mkdir(parents=True,exist_ok=True)
output.write_text(json.dumps({'rows':rows,'scope':'320 deterministic mesh vertices for display only. Nearest hand-to-surface points and distance reconstruction use all original mesh vertices, native object rotations and centers, world Z-up metres.'}))
print(json.dumps({'output':str(output),'checks':[{'run':r['run'],'vertices':r['full_mesh_vertices'],'error_m':r['hand_distance_max_reconstruction_error_m']} for r in rows]}))
