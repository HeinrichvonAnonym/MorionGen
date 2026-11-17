"""Development-only R -> D audit. Never imports or executes the K fitter."""
import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'utils'))
from motiogen.degradation.reference import read_bvh, sample, window
from motiogen.degradation.positions import project, write, SOURCE
from motiogen.degradation.rig import aligned
from motiogen.degradation.codec import encode as aligned_codec
from motiogen.degradation.metrics import compare, stats
from motiogen.evalkit.k.export import retarget, serialize, humanml_codec, target_template
from motiogen.evalkit.k.motion import load
from motiogen.evalkit.k.calibration import UPRIGHT

METHODS=['motionlcm','omnicontrol','ardy','chois','closd','dart','kimodo','primal']
CFG=ROOT/'config/k_degradation_20261117'


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def native_check(target,decoded,parents,reference):
    # Verify exported positions against rotations actually read back, with the
    # declared asset geometry. This is separate from accuracy relative to R.
    idx=np.searchsorted(target.times,decoded.times)
    assert np.allclose(target.times[idx],decoded.times)
    r=decoded.rotations
    if r is None:return {'status':'position_profile; independent_rotations_absent'}
    if target.method=='closd':r=r@UPRIGHT
    rest=np.asarray(target.provenance.get('neutral_joints',target_template(target.method)[1]))
    p=np.zeros((len(idx),len(parents),3))
    for j,par in enumerate(parents):
        p[:,j]=target.positions[idx,j] if par<0 else p[:,par]+np.einsum('tij,j->ti',r[:,par],rest[j]-rest[par])
    n=min(p.shape[1],decoded.positions.shape[1])
    result={'declared_asset_fk_cm':stats(np.linalg.norm(p[:,:n]-decoded.positions[:,:n],axis=-1)*100)}
    # CLoSD canonical positions have 22 nodes; native rotations have 24.
    if target.method=='chois':
        result['declared_asset_fk_cm']=stats(np.linalg.norm(p[:,:22]-decoded.positions[:,:22],axis=-1)*100)
    bases=np.asarray(target.provenance['retarget_bases'])
    source_lookup={n:i for i,n in enumerate(reference.rotation_names)}
    rotations={}
    for key,candidates in [('head',['head']),('thorax',['spine3','Spine3'])]:
        name=next(n for n in candidates if n in target.names)
        j=target.names.index(name)
        source='Head' if key=='head' else ('Spine3' if 'Spine3' in source_lookup else 'Spine2')
        recovered=r[:,j]@bases[j].T
        truth=reference.rotations[:,source_lookup[source]]
        from scipy.spatial.transform import Rotation
        error=np.degrees(Rotation.from_matrix(truth.swapaxes(-1,-2)@recovered).magnitude())
        rotations[key]=stats(error)
    result['declared_frame_transfer_deg']=rotations
    result['rotation_scope']='source-native frames transported by recorded fixed bases; no anatomical calibration claim'
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--limit',type=int)
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--source-geometry',action='store_true')
    ap.add_argument('--baseline-run',type=Path)
    ap.add_argument('--start',type=int,default=0)
    ap.add_argument('--stop',type=int,default=82)
    args=ap.parse_args()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(output).free <80*1024**3:raise RuntimeError('80 GiB reserve')
    windows=json.loads((CFG/'windows.json').read_text())['records']
    trials=[json.loads(s) for s in (ROOT/'config/gold_k_20261115/trials.jsonl').read_text().splitlines()]
    dev={r['id'] for r in trials if r.get('partition')=='development'}
    assert len(windows)==82 and all(r['id'] in dev for r in windows)
    roots=json.loads((ROOT/'config/gold_k_20261115/evidence.json').read_text())['local_roots']
    implementation={str(p.relative_to(ROOT)):digest(p) for p in
                    [Path(__file__),ROOT/'python/scripts/degradation_humanml_codec.py',*sorted((ROOT/'utils/motiogen/degradation').glob('*.py'))]}
    receipt={'implementation':implementation,'windows_sha256':digest(CFG/'windows.json'),
             'scope':'development R-D only; K optimization prohibited','fitter_runs':0,
             'source_geometry':args.source_geometry,
             'baseline_run':str(args.baseline_run) if args.baseline_run else None}
    receipt_file=output/'receipt.json'
    if receipt_file.exists() and json.loads(receipt_file.read_text())!=receipt:raise ValueError('Changed implementation: use new output directory')
    receipt_file.write_text(json.dumps(receipt,indent=2)+'\n')
    for row in windows[args.start:args.stop][:args.limit]:
        key=row['window_key'];resultfile=output/'scores'/f'{key}.json'
        if args.resume and resultfile.exists():continue
        with zipfile.ZipFile(Path(roots['collaborate'])/row['source']['archive']) as archive:
            payload=archive.read(row['source']['member'])
        ref=window(read_bvh(payload),row['source_window_s'])
        refdir=output/'references';refdir.mkdir(exist_ok=True)
        np.savez_compressed(refdir/f'{key}.npz',positions=ref.positions,rotations=ref.rotations,times=ref.times,names=ref.names)
        results={'window':row,'raw_sha256':hashlib.sha256(payload).hexdigest(),'methods':{}}
        for method in METHODS:
            try:
                legacy,parents=retarget(ref,method)
                versions={}
                sr=sample(ref,legacy.times)
                if method in ('motionlcm','omnicontrol'):
                    candidate=project(sr,method)
                else:
                    candidate,_=aligned(sr,legacy,source_geometry=args.source_geometry)
                for label,target in [('legacy',legacy),('candidate',candidate)]:
                    if label=='legacy' and args.baseline_run:
                        base=json.loads((args.baseline_run/'scores'/f'{key}.json').read_text())
                        assert base['raw_sha256']==results['raw_sha256'] and base['window']==row
                        versions[label]=base['methods'][method]['versions'][label]
                        continue
                    folder=output/'inputs'/key/method/label
                    if label=='candidate' and method in ('motionlcm','omnicontrol'):
                        path=write(target,folder)
                        if method=='omnicontrol':aligned_codec(folder/'motion.npz')
                    else:
                        path=serialize(target,parents,folder)
                        if label=='candidate' and args.source_geometry:
                            # This declared custom rest rig supersedes stock body geometry.
                            # Zero betas are placeholders, not an assertion of stock SMPL FK.
                            rest=np.asarray(target.provenance['neutral_joints'])
                            rig={'version':'source-rest-rig-1','units':'m','axes':'right-handed Y-up',
                                 'names':target.names,'parents':parents,'neutral_joints':rest.tolist(),
                                 'geometry_policy':'fixed_for_entire_clip','stock_mesh_fk_compatible':False,
                                 'native_shape_parameters':'placeholder; use this explicit kinematic rig',
                                 'observed_nodes':target.provenance['source_position_observed'],
                                 'source_nodes':target.provenance['source_nodes'],
                                 'segment_frame_basis':target.provenance['retarget_bases'],
                                 'decoded_native_right_basis':UPRIGHT.tolist() if method=='closd' else np.eye(3).tolist(),
                                 'rotation_gauge':target.provenance['rotation_gauge']}
                            rig['chain_compression']=target.provenance.get('chain_compression',{})
                            rig['effective_compressed_links']=target.provenance.get('effective_compressed_links',[])
                            rigfolder=path.parent if path.suffix=='.npz' else path
                            (rigfolder/'input_rig.json').write_text(json.dumps(rig,indent=2)+'\n')
                            provfile=rigfolder/'k_provenance.json'
                            prov=json.loads(provfile.read_text())
                            prov.update({'shape':'per_clip_fixed_source_rig','rig':'input_rig.json',
                                         'rig_sha256':digest(rigfolder/'input_rig.json'),
                                         'geometry_contract':'custom kinematic rig; not stock mesh/physics geometry'})
                            provfile.write_text(json.dumps(prov,indent=2)+'\n')
                            if method=='dart':
                                file=folder/'smplx_params.npz'
                                with np.load(file) as z:data={k:z[k] for k in z.files}
                                from motiogen.motion import Y_TO_Z
                                data['transl']=target.positions[:,0]@Y_TO_Z.T-rest[0]
                                np.savez_compressed(file,**data)
                            elif method=='primal':
                                for name,sl in [('reference',slice(2,None)),('initial_history',slice(None,2))]:
                                    file=folder/'factorial'/f'{name}.npz'
                                    with np.load(file) as z:data={k:z[k] for k in z.files}
                                    data['pose'][0,:,:3]=target.positions[sl,0]-rest[0]
                                    np.savez_compressed(file,**data)
                    decoded=load(path,method)
                    truth=sample(ref,decoded.times)
                    item=compare(truth,decoded)
                    item['native_consistency']=native_check(target,decoded,parents,truth)
                    item['time']={'reference_frames':len(ref.times),'decoded_frames':len(decoded.times),
                                  'start_s':float(decoded.times[0]),'end_s':float(decoded.times[-1])}
                    # Evaluate temporal information removed by the target sampling rate
                    # by reconstructing back to original samples inside its time range.
                    tt=ref.times[(ref.times>=decoded.times[0])&(ref.times<=decoded.times[-1])]
                    item['source_clock_reconstruction']=compare(sample(ref,tt),sample(decoded,tt))
                    versions[label]=item
                results['methods'][method]={'status':'computed','versions':versions,
                    'acceptance':'pending; all six regions and native consistency must be reviewed'}
            except Exception as error:
                import traceback
                results['methods'][method]={'status':'failed','error':str(error),'traceback':traceback.format_exc()}
        resultfile.parent.mkdir(exist_ok=True)
        resultfile.write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
        print(key,{m:r['status'] for m,r in results['methods'].items()},flush=True)
    preservation=json.loads((CFG/'preservation.json').read_text())['files']
    changed=[p for p,h in preservation.items() if not (ROOT/p).exists() or digest(ROOT/p)!=h]
    (output/'preservation_check.json').write_text(json.dumps({'changed':changed,'checked':len(preservation)},indent=2)+'\n')
    assert not changed,changed


if __name__=='__main__':main()
