"""Explicit test data transport and independent scoring with frozen K4 functions.

This module never fits a model or selects candidates. It imports the registered
CPU source tree; the test controller supplies paths rather than changing the
historical development manifests.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def setup(code):
    sys.path[:0] = [str(code/'utils'), str(code/'utils/motiogen'), str(code/'python/scripts')]


def export_input(ref, method, folder):
    """Same source-rig-v4 construction and file contract as the accepted exporter."""
    import numpy as np
    import evalkit_degradation_audit as old
    legacy, parents = old.retarget(ref, method)
    source = old.sample(ref, legacy.times)
    target = (old.project(source, method) if method in ('motionlcm', 'omnicontrol')
              else old.aligned(source, legacy, source_geometry=True)[0])
    if method in ('motionlcm', 'omnicontrol'):
        path = old.write(target, folder)
        if method == 'omnicontrol':
            old.aligned_codec(folder/'motion.npz')
    else:
        path = old.serialize(target, parents, folder)
        rest = np.asarray(target.provenance['neutral_joints'])
        rig = dict(version='source-rest-rig-1', units='m', axes='right-handed Y-up',
                   names=target.names, parents=parents, neutral_joints=rest.tolist(),
                   geometry_policy='fixed_for_entire_clip', stock_mesh_fk_compatible=False,
                   native_shape_parameters='placeholder; use this explicit kinematic rig',
                   observed_nodes=target.provenance['source_position_observed'],
                   source_nodes=target.provenance['source_nodes'],
                   segment_frame_basis=target.provenance['retarget_bases'],
                   decoded_native_right_basis=old.UPRIGHT.tolist() if method == 'closd' else np.eye(3).tolist(),
                   rotation_gauge=target.provenance['rotation_gauge'],
                   chain_compression=target.provenance.get('chain_compression', {}),
                   effective_compressed_links=target.provenance.get('effective_compressed_links', []))
        rigfolder = path.parent if path.suffix == '.npz' else path
        # Keep byte serialization identical to source-rig-v4 for hash equivalence.
        (rigfolder/'input_rig.json').write_text(json.dumps(rig, indent=2)+'\n')
        provfile = rigfolder/'k_provenance.json'
        prov = json.loads(provfile.read_text())
        prov.update(shape='per_clip_fixed_source_rig', rig='input_rig.json',
                    rig_sha256=sha(rigfolder/'input_rig.json'),
                    geometry_contract='custom kinematic rig; not stock mesh/physics geometry')
        provfile.write_text(json.dumps(prov, indent=2)+'\n')
        if method == 'dart':
            from motiogen.motion import Y_TO_Z
            file = folder/'smplx_params.npz'
            with np.load(file) as z:
                data = {k:z[k] for k in z.files}
            data['transl'] = target.positions[:,0]@Y_TO_Z.T-rest[0]
            np.savez_compressed(file, **data)
        elif method == 'primal':
            for name, sl in [('reference', slice(2,None)), ('initial_history', slice(None,2))]:
                file = folder/'factorial'/f'{name}.npz'
                with np.load(file) as z:
                    data = {k:z[k] for k in z.files}
                data['pose'][0,:,:3] = target.positions[sl,0]-rest[0]
                np.savez_compressed(file, **data)
    decoded = old.load(path, method)
    if not np.isfinite(decoded.positions).all():
        raise ValueError('Nonfinite test input; preserve source failure, do not impute')
    truth = old.sample(ref, decoded.times)
    consistency = old.native_check(target, decoded, parents, truth)
    fk = consistency.get('declared_asset_fk_cm')
    if fk and (fk['max'] is None or fk['max'] >= 1e-7):
        raise ValueError('Decoded declared-rig FK differs from serialized positions')
    return path, decoded, old.compare(truth, decoded), consistency


def reference_events(ref):
    """Frozen R-only jump rule; these events never enter the fitting job."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    from motiogen.degradation.metrics import points
    pp = points(ref, True)
    body = np.max(np.stack([np.linalg.norm(np.diff(v,axis=0),axis=-1) for v in pp.values()],1),axis=1)
    ri = [j for j,n in enumerate(ref.names) if n in ('Hips','Head') or 'spine' in n.lower()]
    rr = ref.rotations[:,ri]
    angles = np.degrees(Rotation.from_matrix((rr[:-1].swapaxes(-1,-2)@rr[1:]).reshape(-1,3,3)).magnitude()).reshape(len(rr)-1,-1).max(1)
    return [dict(start_s=float(ref.times[i]), end_s=float(ref.times[i+1]),
                 body_step_m=float(body[i]), rotation_step_deg=float(angles[i]))
            for i in np.flatnonzero((body>.25)|(angles>45))]


def prepare(row, root, output):
    import numpy as np
    from motiogen.degradation.reference import read_bvh, window, sample
    from motiogen.degradation.metrics import stats
    from scipy.spatial.transform import Rotation
    target = output/'prepared'/row['window_key']
    receipt = target/'receipt.json'
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved['row'] != row or any(sha(p)!=h for p,h in saved['artifact_hashes'].items()):
            raise ValueError('Prepared test artifact identity mismatch')
        return saved
    with zipfile.ZipFile(Path(root)/row['source']['archive']) as z:
        member = z.getinfo(row['source']['member'])
        if row['source'].get('crc32') and f'{member.CRC:08x}' != row['source']['crc32']:
            raise ValueError('Registered source archive member CRC changed')
        if member.file_size != row['source']['bytes']:
            raise ValueError('Registered source member size changed')
        raw = z.read(row['source']['member'])
    ref = window(read_bvh(raw), row['source_window_s'])
    target.mkdir(parents=True, exist_ok=True)
    reference = target/'reference.npz'
    np.savez_compressed(reference, positions=ref.positions, rotations=ref.rotations,
                        names=ref.names, times=ref.times)
    events = reference_events(ref)
    records = {}
    methods = ['motionlcm','omnicontrol','ardy','chois','closd','dart','kimodo','primal']
    for method in methods:
        folder = output/'runtime_inputs'/row['window_key']/method
        path, decoded, errors, consistency = export_input(ref, method, folder)
        head_error = None
        rigfile = path.parent/'input_rig.json' if method=='primal' else path/'input_rig.json'
        if rigfile.exists():
            rig = json.loads(rigfile.read_text())
            rot = decoded.rotations@np.asarray(rig['decoded_native_right_basis'])
            names = rig['names']; bases = np.asarray(rig['segment_frame_basis'])
            h = names.index('head'); c = names.index('Spine3' if method=='ardy' else 'spine3')
            dh,dc = rot[:,h]@bases[h].T, rot[:,c]@bases[c].T
            truth = sample(ref, decoded.times)
            rh = truth.rotations[:,truth.names.index('Head')]
            rc = truth.rotations[:,truth.names.index('Spine3' if 'Spine3' in truth.names else 'Spine2')]
            delta = (rc.swapaxes(-1,-2)@rh).swapaxes(-1,-2)@(dc.swapaxes(-1,-2)@dh)
            head_error = stats(np.degrees(Rotation.from_matrix(delta).magnitude()))
        feet = [decoded.names.index(s+'_foot') for s in ('left','right')]
        floor = float(min(np.percentile(decoded.positions[:,j,1],5) for j in feet))
        records[method] = dict(input=str(path), floor_m=floor, mass_kg=75.,
                               source_jump_events=events, head_neck_relative_deg=head_error,
                               degradation=errors, native_consistency=consistency,
                               input_hashes={str(f):sha(f) for f in sorted(folder.rglob('*')) if f.is_file()})
    files={str(reference):sha(reference)}
    for item in records.values():
        files.update(item['input_hashes'])
    result=dict(row=row, raw_sha256=hashlib.sha256(raw).hexdigest(), reference=str(reference),
                records=records, artifact_hashes=files)
    save(receipt,result)
    return result


def score_view(reference, source, method, artifact, fk, code):
    """Frozen score/acceptance maths with explicit test paths; no K tuning."""
    import numpy as np
    import evalkit_k4_ablation_score as old
    from motiogen.evalkit.k.head_scoring import compare
    from motiogen.evalkit.k.motion import Motion, load
    from motiogen.evalkit.k.timebase import supported, BOUNDARY_ROUNDOFF_S
    d = load(source, method, config=code/'config/evalkit_k4_20261117/input.json')
    with np.load(reference) as z:
        r=Motion(z['positions'],list(z['names']),z['times'],'reference',z['rotations'],list(z['names']))
    with np.load(fk) as z:
        fdata={k:z[k] for k in z.files}
        f=Motion(z['positions'],list(z['names']),z['times'],'model_fk')
    mask=old.jump_mask(d.times,d,artifact['source_jump_events'])
    full_r,full_f=old.at(r,d.times),old.at(f,d.times)
    result=old.score(full_r,d,full_f,mask,normalization_reference=r,fixed_observation_support=True)
    raw_d,raw_f=old.at(d,r.times),old.at(f,r.times)
    raw_mask=old.jump_mask(r.times,d,artifact['source_jump_events'])
    result['original_clock']=old.score(r,raw_d,raw_f,raw_mask,fixed_observation_support=True)
    for block,rt,dt,ft,times,mm in [(result,full_r,d,full_f,d.times,mask),
                                  (result['original_clock'],r,raw_d,raw_f,r.times,raw_mask)]:
        old.add_pelvis_diagnostic(block,rt,dt,ft,mm)
        block.update(compare(r,d,fdata,times,mm))
        block['fit_boundary_missing_frames']=int((~supported(times,f.times[0],f.times[-1])).sum())
    result['original_clock']['input_boundary_missing_frames']=int((~supported(r.times,d.times[0],d.times[-1])).sum())
    result.update(native_rotation_readout=artifact['head_neck_relative_deg'],
                  clock_boundary_roundoff_s=BOUNDARY_ROUNDOFF_S,
                  original_reference_file=str(reference), fitted_artifact=str(fk),
                  scope='Frozen full D/R clocks; historical-baseline paired subclock omitted (not an acceptance input).')
    return old.base.serializable(result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['prepare','score'])
    p.add_argument('--code-root',type=Path,required=True)
    p.add_argument('--request',type=Path,required=True)
    a=p.parse_args();setup(a.code_root)
    job=json.loads(a.request.read_text())
    if a.phase=='prepare':
        prepare(job['row'],job['root'],Path(job['output']))
    else:
        from motiogen.evalkit.k.strict_acceptance import assess
        policy=json.loads((a.code_root/'config/evalkit_k4_20261117/strict_v1/acceptance.json').read_text())
        s=score_view(job['reference'],job['input'],job['method'],job['artifact'],job['fk'],a.code_root)
        save(job['output'],s)
        save(str(job['output'])+'.acceptance.json',{'D_clock':assess(s,None,policy),'R_clock':assess(s['original_clock'],None,policy)})


if __name__=='__main__':
    main()
