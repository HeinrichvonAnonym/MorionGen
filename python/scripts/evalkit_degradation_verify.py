"""Independent artifact FK, source-frame and input-boundary checks after export."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'utils'))
from motiogen.evalkit.k.motion import Motion,load
from motiogen.degradation.reference import sample
from motiogen.degradation.metrics import stats,points,REGIONS


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run',type=Path);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();records=[]
    expected={'motionlcm':{'joints','joint_names','fps','times','metadata_json'},
              'omnicontrol':{'joints','joint_names','fps','times','metadata_json'},
              'ardy':{'joints','joint_names','fps','times','metadata_json','global_rotations'},
              'kimodo':{'posed_joints','global_rot_mats','local_rot_mats','fps','times'}}
    for scorefile in sorted((args.run/'scores').glob('*.json')):
        score=json.loads(scorefile.read_text());key=scorefile.stem
        with np.load(args.run/'references'/f'{key}.npz') as z:
            ref=Motion(z['positions'],list(z['names']),z['times'],'reference',z['rotations'],list(z['names']))
        # User-authorized tolerance stratum, selected using R alone. These are
        # engineering event thresholds, not a diagnosis of invalid source data.
        ref_points=points(ref,True)
        body_step=np.max(np.stack([np.linalg.norm(np.diff(v,axis=0),axis=-1)
                                  for v in ref_points.values()],1),axis=1)
        ri=[j for j,n in enumerate(ref.names) if n in ('Hips','Head') or 'spine' in n.lower()]
        rr=ref.rotations[:,ri]
        angle_step=np.degrees(Rotation.from_matrix((rr[:-1].swapaxes(-1,-2)@rr[1:]).reshape(-1,3,3)).magnitude()).reshape(len(rr)-1,-1).max(1)
        event_ids=np.flatnonzero((body_step>.25)|(angle_step>45))
        events=[{'start_s':float(ref.times[i]),'end_s':float(ref.times[i+1]),
                 'body_step_m':float(body_step[i]),'rotation_step_deg':float(angle_step[i])} for i in event_ids]
        for method in score['methods']:
            folder=args.run/'inputs'/key/method/'candidate'
            source=folder/'factorial/reference.npz' if method=='primal' else folder
            motion=load(source,method);truth=sample(ref,motion.times)
            item={'window':key,'method':method,'input_hashes':{},'head_neck_relative_deg':None}
            rp,dp=points(truth,True),points(motion)
            item['velocity_error_cm_s']={region:stats(np.stack([
                np.linalg.norm(np.diff(dp[name]-rp[name],axis=0)/np.diff(motion.times)[:,None],axis=-1)*100
                for name in nodes],1)) for region,nodes in REGIONS.items()}
            mask=(ref.times>=motion.times[0])&(ref.times<=motion.times[-1])
            raw=Motion(ref.positions[mask],ref.names,ref.times[mask],'reference')
            interpolated=sample(Motion(motion.positions,motion.names,motion.times,method),raw.times)
            raw_points,interpolated_points=points(raw,True),points(interpolated)
            jump_mask=np.zeros(len(raw.times),bool)
            for event in events:
                for left,right in zip(motion.times[:-1],motion.times[1:]):
                    if max(left,event['start_s'])<min(right,event['end_s']):
                        # One output-frame neighbourhood on each side covers
                        # interpolation adjoining the source transition.
                        pad=right-left
                        jump_mask|=(raw.times>=left-pad)&(raw.times<=right+pad)
            item['source_jump_events']=events
            item['temporal_strata_policy']='R-only: >25cm body point step OR >45deg root/head/spine step per source frame; overlapping D intervals plus one D-frame on each side; tolerance stratum, never suppress D-only errors'
            item['original_clock']={}
            for region,nodes in REGIONS.items():
                errors=np.stack([np.linalg.norm(interpolated_points[n]-raw_points[n],axis=-1)*100 for n in nodes],1)
                frame,joint=np.unravel_index(np.argmax(errors),errors.shape)
                item['original_clock'][region]={**stats(errors),'worst_time_s':float(raw.times[frame]),
                    'worst_joint':nodes[joint],'within_1cm_points':int((errors<=1).sum()),
                    'within_5cm_points':int((errors<=5).sum()),
                    'worst_in_first_output_interval':bool(raw.times[frame]<=motion.times[1]),
                    'continuous':stats(errors[~jump_mask]),'source_jump_tolerance':stats(errors[jump_mask])}
            item['time_coverage']={'original_window_frames':len(ref.times),'original_frames_in_common_interval':int(mask.sum()),
                                   'output_frames':len(motion.times),'boundary_frames_outside_interval':int((~mask).sum())}
            for file in sorted(folder.rglob('*')):
                if not file.is_file():continue
                assert file.suffix in ('.npz','.npy','.json'),file
                assert not any(n in file.name.lower() for n in ('score','gold','residual')),file
                item['input_hashes'][str(file.relative_to(folder))]=hashlib.sha256(file.read_bytes()).hexdigest()
            if method in expected:
                with np.load(folder/'motion.npz') as z:assert set(z.files)==expected[method],(method,z.files)
            rigfile=folder/'factorial/input_rig.json' if method=='primal' else folder/'input_rig.json'
            if rigfile.exists():
                rig=json.loads(rigfile.read_text());rest=np.asarray(rig['neutral_joints']);names=rig['names'];parents=rig['parents']
                right=np.asarray(rig['decoded_native_right_basis'])
                r=motion.rotations@right
                p=motion.positions
                if method=='closd':
                    from motiogen.motion import Y_TO_Z
                    with np.load(folder/'native_physics.npz') as z:p=z['joints']@Y_TO_Z
                if method=='chois':p=p[:,:22]
                fk=np.zeros_like(p);fk[:,0]=p[:,0]
                # Root translation is also independently checked against pose fields.
                if method=='primal':
                    with np.load(source) as z:fk[:,0]=z['pose'][0,:,:3]+rest[0]
                elif method=='dart':
                    from motiogen.motion import Y_TO_Z
                    with np.load(folder/'smplx_params.npz') as z:fk[:,0]=(z['transl']+rest[0])@Y_TO_Z
                for j,par in enumerate(parents):
                    if par>=0:fk[:,j]=fk[:,par]+np.einsum('tij,j->ti',r[:,par],rest[j]-rest[par])
                item['fk_position_max_cm']=float(np.max(np.linalg.norm(fk-p,axis=-1))*100)
                assert item['fk_position_max_cm']<1e-7,(key,method,item)
                bases=np.asarray(rig['segment_frame_basis'])
                h=names.index('head');c=names.index('Spine3' if method=='ardy' else 'spine3')
                dh=r[:,h]@bases[h].T;dc=r[:,c]@bases[c].T
                rh=truth.rotations[:,truth.names.index('Head')]
                rc=truth.rotations[:,truth.names.index('Spine3' if 'Spine3' in truth.names else 'Spine2')]
                delta=(rc.swapaxes(-1,-2)@rh).swapaxes(-1,-2)@(dc.swapaxes(-1,-2)@dh)
                item['head_neck_relative_deg']=stats(np.degrees(Rotation.from_matrix(delta).magnitude()))
                steps=np.degrees(Rotation.from_matrix((r[:-1].swapaxes(-1,-2)@r[1:]).reshape(-1,3,3)).magnitude()).reshape(len(r)-1,-1)
                spine=[j for j,n in enumerate(names) if 'spine' in n.lower()]
                item['spine_max_frame_step_deg']=float(np.max(steps[:,spine]))
                item['spine_max_frame_step_after_first_deg']=float(np.max(steps[1:,spine]))
                source_spine=[j for j,n in enumerate(truth.names) if 'spine' in n.lower()]
                rr=truth.rotations
                rs=np.degrees(Rotation.from_matrix((rr[:-1].swapaxes(-1,-2)@rr[1:]).reshape(-1,3,3)).magnitude()).reshape(len(rr)-1,-1)
                item['reference_spine_max_frame_step_deg']=float(np.max(rs[:,source_spine]))
                item['reference_spine_max_frame_step_after_first_deg']=float(np.max(rs[1:,source_spine]))
                item['reference_first_frame_identity_rotation_error_deg']=float(np.max(np.degrees(
                    Rotation.from_matrix(rr[0]).magnitude())))
                item['all_max_frame_step_deg']=float(np.max(steps))
                item['chain_compression']=rig.get('chain_compression',{})
                item['effective_compressed_links']=rig.get('effective_compressed_links',[])
            records.append(item)
    preserved=json.loads((ROOT/'config/k_degradation_20261117/preservation.json').read_text())['files']
    changed=[p for p,h in preserved.items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
    assert not changed,changed
    args.output.write_text(json.dumps({'run':str(args.run.resolve()),'export_receipt_sha256':hashlib.sha256((args.run/'receipt.json').read_bytes()).hexdigest(),
                                      'records':records,'count':len(records),'preserved_files':len(preserved),
                                      'changed_protected':changed,'K_fitting_runs':0},indent=2)+'\n')


if __name__=='__main__':main()
