"""Replay only the native trajectory stage on a pinned D-only fixed model/seed.

Source fitting and reference scores are never rerun/read. Outputs go through the
same FK, root, head-chain and export routines as a complete K4 fit.
"""
import argparse,hashlib,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('fit','input','settings','output'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--method',required=True)
    ap.add_argument('--require-control-reproduction',action='store_true')
    a=ap.parse_args()
    import numpy as np
    import opensim as osim
    import biomech
    from motiogen.evalkit.k.motion import load
    from motiogen.evalkit.k.config import digest
    from motiogen.evalkit.k.trajectory import refine
    from motiogen.evalkit.k.k4_fitting import write_coordinates,resample_native
    from motiogen.evalkit.k.output import write_output
    from motiogen.evalkit.k.resampling import interpolate_rotations
    osim.Logger.setLevelString('error')
    if a.output.exists():raise ValueError('Use a new replay output; preserve all prior candidates')
    source=json.loads((a.fit/'receipt.json').read_text())
    motion=load(a.input,a.method,config=ROOT/'config/evalkit_k4_20261117/input.json')
    if source['key']['inputs']!=motion.provenance['inputs']:raise ValueError('Frozen seed input differs')
    config=source['key']['config'].copy();config['trajectory_refinement']=json.loads(a.settings.read_text())
    if config['postprocess']!='native_preserving' or config['export_clock']!='native_multiple':raise ValueError('Replay requires exact native-knot policy')
    files=['model.osim','coordinates_before_trajectory.mot','receipt.json']
    fingerprints={n:digest(a.fit/n) for n in files}
    nt,labels,seed=biomech.read_storage(a.fit/'coordinates_before_trajectory.mot')
    np.testing.assert_allclose(nt-nt[0],motion.times-motion.times[0],atol=1e-7,rtol=0)
    points,names,_=motion.legacy();observed=np.array(points,copy=True)
    for name in motion.legacy_prior_names():
        if name in names:observed[:,list(names).index(name)]=np.nan
    a.output.mkdir(parents=True)
    shutil.copyfile(a.fit/'model.osim',a.output/'model.osim')
    shutil.copyfile(a.fit/'coordinates_before_trajectory.mot',a.output/'coordinates_before_trajectory.mot')
    native,candidate,trajectory=refine(a.output/'model.osim',labels,seed,observed,names,motion.times,config['trajectory_refinement'])
    write_coordinates(a.output/'coordinates_trajectory_candidate.mot',nt,labels,candidate)
    write_coordinates(a.output/'coordinates_refined_native.mot',nt,labels,native)
    (a.output/'trajectory_refinement.json').write_text(json.dumps(trajectory,indent=2)+'\n')
    times,q=resample_native(nt,labels,native,float(config.get('resample_min_rate_hz',100.)),config['export_clock'])
    rotation_receipt=None
    if config.get('rotation_resampling')=='whole_body_so3':
        q,rotation_receipt=interpolate_rotations(a.output/'model.osim',labels,native,nt,times,q)
    elif config.get('rotation_resampling') or config.get('hip_rotation_resampling'):
        raise ValueError('Replay supports audited whole-body SO3 or explicit legacy coordinate export')
    path_receipt=None
    if config.get('coordinated_path'):
        from motiogen.evalkit.k.resampling.coordinated import refine_path
        before=q.copy()
        q,path_receipt=refine_path(a.output/'model.osim',labels,native,nt,q,times,motion,config['coordinated_path'])
        np.savez_compressed(a.output/'interpolation_before_coordination.npz',times=times,labels=labels,coords=before)
        (a.output/'coordinated_path.json').write_text(json.dumps(path_receipt,indent=2)+'\n')
    values,head,_=write_output(motion,a.output,labels,q,times,config)
    from motiogen.evalkit.k.original_domain import assess
    domain=assess(biomech.MODEL,labels,values['coords'])
    (a.output/'original_domain_diagnostics.json').write_text(json.dumps(domain,indent=2)+'\n')
    key=dict(source['key'],config=config,implementation={str(p.relative_to(ROOT/'utils/motiogen/evalkit/k')):digest(p) for p in sorted((ROOT/'utils/motiogen/evalkit/k').rglob('*.py'))},
             native_seed_source=dict(path=str(a.fit),sha256=fingerprints,scope='D-only fixed model and pre-trajectory seed; all conditions use same source'))
    receipt=dict(source,key=key,head_neck_model=head,original_domain_diagnostics=domain,rotation_resampling=rotation_receipt,
                 coordinated_path=None if path_receipt is None else {k:v for k,v in path_receipt.items() if k!='frames'},
                 trajectory_replay=dict(source_fit=str(a.fit),source_sha256=fingerprints,runner_sha256=digest(__file__),
                                        scope='Numerical native-stage replay, no reference access; geometry and seed pinned, output pipeline retained'))
    if any(digest(a.fit/n)!=h for n,h in fingerprints.items()):raise ValueError('Frozen seed changed during replay')
    if a.require_control_reproduction:
        with np.load(a.fit/'fit.npz') as old:
            np.testing.assert_array_equal(values['names'],old['names'])
            np.testing.assert_allclose(values['times'],old['times'],atol=1e-9,rtol=0)
            np.testing.assert_array_equal(np.isfinite(values['positions']),np.isfinite(old['positions']))
            discrepancy=float(np.nanmax(np.abs(values['positions']-old['positions'])))
            comparison=dict(max_abs_position_m=discrepancy,limit_m=1e-7,passed=discrepancy<=1e-7,
                            source_fit_sha256=digest(a.fit/'fit.npz'),scope='Saved F control reproduction, not R accuracy')
        (a.output/'control_reproduction.json').write_text(json.dumps(comparison,indent=2)+'\n')
        if not comparison['passed']:raise ValueError('Control replay did not reproduce source FK; investigate before interpreting paired candidates')
    (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(output=str(a.output),frames=len(nt),squared_residual=trajectory['candidate_squared_residual'],success=trajectory['optimizer_success'])))


if __name__=='__main__':main()
