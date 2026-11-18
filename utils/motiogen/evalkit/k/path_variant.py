"""Compose an export-only candidate with a completed native fit.

The model, native coordinates and fixed head attachment are inherited intact.
Only the inserted path is solved; every result is rebuilt by model FK. This
stage accepts D and a fit receipt, never a reference motion or a score.
"""
import json
from pathlib import Path
import shutil
import numpy as np
from .config import digest


def build(motion, source, output, policy):
    import biomech
    from .k4_fitting import resample_native
    from .timebase import native_clock
    from .resampling import interpolate_rotations
    from .resampling.coordinated import refine_path
    from .candidate_quality import exact_indices
    from .output import write_output
    from .original_domain import assess as assess_original_domain

    source,output=Path(source).resolve(),Path(output).resolve()
    if output==source or output.is_relative_to(source):
        raise ValueError('Export variant must have an independent peer directory')
    original=json.loads((source/'receipt.json').read_text())
    settings=original['key']['config']
    if (original['version']!='k4-fit-1' or settings.get('postprocess')!='native_preserving'
            or settings.get('export_clock')!='native_multiple'
            or settings.get('constraint_policy')!='direct_couplers'
            or settings.get('rotation_resampling')!='whole_body_so3'):
        raise ValueError('Export variant requires a declared single native-preserving SO(3) fit')
    if original['key']['inputs']!=motion.provenance['inputs']:
        raise ValueError('Export variant input identity differs from native fit')
    immutable=('model.osim','coordinates_refined_native.mot','head_neck_model.json')
    fingerprints={name:digest(source/name) for name in (*immutable,'receipt.json','fit.npz')}
    key={**original['key'],'export_variant':{
        'version':'fixed_native_path_variant_v1','policy':policy,'source_sha256':fingerprints,
        'implementation':{str(p.relative_to(Path(__file__).parent)):digest(p)
                          for p in sorted(Path(__file__).parent.rglob('*.py'))}}}
    receipt_file=output/'receipt.json'
    if receipt_file.exists():
        receipt=json.loads(receipt_file.read_text())
        if receipt['key']!=key:raise ValueError('Export variant cache identity changed')
        with np.load(output/'fit.npz') as z:return {**dict(z),'receipt':receipt}
    if output.exists():
        raise ValueError('Incomplete export variant preserved; use an independent attempt directory')
    # Keep the complete source history without allowing partial output reuse.
    shutil.copytree(source,output)
    receipt_file.unlink()
    t,labels,q=biomech.read_storage(source/'coordinates_refined_native.mot')
    t,clock=native_clock(t,motion.times)
    times,initial=resample_native(t,labels,q,float(settings.get('resample_min_rate_hz',100.)),
                                  clock_policy='native_multiple')
    initial,rotation_receipt=interpolate_rotations(output/'model.osim',labels,q,t,times,initial)
    coordinates,path_receipt=refine_path(output/'model.osim',labels,q,t,initial,times,motion,policy)
    np.savez_compressed(output/'interpolation_before_coordination.npz',times=times,labels=labels,coords=initial)
    (output/'coordinated_path.json').write_text(json.dumps(path_receipt,indent=2)+'\n')
    head=json.loads((source/'head_neck_model.json').read_text())
    values,definition,_=write_output(motion,output,labels,coordinates,times,settings,fixed_head=head)
    indices=exact_indices(motion.times,values['times'])
    with np.load(source/'fit.npz') as z:
        old_indices=exact_indices(motion.times,z['times'])
        old=z['positions'][old_indices]
        if not np.array_equal(z['names'],values['names']):raise ValueError('Export point definitions changed')
    new=values['positions'][indices]
    if not np.array_equal(np.isfinite(old),np.isfinite(new)):
        raise ValueError('Native observation coverage changed in export variant')
    delta=float(np.max(np.abs(new[np.isfinite(old)]-old[np.isfinite(old)])))
    if delta>1e-7:raise ValueError('Native FK changed in export-only variant')
    for name in immutable:
        if digest(source/name)!=fingerprints[name]:raise ValueError('Source native fit was modified')
    for name in ('model.osim','coordinates_refined_native.mot'):
        if digest(output/name)!=fingerprints[name]:raise ValueError('Export variant changed its fixed model or native coordinates')
    # The rewritten definition may update descriptive fields; its geometry is fixed.
    for name in ('offsets_m','attachment_local_m','attachment_policy','source_frame_bases','body_to_source_transport'):
        if definition.get(name)!=head.get(name):raise ValueError('Export variant changed fixed head definition')
    original_domain=assess_original_domain(biomech.MODEL,labels,values['coords'])
    (output/'original_domain_diagnostics.json').write_text(json.dumps(original_domain,indent=2)+'\n')
    receipt={**original,'key':key,'version':'k4-export-variant-1','head_neck_model':definition,
             'original_domain_diagnostics':original_domain,'rotation_resampling':rotation_receipt,
             'coordinated_path':{k:v for k,v in path_receipt.items() if k!='frames'},
             'export_variant':{'source_fit':str(source),'source_sha256':fingerprints,
                               'native_fk_max_abs_difference_m':delta,'native_clock':clock,
                               'native_fit_recomputed':False,'head_geometry_reestimated':False,
                               'scope':'Fixed native fit; inserted FK path only; independent R acceptance required'}}
    receipt_file.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    return {**values,'receipt':receipt}
