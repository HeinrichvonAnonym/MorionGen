"""Input-only replay isolates native FK from coordinate/rotation interpolation."""
import argparse,json,hashlib,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
import numpy as np
import opensim as osim
osim.Logger.setLevelString('error')
import biomech
from motiogen.evalkit.k.motion import load
from motiogen.evalkit.k.k4_fitting import model_fk,resample_native
from motiogen.evalkit.k.rotation_resampling import hip_slerp
from motiogen.evalkit.k.root import compensate
from motiogen.evalkit.k.head_neck import attach,combine
from motiogen.evalkit.k.timebase import native_clock


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('fit','input','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('window','method'):p.add_argument('--'+name,required=True)
    p.add_argument('--whole-body',action='store_true',help='Replay all eligible joint rotations; also audit constrained joints')
    p.add_argument('--native-multiple',action='store_true',help='Add independent SO(3) export on a uniform grid containing every native input knot')
    p.add_argument('--coordinated-config',type=Path,help='Fixed-native-anchor path refinement; requires native-multiple')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    if a.coordinated_config and not a.native_multiple:raise ValueError('Coordinated path needs exact native knots')
    motion=load(a.input,a.method,config=ROOT/'config/evalkit_k4_20261117/input.json')
    saved=json.loads((a.fit/'receipt.json').read_text())
    if saved['key']['inputs']!=motion.provenance['inputs']:raise ValueError('Replay input identity changed')
    native_file=a.fit/'coordinates_refined_native.mot'
    if not native_file.exists():native_file=a.fit/'ik/ik.mot'
    t,labels,q=biomech.read_storage(native_file);t,clock=native_clock(t,motion.times)
    new,linear=resample_native(t,labels,q);model=a.fit/'model.osim'
    conditions=[('native',t,q),('linear',new,linear)]
    try:
        corrected,receipt=hip_slerp(model,labels,q,t,new,linear)
        conditions.append(('hip_slerp',new,corrected))
    except ValueError as error:
        if not a.whole_body or str(error)!='Canonical hip chart exceeds declared independent domain':raise
        receipt={'legacy_control_failures':{'hip_slerp':str(error)}}
    if a.whole_body:
        from motiogen.evalkit.k.resampling import interpolate_rotations
        whole,whole_receipt=interpolate_rotations(model,labels,q,t,new,linear)
        conditions.append(('whole_so3',new,whole));receipt['whole_body']=whole_receipt
    if a.native_multiple:
        from motiogen.evalkit.k.resampling import interpolate_rotations
        from motiogen.evalkit.k.resampling.clock import native_multiple_clock
        aligned,aligned_linear=resample_native(t,labels,q,clock_policy='native_multiple')
        aligned_q,aligned_receipt=interpolate_rotations(model,labels,q,t,aligned,aligned_linear)
        _,clock_receipt=native_multiple_clock(t)
        conditions.append(('whole_native_grid',aligned,aligned_q));receipt['native_clock_export']={'clock':clock_receipt,'rotations':aligned_receipt}
        if a.coordinated_config:
            from motiogen.evalkit.k.resampling.coordinated import refine_path
            policy=json.loads(a.coordinated_config.read_text())
            refined,path_receipt=refine_path(model,labels,q,t,aligned_q,aligned,motion,policy)
            conditions.append(('coordinated',aligned,refined));receipt['coordinated_path']=path_receipt
            receipt['coordinated_config_sha256']=hashlib.sha256(a.coordinated_config.read_bytes()).hexdigest()
    definition_file=a.fit/'head_neck_model.json'
    if definition_file.exists():
        definition=json.loads(definition_file.read_text());definition_source=str(definition_file)
    else:
        _,_,consistent=model_fk(model,labels,q,direct_couplers=True,return_coordinates=True)
        native_after,_=compensate(model,t+motion.times[0],labels,consistent,motion)
        definition,_=attach(motion,model,labels,native_after,t+motion.times[0],attachment_policy='input_fixed_offset')
        definition_source='D-only calibration at native clock; shared unchanged by all replay conditions'
    (a.output/'head_neck_model.json').write_text(json.dumps(definition,indent=2)+'\n')
    for condition,times,coordinates in conditions:
        physical=times+motion.times[0]
        _,_,consistent=model_fk(model,labels,coordinates,direct_couplers=True,return_coordinates=True)
        after,root=compensate(model,physical,labels,consistent,motion)
        _,names=model_fk(model,labels,after,direct_couplers=True)
        _,chain=attach(motion,model,labels,after,physical,fixed_definition=definition)
        points,names=combine(root['after'],names,chain)
        dest=a.output/'views'/condition/a.window/a.method;dest.mkdir(parents=True)
        np.savez_compressed(dest/'fk.npz',positions=points,names=names,times=physical,
            head_relative_thorax=chain['head_relative_thorax'],head_rotation_valid=chain['relative_rotation_valid'],
            head_world_rotation=chain['world_rotations'][:,-1])
        np.savez_compressed(dest/'coordinates.npz',coords=after,pre_root_coords=coordinates,labels=labels,times=physical)
        np.savez_compressed(dest/'root.npz',before=root['before'],after=root['after'],times=physical)
    receipt.update(source_fit=str(a.fit),native_file=str(native_file),head_definition_source=definition_source,
        implementation_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),*sorted((ROOT/'utils/motiogen/evalkit/k').rglob('*.py'))]},source_fit_key=saved['key'],clock=clock,
        native_coordinates_sha256=hashlib.sha256(native_file.read_bytes()).hexdigest(),
        commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD']).decode().strip(),
        interpretation='Isolated postprocessing replay, identical native fit and fixed head attachment; no R access or parameter selection')
    (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
