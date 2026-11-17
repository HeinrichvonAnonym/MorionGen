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
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    motion=load(a.input,a.method,config=ROOT/'config/evalkit_k4_20261117/input.json')
    saved=json.loads((a.fit/'receipt.json').read_text())
    if saved['key']['inputs']!=motion.provenance['inputs']:raise ValueError('Replay input identity changed')
    native_file=a.fit/'coordinates_refined_native.mot'
    if not native_file.exists():native_file=a.fit/'ik/ik.mot'
    t,labels,q=biomech.read_storage(native_file);t,clock=native_clock(t,motion.times)
    new,linear=resample_native(t,labels,q);model=a.fit/'model.osim'
    corrected,receipt=hip_slerp(model,labels,q,t,new,linear)
    conditions=[('native',t,q),('linear',new,linear),('hip_slerp',new,corrected)]
    if a.whole_body:
        from motiogen.evalkit.k.resampling import interpolate_rotations
        whole,whole_receipt=interpolate_rotations(model,labels,q,t,new,linear)
        conditions.append(('whole_so3',new,whole));receipt['whole_body']=whole_receipt
    definition_file=a.fit/'head_neck_model.json'
    if definition_file.exists():
        definition=json.loads(definition_file.read_text());definition_source=str(definition_file)
    else:
        native_after,_=compensate(model,t+motion.times[0],labels,q,motion)
        definition,_=attach(motion,model,labels,native_after,t+motion.times[0],attachment_policy='input_fixed_offset')
        definition_source='D-only calibration at native clock; shared unchanged by all replay conditions'
    (a.output/'head_neck_model.json').write_text(json.dumps(definition,indent=2)+'\n')
    for condition,times,coordinates in conditions:
        physical=times+motion.times[0]
        after,root=compensate(model,physical,labels,coordinates,motion)
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
