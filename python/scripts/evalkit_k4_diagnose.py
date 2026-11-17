"""Read-only decomposition of a saved fit using D and model assets, never R.

Optional foot scans fix all proximal coordinates. Releasing subtalar locking
uses an in-memory model copy and the asset's original range; it is diagnostic.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
import opensim as osim
osim.Logger.setLevelString('error')
import biomech
from motiogen.evalkit.k.motion import load,Motion
from motiogen.evalkit.k.k4_fitting import model_fk,resample_native
from motiogen.degradation.reference import sample
from motiogen.degradation.metrics import points,angle,stats
from motiogen.evalkit.k.scoring import DIRECTIONS


def fingerprint(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def compare_input(motion,times,positions,names):
    valid=(motion.times>=times[0])&(motion.times<=times[-1]);ts=motion.times[valid]
    d=points(sample(motion,ts));f=points(sample(Motion(positions,names,times,'model_fk'),ts))
    result={'D_frames':len(motion.times),'paired_frames':len(ts),'directions_deg':{},'positions_cm':{}}
    for name,(a,b) in {**DIRECTIONS,'pelvis_line':('left_hip','right_hip')}.items():
        result['directions_deg'][name]=stats(angle(d[b]-d[a],f[b]-f[a]))
    delta=f['pelvis_mid']-d['pelvis_mid']
    for name in ('pelvis_mid','left_hip','right_hip','shoulder_mid','left_shoulder','right_shoulder','left_wrist','right_wrist',
                 'left_knee','right_knee','left_ankle','right_ankle','left_foot','right_foot'):
        v=f[name]-d[name]
        result['positions_cm'][name]={'world':stats(np.linalg.norm(v,axis=-1)*100),
                                     'pelvis_relative':stats(np.linalg.norm(v-delta,axis=-1)*100)}
    return result


def diagnose(folder,source,method,out,scan=False):
    out.mkdir(parents=True,exist_ok=True)
    motion=load(source,method,config=ROOT/'config/evalkit_k4_20261117/input.json')
    with np.load(folder/'fit.npz') as z:data={k:z[k] for k in z.files}
    names=list(data['names']);labels=list(data['labels']);model_file=folder/'model.osim'
    stage_data={'stored':(data['times'],data['positions']),
                'pre_root':(data['times'],data.get('positions_pre_root',data['positions']))}
    rebuilt,_=model_fk(model_file,labels,data['coords'],assemble=False)
    rebuild_delta=float(np.max(np.linalg.norm(rebuilt-data['positions'],axis=-1)))
    if rebuild_delta>1e-7:raise ValueError('Saved FK and model coordinates disagree')
    coords=data.get('coordinates_filtered_unassembled',data['coords'])
    direct,_=model_fk(model_file,labels,coords,direct_couplers=True)
    stage_data['filtered_direct']=(data['times'],direct)
    for label,file in [('native_ik_direct',folder/'ik/ik.mot'),('refined_native_direct',folder/'coordinates_refined_native.mot')]:
        if not file.exists():continue
        ts,cols,q=biomech.read_storage(file)
        p,_=model_fk(model_file,cols,q,direct_couplers=True)
        stage_data[label]=(ts+motion.times[0],p)
        if label=='native_ik_direct':
            ti,qi=resample_native(ts,cols,q)
            pi,_=model_fk(model_file,cols,qi,direct_couplers=True)
            stage_data['native_linear_direct']=(ti+motion.times[0],pi)
    receipt={'input_files':motion.provenance['inputs'],'fit_sha256':fingerprint(folder/'fit.npz'),
             'model_sha256':fingerprint(model_file),'model_fk_reconstruction_max_m':rebuild_delta,
             'diagnostic_code_sha256':fingerprint(Path(__file__)),
             'stages':{},'scope':'D-only diagnostics, separate from original-reference acceptance'}
    for name,(times,p) in stage_data.items():
        np.savez_compressed(out/(name+'.npz'),times=times,positions=p,names=names)
        receipt['stages'][name]=compare_input(motion,times,p,names)
    if 'root_shift_m' in data:
        receipt['root_shift_cm']=stats(np.linalg.norm(data['root_shift_m'],axis=-1)*100)
    model=osim.Model(str(model_file));state=model.initSystem();cs=model.getCoordinateSet()
    receipt['coordinate_bounds']={}
    for j,name in enumerate(labels):
        c=cs.get(name)
        if c.getMotionType()!=1:continue
        bounds=np.rad2deg([c.getRangeMin(),c.getRangeMax()]);q=data['coords'][:,j]
        receipt['coordinate_bounds'][name]={'range_deg':bounds.tolist(),'locked':c.getDefaultLocked(),
            'observed_min_max_deg':[float(q.min()),float(q.max())],
            'within_0_1_deg_of_bound_fraction':float(np.mean((abs(q-bounds[0])<.1)|(abs(q-bounds[1])<.1)))}
    if scan:
        scans=[]
        input_points=points(sample(motion,np.clip(data['times'],motion.times[0],motion.times[-1])))
        for side,suffix in [('left','l'),('right','r')]:
            target=input_points[side+'_foot']-input_points[side+'_ankle']
            fv=data['positions'][:,names.index(side+'_foot')]-data['positions'][:,names.index(side+'_ankle')]
            errors=angle(fv,target)
            errors[(data['times']<motion.times[0])|(data['times']>motion.times[-1])]=np.nan
            chosen=sorted(set([int(np.nanargmin(errors)),int(np.nanargmax(errors)),int(np.nanargmin(abs(errors-np.nanmedian(errors))))]))
            probe=osim.Model(str(model_file));sub=probe.updCoordinateSet().get('subtalar_angle_'+suffix)
            sub.setDefaultLocked(False);state=probe.initSystem()
            ankle=probe.getCoordinateSet().get('ankle_angle_'+suffix);sub=probe.getCoordinateSet().get('subtalar_angle_'+suffix)
            aa=np.linspace(ankle.getRangeMin(),ankle.getRangeMax(),101)
            ss=np.linspace(sub.getRangeMin(),sub.getRangeMax(),71)
            for i in chosen:
                row={'side':side,'frame':i,'time_s':float(data['times'][i]),'stored_error_deg':float(errors[i])}
                biomech._set(probe,state,labels,data['coords'][i])
                current_a=ankle.getValue(state)
                ankles=np.unique(np.r_[aa,current_a])
                for condition,subvalues in [('asset_locked',[sub.getDefaultValue()]),('subtalar_released_original_range',ss)]:
                    best=(float('inf'),None,None)
                    for a in ankles:
                        ankle.setValue(state,float(a),False)
                        for s in subvalues:
                            sub.setValue(state,float(s),False);probe.realizePosition(state)
                            v=biomech._child_origin(probe,state,'mtp_'+suffix)-biomech._child_origin(probe,state,'ankle_'+suffix)
                            e=float(angle(v,target[i]))
                            if e<best[0]:best=(e,float(np.rad2deg(a)),float(np.rad2deg(s)))
                    row[condition]={'min_grid_error_deg':best[0],'ankle_deg':best[1],'subtalar_deg':best[2]}
                scans.append(row)
        receipt['foot_reachability_scan']={'samples':scans,'grid':'101 ankle x 71 subtalar plus original ankle coordinate; original ranges; in-D-clock frames only',
            'scope':'Selected D-error min/median/max frames, fixed proximal pose; subset grid diagnostic, not full-window validation or anatomical qualification'}
    (out/'diagnostics.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print('diagnosed',folder,'->',out,flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--fit',type=Path,required=True)
    ap.add_argument('--input',type=Path,required=True);ap.add_argument('--method',required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--scan',action='store_true')
    a=ap.parse_args();diagnose(a.fit,a.input,a.method,a.output,a.scan)


if __name__=='__main__':main()
