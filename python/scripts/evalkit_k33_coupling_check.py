"""Repair model-authored dependent coordinates in new artifacts; verify K endpoint invariance."""
import sys,json,shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from motiogen.evalkit.k.constraints import synchronize_couplers
from motiogen.evalkit.k.fitting3 import positions,write_coordinates
from motiogen.evalkit.k.config import digest
BASE=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs')
OUT=BASE/'evalkit_k33_conversion_20261117';LOG=ROOT/'develop_logs/test_runs/20261117_k_conversion'

def one(task):
    source,tag=task
    import opensim as osim,biomech
    osim.Logger.setLevelString('error')
    work=OUT/'coupled_fits'/tag;work.mkdir(parents=True,exist_ok=True)
    model_path=work/'model.osim';shutil.copy2(source/'model.osim',model_path)
    model=osim.Model(str(model_path));state=model.initSystem()
    with np.load(source/'kinematics.npz') as z:data={k:z[k] for k in z.files}
    t=data['times'];labels=list(data['labels']);q=data['coordinates']
    corrected,receipt=synchronize_couplers(model_path,model,labels,q)
    p,names=positions(model_path,t,labels,corrected)
    delta=float(np.max(abs(p-data['fitted'])));assert delta<1e-12,delta
    changed={r['dependent'] for r in receipt['couplers']};ix=[i for i,n in enumerate(labels) if n not in changed]
    np.testing.assert_array_equal(q[:,ix],corrected[:,ix])
    after,_=synchronize_couplers(model_path,model,labels,corrected);np.testing.assert_array_equal(after,corrected)
    states=[]
    for side in 'lr':
        k,b=labels.index('knee_angle_'+side),labels.index('knee_angle_'+side+'_beta');error=np.deg2rad(q[:,k])-q[:,b];peak=int(np.argmax(abs(error)))
        biomech._set(model,state,labels,q[peak]);before=model.getBodySet().get('patella_'+side).getPositionInGround(state).to_numpy()
        biomech._set(model,state,labels,corrected[peak]);after=model.getBodySet().get('patella_'+side).getPositionInGround(state).to_numpy()
        states.append({'side':side,'peak_time':float(t[peak]),'before_max_coupler_error_deg':float(np.degrees(np.max(abs(error)))),
                       'after_max_coupler_error_deg':float(np.degrees(np.max(abs(np.deg2rad(corrected[:,k])-corrected[:,b])))),
                       'patella_position_change_at_peak_m':float(np.linalg.norm(after-before))})
    data['coordinates']=corrected;np.savez_compressed(work/'kinematics.npz',**data)
    write_coordinates(work/'coordinates.mot',t,labels,corrected)
    result={'source':str(source),'source_kinematics_sha256':digest(source/'kinematics.npz'),'output':str(work),'endpoint_max_difference_m':delta,'coupling':receipt,'sides':states,
        'scope':'only declared dependent coordinates changed; original/degraded/reference and independent fitted coordinates untouched; no muscle-load validation'}
    (work/'receipt.json').write_text(json.dumps(result,indent=2));return result

def main():
    selected=[(p.parent/'fit',str(p.parent.relative_to(OUT/'fits'))) for p in (OUT/'fits').glob('*/*/result.json')]
    scan=json.loads((LOG/'coupling_scan.json').read_text());worst=max(scan['records'],key=lambda r:r['max_difference_deg'])
    selected.append((Path(worst['path']).parent,'historical_worst'))
    with ProcessPoolExecutor(max_workers=2) as pool:results=list(pool.map(one,selected))
    (LOG/'coupling_check.json').write_text(json.dumps(results,indent=2));print('checked',len(results),'endpoint_max',max(r['endpoint_max_difference_m'] for r in results),flush=True)
if __name__=='__main__':main()
