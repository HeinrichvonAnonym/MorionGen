"""Read-only full-model/native-coordinate rotation audit; no reference access."""
import argparse,json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
import numpy as np
from scipy.spatial.transform import Rotation
import biomech
from motiogen.evalkit.k.resampling import audit_model


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fits',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--expected',type=int,default=656);a=p.parse_args()
    if a.output.exists():raise ValueError('Keep audit snapshots immutable')
    rows=[]
    for path in sorted(a.fits.glob('*/*/model.osim')):
        row={'window':path.parent.parent.name,'method':path.parent.name,'status':'ok'}
        try:
            audit=audit_model(path);file=path.parent/'coordinates_refined_native.mot'
            if not file.exists():file=path.parent/'ik/ik.mot'
            t,labels,q=biomech.read_storage(file)
            if len(t)<2 or not np.isfinite(q).all() or not np.isfinite(t).all() or np.any(np.diff(t)<=0):raise ValueError('Invalid native data')
            row.update(audit=audit,native_file=str(file),native_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),native_frames=len(t))
            for name,joint in audit['joints'].items():
                if joint['classification']=='three_axis_so3':
                    v=q[:,[labels.index(n) for n in joint['rotation_coordinates']]]
                    r=Rotation.from_euler(joint['order'],v*np.pi/180*np.array(joint['signs']))
                    step=np.rad2deg((r[:-1].inv()*r[1:]).magnitude())
                    joint.update(max_native_coordinate_step_deg=float(abs(np.diff(v,axis=0)).max()),max_native_rotation_step_deg=float(step.max()),native_rotation_step_p95_deg=float(np.percentile(step,95)))
                elif joint['classification']=='one_axis_geodesic_equivalent':
                    v=q[:,labels.index(next(iter(joint['coordinates'])))];d=abs(np.diff(v))
                    joint.update(max_native_step_deg=float(d.max()),short_arc_equivalence_verified=bool(np.all(d<180-1e-8)))
        except Exception as error:row.update(status='failed',error=repr(error))
        rows.append(row)
    result={'expected':a.expected,'found':len(rows),'success':sum(r['status']=='ok' for r in rows),'rows':rows,
            'scope':'All declared model joints and native coordinate increments; FK equivalence is checked separately in controlled replays, not asserted by this XML audit',
            'attached_head_neck':{'policy':'global observed matrices: mask-aware SLERP; local chain rotations from matrix products, FK positions; no independent twist for position-only inputs','source':'utils/motiogen/evalkit/k/head_neck.py','sha256':hashlib.sha256((ROOT/'utils/motiogen/evalkit/k/head_neck.py').read_bytes()).hexdigest()}}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print({k:result[k] for k in ('expected','found','success')})
    if result['found']!=a.expected or result['success']!=a.expected:sys.exit(1)


if __name__=='__main__':main()
