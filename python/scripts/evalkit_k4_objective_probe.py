"""Original-domain paired objective probe on prespecified D-only sentinel frames."""
import argparse,json,hashlib,subprocess
from pathlib import Path
import numpy as np
import opensim as osim
osim.Logger.setLevelString('error')
import biomech
from motiogen.evalkit.k.config import ROOT
from motiogen.evalkit.k.motion import load
from motiogen.evalkit.k.refinement import refine
from motiogen.evalkit.k.k4_fitting import model_fk


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--protocol',type=Path,default=ROOT/'config/evalkit_k4_20261117/strict_v1/objective_probe.json')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);protocol=json.loads(a.protocol.read_text())
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    code={'commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD']).decode().strip(),
          'implementation_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (ROOT/'utils/motiogen/evalkit/k').glob('*.py')},
          'new_holdout_uses':0}
    (a.output/'identity.json').write_text(json.dumps(code,indent=2)+'\n')
    for w in protocol['windows']:
        source=a.run/'optimization_v2/fits/multi_pelvis_subtalar_registered_position1'/w/'kimodo'
        job=json.loads((a.run/'jobs'/f'{w}_kimodo.json').read_text());motion=load(job['input'],'kimodo',config=ROOT/'config/evalkit_k4_20261117/input.json')
        points,names,_=motion.legacy();t,labels,q=biomech.read_storage(source/'ik/ik.mot');indices=np.array([0,len(q)//2,len(q)-1])
        for condition,changes in protocol['conditions'].items():
            dest=a.output/f'{w}_kimodo_{condition}';dest.mkdir()
            settings={**protocol['shared_refinement'],**changes}
            fitted,receipt=refine(source/'model.osim',labels,q[indices],points[indices],names,settings)
            p,n=model_fk(source/'model.osim',labels,fitted,direct_couplers=True)
            np.savez_compressed(dest/'probe.npz',times=motion.times[indices],coords=fitted,labels=labels,positions=p,names=n)
            receipt.update(source_model=str(source/'model.osim'),source_model_sha256=hashlib.sha256((source/'model.osim').read_bytes()).hexdigest(),
                           inputs=motion.provenance['inputs'],indices=indices.tolist())
            (dest/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
            print(w,condition,[f.get('after_loss') for f in receipt['frames']],flush=True)
    (a.output/'completion.json').write_text(json.dumps({'conditions':len(protocol['conditions']),'windows':len(protocol['windows']),'promotion':'none'})+'\n')


if __name__=='__main__':main()
