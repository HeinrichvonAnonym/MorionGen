"""D-only paired coordinate-domain/initialization probes, never window acceptance.

The same fixed model and objective are retained. Frames are prescribed before
fitting. Each frame is independent: this probe makes no temporal claim.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
import numpy as np
import opensim as osim
import biomech
from motiogen.evalkit.k.config import ROOT
from motiogen.evalkit.k.motion import load
from motiogen.evalkit.k.refinement import refine
from motiogen.evalkit.k.k4_fitting import model_fk
from motiogen.evalkit.k.diagnostic_domain import apply


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','input','method','protocol','output'):
        p.add_argument('--'+name,required=True,type=str if name=='method' else Path)
    a=p.parse_args();osim.Logger.setLevelString('error')
    protocol=json.loads(a.protocol.read_text());a.output.mkdir(parents=True,exist_ok=False)
    motion=load(a.input,a.method,config=ROOT/'config/evalkit_k4_20261117/input.json')
    points,names,_=motion.legacy()
    t,labels,q=biomech.read_storage(a.source/'coordinates_refined_native.mot')
    if len(q)!=len(points) or not np.allclose(t,motion.times-motion.times[0],atol=1e-8):
        raise ValueError('Native D clock and warm-start coordinates must coincide')
    indices=[0,len(q)//2,len(q)-1]
    identity={'commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD']).decode().strip(),
              'source_model_sha256':hashlib.sha256((a.source/'model.osim').read_bytes()).hexdigest(),
              'protocol':protocol,'input_provenance':motion.provenance,'indices':indices,
              'scope':'D-only three-frame diagnostic; no R or temporal acceptance; no promotion'}
    (a.output/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    rows=[]
    for condition,ranges in protocol['conditions'].items():
        folder=a.output/condition;folder.mkdir();modelpath=folder/'model.osim'
        shutil.copyfile(a.source/'model.osim',modelpath)
        domain=apply(modelpath,{'base':'release_subtalar_original_range','scope':'independent_kinematic',
                                'coordinate_ranges_deg':ranges})
        model=osim.Model(str(modelpath));state=model.initSystem();cs=model.getCoordinateSet()
        defaults=np.array([cs.get(n).getDefaultValue()*(180/np.pi if cs.get(n).getMotionType()==1 else 1) for n in labels])
        for frame in indices:
            for seed in protocol['seeds']:
                initial=q[frame].copy()
                if seed=='neutral':
                    for j,n in enumerate(labels):
                        if cs.get(n).getMotionType()==1:initial[j]=defaults[j]
                    for side in ('l','r'):initial[labels.index('knee_angle_'+side)]=q[frame,labels.index('knee_angle_'+side)]
                elif seed!='previous_fit':raise ValueError('Undeclared seed')
                fitted,receipt=refine(modelpath,labels,initial[None],points[frame:frame+1],names,protocol['refinement'])
                positions,outnames=model_fk(modelpath,labels,fitted,direct_couplers=True)
                dest=folder/f'{frame}_{seed}';dest.mkdir()
                np.savez_compressed(dest/'probe.npz',coords=fitted,labels=labels,positions=positions,names=outnames,
                                    times=motion.times[frame:frame+1])
                receipt.update(domain=domain,frame=frame,seed=seed)
                (dest/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
                row={'condition':condition,'frame':frame,'seed':seed,'loss':receipt['frames'][0]['after_loss']}
                rows.append(row);print(row,flush=True)
    (a.output/'completion.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
