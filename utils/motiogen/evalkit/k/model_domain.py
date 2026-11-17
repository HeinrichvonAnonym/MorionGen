"""Explicit copied-model domain probes, restricted to kinematic research.

The source asset is never overwritten. Extended ranges are engineering
hypotheses, not anatomical validation or a muscle-model operating domain.
"""
import numpy as np

PROFILES={
 'knee':{'ranges':{'knee_angle':(0,175)},'unlock':[]},
 'hip':{'ranges':{'hip_flexion':(-90,175),'hip_adduction':(-90,90),'hip_rotation':(-120,120)},'unlock':[]},
 'ankle':{'ranges':{'ankle_angle':(-90,90),'subtalar_angle':(-90,90)},'unlock':['subtalar_angle']},
 'combined':{'ranges':{'knee_angle':(0,175),'hip_flexion':(-90,175),'hip_adduction':(-90,90),'hip_rotation':(-120,120),'ankle_angle':(-90,90),'subtalar_angle':(-90,90)},'unlock':['subtalar_angle']},
}


def prepare(source,destination,labels,coordinates,profile):
    import opensim as osim
    from pathlib import Path
    from .config import digest
    if Path(source).resolve()==Path(destination).resolve():raise ValueError('domain probes require a distinct model path')
    spec={'ranges':{},'unlock':[]} if profile=='original' else PROFILES[profile];model=osim.Model(str(source));state=model.initSystem();cs=model.getCoordinateSet();data=coordinates.copy();fixed={};changes={}
    # Legacy optimizer columns may contain nominal values for locked DOFs.
    # Realize their actual fixed values before unlocking, preserving baseline FK.
    for j,n in enumerate(labels):
        c=cs.get(n)
        if c.getLocked(state):
            value=c.getValue(state);stored=np.degrees(value) if c.getMotionType()==1 else value
            fixed[n]={'physical_value':float(stored),'maximum_nominal_correction':float(abs(data[:,j]-stored).max())};data[:,j]=stored
    for side in ['l','r']:
        for prefix,ranges in spec['ranges'].items():
            n=prefix+'_'+side;c=cs.get(n);before=[float(np.degrees(c.getRangeMin())),float(np.degrees(c.getRangeMax()))]
            c.setRangeMin(np.deg2rad(ranges[0]));c.setRangeMax(np.deg2rad(ranges[1]));changes[n]={'before':before,'after':list(ranges)}
        for prefix in spec['unlock']:cs.get(prefix+'_'+side).setDefaultLocked(False)
    model.finalizeConnections();model.printToXML(str(destination))
    return data,{'profile':profile,'source_sha256':digest(source),'probe_sha256':digest(destination),'fixed_coordinate_realization':fixed,'ranges_deg':changes,'unlocked':[p+'_'+s for p in spec['unlock'] for s in ['l','r']],
        'scope':'kinematic-domain diagnostic only; no muscle-force validity; source geometry and input fields unchanged'}
