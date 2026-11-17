"""Derive model-FK/root views from existing fits using D only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
import opensim as osim
osim.Logger.setLevelString('error')
from motiogen.evalkit.k.motion import load
from motiogen.evalkit.k.k4_fitting import model_fk
from motiogen.evalkit.k.root import compensate


def derive(fit,source,method,window,output,control=False):
    motion=load(source,method,config=ROOT/'config/evalkit_k4_20261117/input.json')
    with np.load(fit/'fit.npz') as z:data={k:z[k] for k in z.files}
    names=list(data['names']);labels=list(data['labels']);times=data['times'];model=fit/'model.osim'
    rebuilt,_=model_fk(model,labels,data['coords'],assemble=False)
    if np.max(np.linalg.norm(rebuilt-data['positions'],axis=-1))>1e-7:raise ValueError('Saved/model FK mismatch')
    coordinates=data.get('coordinates_pre_root',data['coords'])
    before,_=model_fk(model,labels,coordinates,assemble=False)
    _,root=compensate(model,times,labels,coordinates,motion)
    views={'before':before,'after':root['after']}
    if control:
        direct,_,q=model_fk(model,labels,data['coordinates_filtered_unassembled'],direct_couplers=True,return_coordinates=True)
        _,dr=compensate(model,times,labels,q,motion)
        views.update({'filtered_direct_before':direct,'filtered_direct_after':dr['after']})
    else:
        if np.max(abs(root['after']-data['positions']))>1e-7:raise ValueError('Root replay mismatch')
    receipt={'input_hashes':motion.provenance['inputs'],'fit_sha256':hashlib.sha256((fit/'fit.npz').read_bytes()).hexdigest(),
             'model_sha256':hashlib.sha256(model.read_bytes()).hexdigest(),'views':list(views),
             'scope':'D-derived root translation and constraint-consistent model FK; no R access'}
    for name,p in views.items():
        dest=output/name/window/method;dest.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest/'fk.npz',times=times,positions=p,names=names)
    dest=output/'receipts';dest.mkdir(parents=True,exist_ok=True)
    (dest/f'{window}_{method}.json').write_text(json.dumps(receipt,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('fit','input','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--method',required=True);p.add_argument('--window',required=True);p.add_argument('--control',action='store_true')
    a=p.parse_args();derive(a.fit,a.input,a.method,a.window,a.output,a.control)


if __name__=='__main__':main()
