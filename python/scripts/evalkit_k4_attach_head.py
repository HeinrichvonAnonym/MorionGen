"""D-only attached-chain diagnostic on an existing body fit; no body re-optimization."""
import argparse,json,hashlib,shutil
from pathlib import Path
import numpy as np
from motiogen.evalkit.k.config import ROOT
from motiogen.evalkit.k.motion import load
from motiogen.evalkit.k.head_neck import attach,combine


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('fit','input','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--method',required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve candidate directories')
    a.output.mkdir(parents=True)
    motion=load(a.input,a.method,config=ROOT/'config/evalkit_k4_20261117/input.json')
    with np.load(a.fit/'fit.npz') as z:data={k:z[k] for k in z.files}
    if 'body_positions' in data:raise ValueError('Head already attached')
    shutil.copyfile(a.fit/'model.osim',a.output/'model.osim')
    labels=list(data['labels']);times=data['times'];body_names=list(data['names'])
    definition,head=attach(motion,a.output/'model.osim',labels,data['coords'],times)
    _,before=attach(motion,a.output/'model.osim',labels,data['coordinates_pre_root'],times,fixed_definition=definition)
    data['body_positions']=data['positions'].copy();data['body_names']=data['names'].copy()
    data['positions'],names=combine(data['positions'],body_names,head)
    data['positions_pre_root'],_=combine(data['positions_pre_root'],body_names,before)
    data['names']=np.asarray(names);data['position_finite']=np.isfinite(data['positions']).all(-1)
    np.savez_compressed(a.output/'fit.npz',**data)
    np.savez_compressed(a.output/'head_neck.npz',**head)
    np.savez_compressed(a.output/'head_neck_pre_root.npz',**before)
    (a.output/'head_neck_model.json').write_text(json.dumps(definition,indent=2)+'\n')
    receipt={'body_fit':str(a.fit),'source_hashes':{n:hashlib.sha256((a.fit/n).read_bytes()).hexdigest() for n in ['fit.npz','model.osim','receipt.json']},
             'input_hashes':motion.provenance['inputs'],'scope':'D-only head attachment; body coordinates unchanged; diagnostic, no automatic promotion',
             'head_code_sha256':hashlib.sha256((ROOT/'utils/motiogen/evalkit/k/head_neck.py').read_bytes()).hexdigest()}
    (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
