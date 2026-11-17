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
    p.add_argument('--attachment-policy',choices=['registered_chest','input_fixed_offset'],default='registered_chest')
    p.add_argument('--method',required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve candidate directories')
    a.output.mkdir(parents=True)
    motion=load(a.input,a.method,config=ROOT/'config/evalkit_k4_20261117/input.json')
    with np.load(a.fit/'fit.npz') as z:data={k:z[k] for k in z.files}
    if 'body_positions' in data:
        body_names=list(data['body_names']);indices=[list(data['names']).index(n) for n in body_names]
        data['positions']=data['body_positions'].copy()
        data['positions_pre_root']=data['positions_pre_root'][:,indices]
        data['names']=np.asarray(body_names)
    shutil.copyfile(a.fit/'model.osim',a.output/'model.osim')
    labels=list(data['labels']);times=data['times'];body_names=list(data['names'])
    source_receipt=json.loads((a.fit/'receipt.json').read_text())
    if 'coordinates_pre_root' not in data:
        if source_receipt.get('root_policy','IK')!='IK':raise ValueError('Missing pre-root coordinates for compensated fit')
        data['coordinates_pre_root']=data['coords'].copy();data['positions_pre_root']=data['positions'].copy()
    definition,head=attach(motion,a.output/'model.osim',labels,data['coords'],times,attachment_policy=a.attachment_policy)
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
             'input_hashes':motion.provenance['inputs'],'attachment_policy':a.attachment_policy,'scope':'D-only head attachment; body coordinates unchanged; diagnostic, no automatic promotion',
             'head_code_sha256':hashlib.sha256((ROOT/'utils/motiogen/evalkit/k/head_neck.py').read_bytes()).hexdigest()}
    (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
