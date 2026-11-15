"""Reference frames from observed markers; no generated-motion imputer is used."""
import io
from functools import lru_cache
from pathlib import Path
import zipfile
import numpy as np
from .geometry import frame, valid_rotation


@lru_cache(maxsize=8)
def archive(path):
    return zipfile.ZipFile(path)


def andy(row, root):
    src = next(s for s in row['sources'] if s['archive']=='qualisys_csv.zip')
    text = archive(str(Path(root)/src['archive'])).read(src['member']).decode('utf-8-sig').splitlines()
    header = next(i for i,line in enumerate(text) if line.startswith('timestamp,'))
    names = [n[:-2] for n in text[header].split(',')[1::3]]
    a = np.genfromtxt(io.StringIO('\n'.join(text[header+2:])),delimiter=',')
    # Qualisys world is Z-up; K uses Y-up. Preserve a proper world rotation.
    z_to_y=np.array([[1.,0,0],[0,0,1],[0,-1,0]])
    return a[:,0]-a[0,0],names,(a[:,1:].reshape(len(a),-1,3)/1000)@z_to_y.T


def marker_frames(points,names,dataset):
    p = {n:points[:,i] for i,n in enumerate(names)}
    if dataset=='andy':
        head_names=['LFHD','RFHD','LBHD','RBHD']
        thorax_names=['C7','T10','CLAV','STRN','RBAK']
        right=(p['RFHD']+p['RBHD']-p['LFHD']-p['LBHD'])/2
        forward=(p['LFHD']+p['RFHD']-p['LBHD']-p['RBHD'])/2
        up=(p['C7']+p['CLAV']-p['T10']-p['STRN'])/2
        front=(p['CLAV']+p['STRN']-p['C7']-p['T10'])/2
    else:
        head_names=['head_l_ant','head_r_ant','head_l_post','head_r_post']
        thorax_names=['trunk_c7','trunk_t12','trunk_manu','trunk_xyph','trunk_t8']
        right=(p[head_names[1]]+p[head_names[3]]-p[head_names[0]]-p[head_names[2]])/2
        forward=(p[head_names[0]]+p[head_names[1]]-p[head_names[2]]-p[head_names[3]])/2
        up=(p[thorax_names[0]]+p[thorax_names[2]]-p[thorax_names[1]]-p[thorax_names[3]])/2
        front=(p[thorax_names[2]]+p[thorax_names[3]]-p[thorax_names[0]]-p[thorax_names[1]])/2
    # Right-handed segment basis: anatomical right, superior, posterior.
    # Head and thorax must share the same convention, including at neutral.
    return {'head':(frame(np.cross(right,forward),right),np.stack([p[n] for n in head_names],1)),
            'thorax':(frame(up,np.cross(front,up)),np.stack([p[n] for n in thorax_names],1))}


def template_fit(points, reference, calibration_frames):
    """Independent template calibration, then batched >=3 non-collinear point fits."""
    valid = np.isfinite(points).all(-1)
    complete = valid.all(1)&valid_rotation(reference)
    ids = np.flatnonzero(complete)[:calibration_frames]
    out = np.full((len(points),3,3),np.nan); residual=np.full(len(points),np.nan)
    if len(ids)<3:return out,residual,ids
    centred=points[ids]-points[ids].mean(1,keepdims=True)
    template=np.median(np.einsum('tji,tkj->tki',reference[ids],centred),0)
    signatures=(valid * (1<<np.arange(points.shape[1]))).sum(1)
    for code in np.unique(signatures):
        mask=(code & (1<<np.arange(points.shape[1])))!=0
        if mask.sum()<3:continue
        a=template[mask]-template[mask].mean(0)
        if np.linalg.svd(a,compute_uv=False)[1]<1e-5:continue
        ix=np.flatnonzero(signatures==code)
        b=points[ix][:,mask];b=b-b.mean(1,keepdims=True)
        h=np.einsum('ki,tkj->tij',a,b)
        u,s,vh=np.linalg.svd(h)
        d=np.broadcast_to(np.eye(3),(len(ix),3,3)).copy();d[:,2,2]=np.linalg.det(vh.swapaxes(1,2)@u.swapaxes(1,2))
        r=vh.swapaxes(1,2)@d@u.swapaxes(1,2)
        ok=s[:,1]>1e-10
        out[ix[ok]]=r[ok]
        residual[ix]=np.sqrt(np.mean(np.sum((np.einsum('tij,kj->tki',r,a)-b)**2,-1),1))
    return out,residual,ids
