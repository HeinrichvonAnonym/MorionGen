"""Narrow runtime compatibility helpers for the pinned CHOIS dependencies."""
import sys
from types import ModuleType


def _validate_points(x,y):
    import torch
    if not torch.is_tensor(x) or not torch.is_tensor(y):
        raise TypeError('CHOIS BPS chamfer fallback accepts tensor point clouds only')
    if x.ndim!=3 or y.ndim!=3 or x.shape[0]!=y.shape[0] or x.shape[2]!=y.shape[2]:
        raise ValueError('point clouds must have shapes (N, P, D) with matching N and D')


class PureTorchChamferDistance:
    """BPS-compatible nearest-neighbour distances without PyTorch3D's C++ ops.

    The pinned ``bps_torch`` package consumes the four unreduced tensors returned
    by its vendored ``ChamferDistance`` implementation.  CHOIS only supplies dense
    tensor point clouds, so squared ``torch.cdist`` is an exact API-level fallback.
    """
    def __call__(self,x,y,x_lengths=None,y_lengths=None,x_normals=None,y_normals=None,
                 weights=None,batch_reduction='mean',point_reduction='mean'):
        import torch
        _validate_points(x,y)
        if any(value is not None for value in (x_normals,y_normals,weights)):
            raise ValueError('CHOIS BPS fallback does not support normals or weights')
        distances=torch.cdist(x,y,p=2).square()
        if y_lengths is not None:
            y_mask=torch.arange(y.shape[1],device=y.device)[None]>=y_lengths[:,None]
            distances=distances.masked_fill(y_mask[:,None,:],float('inf'))
        if x_lengths is not None:
            x_mask=torch.arange(x.shape[1],device=x.device)[None]>=x_lengths[:,None]
        else:
            x_mask=None
        cham_x,idx_x=distances.min(dim=2)
        cham_y,idx_y=distances.min(dim=1)
        if x_mask is not None:
            cham_x=cham_x.masked_fill(x_mask,0)
            reverse=distances.masked_fill(x_mask[:,:,None],float('inf'))
            cham_y,idx_y=reverse.min(dim=1)
        if y_lengths is not None:
            cham_y=cham_y.masked_fill(y_mask,0)
        return cham_x,cham_y,idx_x,idx_y


def install_chamfer_fallback():
    """Install the fallback only when the optional PyTorch3D extension is absent."""
    try:
        import pytorch3d._C  # noqa: F401
        return 'pytorch3d'
    except (ImportError,OSError):
        module=ModuleType('chamfer_distance')
        module.ChamferDistance=PureTorchChamferDistance
        module.__doc__='MorionGen CHOIS tensor-only chamfer compatibility module'
        sys.modules['chamfer_distance']=module
        # bps_torch imports its unused mesh-distance helper eagerly.  The CHOIS
        # dataset calls enc_points, never enc_mesh, but the eager import would
        # otherwise pull in the same unavailable PyTorch3D extension.
        def unavailable_mesh_distance(*_args,**_kwargs):
            raise RuntimeError('PyTorch3D mesh-distance extension is unavailable; CHOIS supports BPS point encoding only')
        loss=ModuleType('pytorch3d.loss')
        loss.__path__=[]
        point_mesh=ModuleType('pytorch3d.loss.point_mesh_distance')
        point_mesh.point_face_distance=unavailable_mesh_distance
        point_mesh.face_point_distance=unavailable_mesh_distance
        loss.point_mesh_distance=point_mesh
        sys.modules['pytorch3d.loss']=loss
        sys.modules['pytorch3d.loss.point_mesh_distance']=point_mesh
        return 'torch_cdist'
