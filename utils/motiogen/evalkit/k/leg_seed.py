"""D-only hip/knee initialization from observed thigh and shank directions."""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation
from .resampling.model import audit_model,_matrix,_represent


class LegDirectionSeeder:
    def __init__(self,model_path,labels,side):
        import opensim as osim
        from .constraints import couplers
        if side not in ('left','right'):raise ValueError('Explicit leg side required')
        self.s='l' if side=='left' else 'r';self.labels=list(labels)
        self.model=osim.Model(str(model_path));self.state=self.model.initSystem()
        self.bindings=couplers(self.model,self.state);self.cs=self.model.getCoordinateSet()
        self.hip=self.model.getJointSet().get('hip_'+self.s)
        self.chart=audit_model(model_path)['joints']['hip_'+self.s]
        if self.chart['classification']!='three_axis_so3':raise ValueError('Audited independent hip rotation required')
        self.hip_names=self.chart['rotation_coordinates'];self.cols=[self.labels.index(n) for n in self.hip_names]
        self.knee=self.cs.get('knee_angle_'+self.s);self.knee_col=self.labels.index(self.knee.getName())
        self.lower=np.array([self.cs.get(n).getRangeMin() for n in self.hip_names])
        self.upper=np.array([self.cs.get(n).getRangeMax() for n in self.hip_names])
        self.knee_table=None

    def points(self,row):
        import biomech
        from .constraints import synchronize
        biomech._set(self.model,self.state,self.labels,row);synchronize(self.state,self.bindings)
        self.model.realizePosition(self.state)
        return np.array([biomech._child_origin(self.model,self.state,n+'_'+self.s) for n in ['hip','walker_knee','ankle']])

    def seed(self,row,target):
        target=np.asarray(target,float);edges=np.diff(target,axis=0);lengths=np.linalg.norm(edges,axis=-1)
        if target.shape!=(3,3) or not np.isfinite(target).all() or np.any(lengths<1e-8):
            return None,{'status':'missing_or_degenerate_leg'}
        directions=edges/lengths[:,None]
        if np.linalg.norm(np.cross(*directions))<1e-4:return None,{'status':'weak_knee_plane'}
        wanted=float(np.arctan2(np.linalg.norm(np.cross(*directions)),np.dot(*directions)))
        guess=np.array(row,float,copy=True);guess[self.cols]=np.rad2deg(np.clip(0.,self.lower,self.upper))
        lo,hi=self.knee.getRangeMin(),self.knee.getRangeMax()
        def angle(q):
            guess[self.knee_col]=np.rad2deg(q);v=np.diff(self.points(guess),axis=0)
            return float(np.arctan2(np.linalg.norm(np.cross(*v)),np.dot(*v)))
        # The inter-segment angle depends on this fixed model's knee coordinate,
        # not on the world pose. Reuse the same coarse bracket across frames.
        if self.knee_table is None:
            grid=np.linspace(lo,hi,361)
            self.knee_table=(grid,np.array([angle(q) for q in grid]))
        grid,angles=self.knee_table;errors=(angles-wanted)**2;i=int(errors.argmin())
        left,right=grid[max(0,i-1)],grid[min(len(grid)-1,i+1)]
        result=minimize_scalar(lambda q:(angle(q)-wanted)**2,bounds=(left,right),method='bounded',options={'xatol':1e-10})
        knee=min([grid[i],result.x],key=lambda q:(angle(q)-wanted)**2)
        guess[self.knee_col]=np.rad2deg(knee);v=np.diff(self.points(guess),axis=0);v/=np.linalg.norm(v,axis=-1,keepdims=True)
        align,_=Rotation.align_vectors(directions,v)
        parent=_matrix(self.hip.getParentFrame(),self.state);child=_matrix(self.hip.getChildFrame(),self.state)
        desired=Rotation.from_matrix(parent.T@align.as_matrix()@child)
        hip,valid,_=_represent(Rotation.from_matrix(desired.as_matrix()[None]),self.chart['order'],np.array(self.chart['signs']),
                               np.deg2rad(np.asarray(row)[self.cols])[None],self.lower,self.upper)
        if not valid[0]:return None,{'status':'desired_hip_outside_declared_domain'}
        guess[self.cols]=np.rad2deg(hip[0]);actual=np.diff(self.points(guess),axis=0);actual/=np.linalg.norm(actual,axis=-1,keepdims=True)
        error=np.rad2deg(np.arctan2(np.linalg.norm(np.cross(actual,directions),axis=-1),np.sum(actual*directions,axis=-1)))
        return guess,{'status':'computed','knee_coordinate_deg':float(np.rad2deg(knee)),
                     'direction_error_deg':error.tolist(),'scope':'Observed thigh/shank frame and fixed model knee map; no foot target, R, or anatomical calibration'}
