"""Sparse whole-trajectory residuals in observable feature space, never Euler differences."""
import numpy as np
from scipy import sparse


class TrajectoryObjective:
    def __init__(self, feature, target, times, scales, rate_indices, rate_scale, rate_weight, lower, upper, step=1e-5,
                 violation_groups=(),violation_weight=0.,violation_margin=.8):
        self.feature=feature;self.target=np.asarray(target,float);self.times=np.asarray(times,float)
        self.scales=np.asarray(scales,float);self.indices=np.asarray(rate_indices,int)
        self.lower=np.asarray(lower,float);self.upper=np.asarray(upper,float);self.step=float(step)
        if self.target.ndim!=2 or self.target.shape[0]!=len(self.times) or len(self.times)<2:
            raise ValueError('Matching multi-frame target clock required')
        if not np.isfinite(self.times).all() or np.any(np.diff(self.times)<=0):raise ValueError('Strictly increasing physical clock required')
        if self.scales.shape!=(self.target.shape[1],) or not np.isfinite(self.scales).all() or np.any(self.scales<=0):raise ValueError('Positive feature scales required')
        if not np.isfinite([rate_scale,rate_weight,step]).all() or rate_scale<=0 or rate_weight<0 or step<=0:raise ValueError('Invalid rate or difference scale')
        if self.lower.shape!=self.upper.shape or np.any(self.lower>=self.upper):raise ValueError('Nondegenerate coordinate domain required')
        self.n,self.k=self.target.shape;self.q=len(self.lower);self.mask=np.isfinite(self.target)
        self.groups=[np.arange(start,end) for start,end in violation_groups]
        self.violation_weight=float(violation_weight);self.violation_margin=float(violation_margin)
        if not np.isfinite([violation_weight,violation_margin]).all() or violation_weight<0 or violation_margin<=0:
            raise ValueError('Finite nonnegative target weight and positive margin required')
        if any(not len(g) or g[0]<0 or g[-1]>=self.k for g in self.groups):raise ValueError('Invalid observable feature group')
        self.local_weight=self.mask/self.scales
        pair=self.mask[1:,self.indices]&self.mask[:-1,self.indices]
        self.rate_weight=pair*rate_weight/rate_scale
        dt=np.diff(self.times)
        difference=sparse.diags([-1/dt,1/dt],[0,1],shape=(self.n-1,self.n),format='csr')
        selected=(np.arange(self.n)[:,None]*self.k+self.indices).ravel()
        select=sparse.csr_matrix((np.ones(len(selected)),(np.arange(len(selected)),selected)),shape=(len(selected),self.n*self.k))
        self.transport=sparse.diags(self.rate_weight.ravel())@sparse.kron(difference,sparse.eye(len(self.indices)),format='csr')@select
        self.cached_x=None;self.cached_values=None

    def values(self,x):
        x=np.asarray(x,float).reshape(self.n,self.q)
        if self.cached_x is None or not np.array_equal(x,self.cached_x):
            values=np.asarray([self.feature(i,row) for i,row in enumerate(x)])
            if values.shape!=self.target.shape or not np.isfinite(values).all():raise ValueError('Model feature FK must be finite')
            self.cached_x=x.copy();self.cached_values=values
        return self.cached_values

    def residual(self,x):
        error=np.where(self.mask,self.values(x)-self.target,0.)
        terms=[(error*self.local_weight).ravel(),self.transport@error.ravel()]
        if self.violation_weight:
            normalized=error/self.scales
            terms.append(np.array([self.violation_weight*np.maximum(np.linalg.norm(normalized[:,g],axis=-1)-self.violation_margin,0)*self.mask[:,g].all(-1) for g in self.groups]).T.ravel())
        return np.concatenate(terms)

    def jacobian(self,x):
        x=np.asarray(x,float).reshape(self.n,self.q);blocks=[]
        for i,row in enumerate(x):
            columns=[]
            for j in range(self.q):
                left=row.copy();right=row.copy()
                left[j]=max(self.lower[j],row[j]-self.step);right[j]=min(self.upper[j],row[j]+self.step)
                columns.append((self.feature(i,right)-self.feature(i,left))/(right[j]-left[j]))
            blocks.append(np.stack(columns,axis=1))
        feature_jac=sparse.block_diag(blocks,format='csr')
        terms=[sparse.diags(self.local_weight.ravel())@feature_jac,self.transport@feature_jac]
        if self.violation_weight:
            error=np.where(self.mask,self.values(x)-self.target,0.)/self.scales
            rows=[];cols=[];data=[]
            for i in range(self.n):
                for j,g in enumerate(self.groups):
                    norm=np.linalg.norm(error[i,g])
                    if norm>self.violation_margin and self.mask[i,g].all():
                        rows.extend([i*len(self.groups)+j]*len(g));cols.extend(i*self.k+g)
                        data.extend(self.violation_weight*error[i,g]/norm/self.scales[g])
            operator=sparse.csr_matrix((data,(rows,cols)),shape=(self.n*len(self.groups),self.n*self.k))
            terms.append(operator@feature_jac)
        return sparse.vstack(terms,format='csr')
