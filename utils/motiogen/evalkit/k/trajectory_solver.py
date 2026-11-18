"""Bounded Gauss–Newton with actual-objective line search for native trajectories."""
import numpy as np
from scipy import sparse
from scipy.optimize import OptimizeResult,lsq_linear


def solve(objective,x0,settings):
    x=np.array(x0,float,copy=True)
    lower=np.tile(objective.lower,objective.n);upper=np.tile(objective.upper,objective.n)
    if x.shape!=lower.shape or np.any(x<lower) or np.any(x>upper):raise ValueError('Feasible trajectory seed required')
    budget=int(settings.get('max_nfev',20))
    if budget<1:raise ValueError('Positive objective evaluation budget required')
    residual=objective.residual(x);cost=float(residual@residual);nfev=1;njev=0;history=[]
    status=0;message='Objective evaluation budget exhausted';optimality=None
    while nfev<budget:
        jac=objective.jacobian(x);njev+=1;gradient=np.asarray(jac.T@residual).ravel()
        projected=gradient.copy()
        outward=((x-lower<1e-7)&(gradient>0))|((upper-x<1e-7)&(gradient<0))
        projected[outward]=0;optimality=float(abs(projected).max())
        if optimality<1e-6:status=1;message='Projected gradient tolerance reached';break
        norms=np.sqrt(np.asarray(jac.power(2).sum(0)).ravel());scale=1/np.maximum(norms,1.)
        linear=lsq_linear(jac@sparse.diags(scale),-residual,bounds=((lower-x)/scale,(upper-x)/scale),
            tol=1e-6,max_iter=int(settings.get('linear_max_iter',50)),lsmr_tol=1e-8,
            lsmr_maxiter=int(settings.get('linear_lsmr_maxiter',500)))
        step=scale*linear.x;descent=float(gradient@step)
        entry={'cost_before':cost,'linear_status':int(linear.status),'linear_iterations':int(linear.nit),
               'projected_gradient_max':optimality,'gradient_dot_step':descent,'trials':[]}
        accepted=False
        if np.isfinite(step).all() and descent<0:
            for attempt in range(12):
                if nfev>=budget:break
                alpha=2.**(-attempt);trial=np.clip(x+alpha*step,lower,upper)
                values=objective.residual(trial);nfev+=1;value=float(values@values)
                entry['trials'].append({'alpha':alpha,'cost':value})
                if np.isfinite(value) and value<cost and value<=cost+2e-4*alpha*descent:
                    previous=cost;entry['accepted_alpha']=alpha
                    entry['max_coordinate_step_rad']=float(abs(trial-x).max())
                    x,residual,cost=trial,values,value;accepted=True;break
        entry['cost_after']=cost;history.append(entry)
        if not accepted:
            if nfev<budget:status=-2;message='No bounded descent step passed actual-objective line search'
            break
        if previous-cost<=1e-8*max(1.,previous):status=2;message='Relative objective tolerance reached';break
    # Report the final iterate gradient rather than the preceding linearization.
    jac=objective.jacobian(x);njev+=1;gradient=np.asarray(jac.T@residual).ravel()
    gradient[((x-lower<1e-7)&(gradient>0))|((upper-x<1e-7)&(gradient<0))]=0
    optimality=float(abs(gradient).max())
    return OptimizeResult(x=x,fun=residual,cost=cost/2,nfev=nfev,njev=njev,status=status,
                          success=status>0,message=message,optimality=optimality,history=history)
