"""Fixed-knot model-coordinate refinement of the interpolated body path.

Fitted native body positions define interpolation targets, gated by actual D
support. These targets are a path assumption, not additional measurements.
Every returned point is obtained by downstream model FK; native coordinates
and the fixed model are preserved.
"""
import numpy as np
from ..candidate_quality import exact_indices


def anchored_targets(native_times, native_points, output_times, support):
    native_times=np.asarray(native_times);output_times=np.asarray(output_times)
    native_points=np.asarray(native_points);support=np.asarray(support)
    if (native_points.ndim!=3 or native_points.shape[0]!=len(native_times)
            or native_points.shape[-1]!=3 or support.shape!=native_points.shape[:2]
            or support.dtype!=np.bool_ or len(native_times)<2
            or not np.isfinite(native_times).all() or not np.isfinite(output_times).all()
            or np.any(np.diff(native_times)<=0) or np.any(np.diff(output_times)<=0)
            or output_times[0]<native_times[0]-1e-9 or output_times[-1]>native_times[-1]+1e-9):
        raise ValueError('Ordered bounded clocks, body points and boolean observation support required')
    knots=exact_indices(native_times,output_times)
    left=np.clip(np.searchsorted(native_times,output_times,side='right')-1,0,len(native_times)-2)
    alpha=(output_times-native_times[left])/(native_times[left+1]-native_times[left])
    points=(1-alpha[:,None,None])*native_points[left]+alpha[:,None,None]*native_points[left+1]
    valid=support[left]&support[left+1]
    valid[knots]=support
    points[knots]=native_points
    points[~valid]=np.nan
    fixed=np.zeros(len(output_times),bool);fixed[knots]=True
    return points,fixed


def refine_path(model_path, labels, native, native_times, coordinates, output_times, motion, settings):
    from ..k4_fitting import model_fk
    from ..refinement import refine
    if settings.get('version')!='fixed_native_anchors_v1':
        raise ValueError('Unknown coordinated path policy')
    native_points,names=model_fk(model_path,labels,native,direct_couplers=True)
    if len(motion.times)!=len(native_times) or not np.allclose(
            motion.times-motion.times[0],native_times-native_times[0],atol=1e-8,rtol=0):
        raise ValueError('Native input clock mismatch')
    points,input_names,_=motion.legacy();index={n:i for i,n in enumerate(input_names)}
    priors=motion.legacy_prior_names()
    support=np.stack([np.isfinite(points[:,index[n]]).all(-1) & (n not in priors) for n in names],1)
    targets,fixed=anchored_targets(native_times,native_points,output_times,support)
    starts=np.array(coordinates,copy=True);knots=np.flatnonzero(fixed)
    starts[knots]=native  # Exact native charts, not just equivalent SO(3) charts.
    result,receipt=refine(model_path,labels,starts,targets,names,settings['refinement'],
                          times=output_times,fixed_frames=fixed)
    if not np.array_equal(result[knots],native):
        raise ValueError('Coordinated path changed a native coordinate')
    receipt.update(version=settings['version'],native_knots=len(knots),inserted_frames=int((~fixed).sum()),
        inserted_supported_frames=int((np.isfinite(targets).all((1,2))&~fixed).sum()),
        inserted_computed_frames=sum(r['status']=='computed' for i,r in enumerate(receipt['frames']) if not fixed[i]),
        observation_policy=settings['refinement'].get('observation_policy','complete_body_v1'),
        scope='Fixed native coordinates; joint-coupled inserted-frame fit on unchanged model; no R',
        targets='Linear fitted native body positions gated by D support; approximate interpolation path, not observed motion',
        fallback='Frames failing the declared observation policy retain original SO(3) interpolation; partial policy omits unsupported targets and requires both hip centres. Output coverage is unchanged.')
    return result,receipt
