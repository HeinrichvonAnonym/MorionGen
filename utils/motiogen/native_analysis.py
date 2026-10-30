"""Additional auditable metrics; never overwrite original metric receipts."""
import numpy as np
import itertools
import math

def geometric_contact_persistence(distances,fps,threshold=.05):
    """Same-hand sampled geometric contact, not force or continuous grasp."""
    d=np.asarray(distances,float)
    if d.ndim!=2 or d.shape[1]!=2 or len(d)<2 or not np.isfinite(d).all() or (d<0).any() or not np.isfinite(fps) or fps<=0 or not np.isfinite(threshold) or threshold<=0:
        raise ValueError('Finite nonnegative [frame,2] distances and positive FPS required')
    contact=d<threshold
    eligible=int(contact[:-1].sum());retained=int((contact[:-1]&contact[1:]).sum())
    longest=0
    for hand in contact.T:
        edges=np.diff(np.r_[False,hand,False].astype(int))
        lengths=np.flatnonzero(edges==-1)-np.flatnonzero(edges==1)
        if len(lengths):longest=max(longest,int(lengths.max()))
    return {'eligible_hand_transitions':eligible,'retained_hand_transitions':retained,
            'retention_fraction':retained/eligible if eligible else None,
            'longest_observed_span_seconds':max(0,longest-1)/fps,
            'threshold_m':threshold,'frames':len(d),
            'scope':'Strict distance threshold on each native hand anchor separately. Retention denominator is contact-positive hand states except the last frame. Longest span is (consecutive frames minus one)/FPS, censored by the window. No claim of force, grasp, or unobserved between-frame contact.'}


def process_incidents(receipt):
    """Preserve failed attempts even after the final queue row succeeds."""
    found={}
    for attempt in [*receipt.get('attempts',[]),receipt]:
        if attempt.get('status') in (None,'success','pending','running','absent'):continue
        key=(attempt.get('run_dir'),attempt['status'])
        found[key]={k:attempt.get(k) for k in ['status','reason','run_dir','returncode','elapsed_seconds']}
    return list(found.values())


def boundary_kinematics(joints,fps,boundaries):
    """Recompute boundary quantities from native commit points, not stitched output."""
    xyz=np.asarray(joints);bounds=np.asarray(boundaries,dtype=int)
    bounds=bounds[(bounds>=2)&(bounds<len(xyz)-1)]
    if not len(bounds):return {}
    velocity_jump=(xyz[bounds+1]-xyz[bounds])*fps-(xyz[bounds-1]-xyz[bounds-2])*fps
    jerk=np.diff(xyz,n=3,axis=0)*fps**3
    return {'boundary_position_jump':float(np.linalg.norm(xyz[bounds]-xyz[bounds-1],axis=-1).mean()),
        'boundary_velocity_jump':float(np.linalg.norm(velocity_jump,axis=-1).mean()),
        'boundary_jerk_rms':float(np.sqrt(np.mean(jerk[bounds-2]**2)))}


def step_frames(fps,step_seconds=.1):
    if not np.isfinite(fps) or fps<=0 or step_seconds<=0:raise ValueError('Positive FPS and step required')
    return max(1,int(round(step_seconds*fps)))


def common_step_jerk(joints,fps,step_seconds=.1):
    """Third difference on a common time step, so 20 Hz and 30 Hz outputs compare.

    Same normalisation as the archived native metric (component RMS over all
    joints and axes); only the difference step changes from 1/FPS to ~0.1 s.
    """
    xyz=np.asarray(joints,float);k=step_frames(fps,step_seconds)
    if len(xyz)<=3*k:raise ValueError('Motion shorter than three common steps')
    d=(xyz[3*k:]-3*xyz[2*k:-k]+3*xyz[k:-2*k]-xyz[:-3*k])/(k/fps)**3
    return float(np.sqrt(np.mean(d**2)))


def common_step_boundary_jerk(joints,fps,boundaries,step_seconds=.1):
    """Common-step third difference whose middle interval straddles each commit.

    Samples s, s+k, s+2k, s+3k with s = b - ceil(1.5k); k = 1 reproduces the
    native boundary_jerk_rms window (b-2 .. b+1).
    """
    xyz=np.asarray(joints,float);k=step_frames(fps,step_seconds);start=int(math.ceil(1.5*k))
    bounds=np.asarray(boundaries,dtype=int);s=bounds-start
    s=s[(s>=0)&(s+3*k<len(xyz))]
    if not len(s):return None
    d=(xyz[s+3*k]-3*xyz[s+2*k]+3*xyz[s+k]-xyz[s])/(k/fps)**3
    return float(np.sqrt(np.mean(d**2)))


def bone_length_cv(joints,names,bones):
    """Within-clip coefficient of variation of each named bone length."""
    xyz=np.asarray(joints,float);index={str(n):i for i,n in enumerate(names)};result={}
    for a,b in bones:
        if a not in index or b not in index:continue
        length=np.linalg.norm(xyz[:,index[a]]-xyz[:,index[b]],axis=-1)
        result[f'{a}-{b}']={'mean_m':float(length.mean()),'cv':float(length.std()/length.mean())}
    return result


def threshold_success(errors,thresholds=(.02,.05,.1)):
    errors=np.asarray(errors,float)
    return {f'success_{round(t*100)}cm':float((errors<=t).mean()) for t in thresholds}


def facing_heading(joints,names,frame=0):
    """Planar facing from hip and shoulder axes; degrees from +X in the Z-up report frame."""
    xyz=np.asarray(joints,float);ix={str(n):i for i,n in enumerate(names)}
    across=xyz[frame,ix['right_hip']]-xyz[frame,ix['left_hip']]
    if 'right_shoulder' in ix and 'left_shoulder' in ix:across=across+xyz[frame,ix['right_shoulder']]-xyz[frame,ix['left_shoulder']]
    forward=np.cross([0.,0.,1.],across);forward[2]=0;forward/=np.linalg.norm(forward)
    return forward,float(np.degrees(np.arctan2(forward[1],forward[0])))


def cluster_t_interval(values,clusters,level=.95):
    """Equal-weight cluster means with a t interval on G-1 degrees of freedom."""
    from scipy import stats
    values=np.asarray(values,float);clusters=np.asarray(clusters)
    if len(values)!=len(clusters) or not np.isfinite(values).all():
        raise ValueError('Finite values and one cluster ID per paired result are required')
    keys=sorted(set(clusters.tolist()))
    means=np.array([values[clusters==k].mean() for k in keys])
    result={'clusters':len(keys),'paired_n':len(values),'mean':float(means.mean()) if len(means) else None,'ci':None,'cluster_means':means.tolist()}
    if len(means)>1:
        se=means.std(ddof=1)/math.sqrt(len(means));q=float(stats.t.ppf(.5+level/2,len(means)-1))
        result.update(se=float(se),df=len(means)-1,ci=[float(means.mean()-q*se),float(means.mean()+q*se)])
    return result


def sign_flip_p(cluster_means,exact_limit=24,draws=200000,seed=20261030):
    """Two-sided sign-flip test of a zero mean over cluster means.

    Exact enumeration of all 2^G sign patterns up to G = exact_limit, else a
    seeded Monte Carlo over patterns including the observed one.
    """
    x=np.asarray(cluster_means,float);g=len(x)
    if g<2 or not np.isfinite(x).all():raise ValueError('At least two finite cluster means required')
    observed=abs(x.sum());tolerance=1e-12*max(1.,np.abs(x).sum())
    if g<=exact_limit:
        bits=np.arange(g,dtype=np.int64);hits=0;total=1<<g;chunk=1<<min(g,20)
        for start in range(0,total,chunk):
            codes=np.arange(start,min(total,start+chunk),dtype=np.int64)
            signs=((codes[:,None]>>bits)&1)*2-1
            hits+=int((np.abs(signs@x)>=observed-tolerance).sum())
        return {'p':hits/total,'method':'exact','patterns':total}
    rng=np.random.default_rng(seed)
    signs=rng.integers(0,2,(draws,g))*2-1
    hits=int((np.abs(signs@x)>=observed-tolerance).sum())
    return {'p':(hits+1)/(draws+1),'method':'monte_carlo','patterns':draws}


def holm(pvalues):
    """Holm step-down adjusted p values in input order."""
    p=np.asarray(pvalues,float);order=np.argsort(p);m=len(p);adjusted=np.empty(m);running=0.
    for rank,index in enumerate(order):
        running=max(running,min(1.,(m-rank)*p[index]));adjusted[index]=running
    return adjusted.tolist()


def foot_sensitivity(motion,contract):
    names=list(motion['joint_names']); required=['left_foot','right_foot']
    if not all(name in names for name in required):
        return {'status':'unavailable','reason':'Both named skeletal foot joints are required'}
    feet=motion['joints'][:,[names.index(name) for name in required]]
    speed=np.linalg.norm(np.diff(feet[...,:2],axis=0),axis=-1)*float(motion['fps'])
    rows=[]
    for height in contract['foot_height_thresholds_m']:
        near=np.maximum(np.abs(feet[:-1,:,2]),np.abs(feet[1:,:,2]))<=height
        for velocity in contract['foot_speed_thresholds_m_s']:
            n=int(near.sum())
            rows.append({'height_m':height,'speed_m_s':velocity,'candidate_contact_samples':n,
                'all_samples':int(near.size),'candidate_fraction':float(near.mean()),
                'sliding_fraction_given_candidate':float((speed[near]>velocity).mean()) if n else None,
                'mean_speed_given_candidate_m_s':float(speed[near].mean()) if n else None})
    return {'status':'measured','proxy':True,'rows':rows,
            'floor_penetration_fraction':float((feet[:,:,2]<-contract['floor_penetration_threshold_m']).mean()),
            'definition':contract['foot_contact_basis']}


def physics_contact_sensitivity(joints, forces, names, fps, contract):
    """Native force-gated foot-body translation; do not invent contact points."""
    joints=np.asarray(joints);forces=np.asarray(forces);names=list(names)
    if joints.shape!=forces.shape or joints.ndim!=3 or joints.shape[-1]!=3:
        raise ValueError('Native positions and force vectors must share [frame,body,3]')
    if not np.isfinite(joints).all() or not np.isfinite(forces).all():
        raise ValueError('Non-finite native physical evidence')
    labels=['L_Ankle','L_Toe','R_Ankle','R_Toe']
    feet=[names.index(label) for label in labels]
    nonfeet=[i for i in range(len(names)) if i not in feet]
    speed=np.linalg.norm(np.diff(joints[:,feet,:2],axis=0),axis=-1)*fps
    support=np.maximum(forces[:,feet,2],0.)
    nonfoot_force=np.linalg.norm(forces[:,nonfeet],axis=-1)
    rows=[]
    for threshold in contract['physics_contact_force_thresholds_n']:
        contact=(support[:-1]>=threshold)&(support[1:]>=threshold)
        n=int(contact.sum())
        rows.append({'force_threshold_n':threshold,'contact_body_intervals':n,
            'all_body_intervals':int(contact.size),'contact_fraction':float(contact.mean()),
            'foot_body_translation_over_10cms_given_contact':float((speed[contact]>.1).mean()) if n else None,
            'foot_body_mean_speed_given_contact_m_s':float(speed[contact].mean()) if n else None,
            'nonfoot_contact_frame_fraction':float((nonfoot_force>=threshold).any(axis=1).mean())})
    return {'rows':rows,'body_names':labels,'definition':contract['physics_contact_basis'],
            'mean_total_vertical_support_n':float(support.sum(axis=1).mean())}


def clustered_interval(values,clusters,repetitions=2000,seed=20261024,resample_groups=None):
    values=np.asarray(values,float); clusters=np.asarray(clusters)
    if len(values)!=len(clusters) or not np.isfinite(values).all():
        raise ValueError('Finite values and one cluster ID per paired result are required')
    task_ids=sorted(set(clusters.tolist()))
    means=np.array([values[clusters==key].mean() for key in task_ids])
    parents=np.asarray(resample_groups) if resample_groups is not None else clusters
    if len(parents)!=len(values):raise ValueError('One resampling parent per observation is required')
    task_parents=[]
    for task in task_ids:
        keys=set(parents[clusters==task].tolist())
        if len(keys)!=1:raise ValueError('A task cannot span different resampling parents')
        task_parents.append(next(iter(keys)))
    task_parents=np.asarray(task_parents)
    parent_ids=sorted(set(task_parents.tolist()))
    result={'paired_n':len(values),'independent_groups':len(parent_ids),'task_groups':len(means),
            'mean_equal_group_weight':float(means.mean()) if len(means) else None,
            'median_pair':float(np.median(values)) if len(values) else None,
            'improved_fraction':float((values<0).mean()) if len(values) else None,
            'ci95':None}
    if len(parent_ids)>1:
        rng=np.random.default_rng(seed)
        sums=np.array([means[task_parents==p].sum() for p in parent_ids])
        counts=np.array([(task_parents==p).sum() for p in parent_ids])
        indices=rng.integers(0,len(parent_ids),(repetitions,len(parent_ids)))
        draws=sums[indices].sum(axis=1)/counts[indices].sum(axis=1)
        result['ci95']=np.quantile(draws,[.025,.975]).tolist()
        result['interval_degenerate']=bool(result['ci95'][0]==result['ci95'][1])
        if result['interval_degenerate']:
            result['interval_caveat']='Empirical cluster bootstrap has zero width; this does not establish zero population uncertainty or a zero failure probability.'
    else:
        result['interval_reason']='Fewer than two independent task groups; no population interval'
    result['weighting_scope']='Point estimate gives tasks equal weight; bootstrap resamples parent groups jointly, retaining all their tasks and task weights.'
    return result


def task_sensitivity(values, clusters, parents=None):
    """Describe scene variation and leave-one-parent-out influence, not extra CIs."""
    values=np.asarray(values,float);clusters=np.asarray(clusters)
    parents=np.asarray(parents) if parents is not None else clusters
    if len(values)!=len(clusters) or len(values)!=len(parents) or not np.isfinite(values).all():
        raise ValueError('Finite paired deltas and matching task/parent IDs required')
    rows=[]
    for task in sorted(set(clusters.tolist())):
        mask=clusters==task;group=set(parents[mask].tolist())
        if len(group)!=1:raise ValueError('A task must belong to one parent group')
        local=values[mask]
        rows.append({'task':task,'parent':next(iter(group)),'pairs':len(local),
                     'mean_delta':float(local.mean()),'min_pair_delta':float(local.min()),
                     'max_pair_delta':float(local.max())})
    parent_ids=sorted({r['parent'] for r in rows})
    omitted=[{'omitted_parent':parent,'equal_task_mean_delta':float(np.mean([r['mean_delta'] for r in rows if r['parent']!=parent]))}
             for parent in parent_ids] if len(parent_ids)>1 else []
    return {'task_effects':rows,'leave_one_parent_out':omitted,
            'task_means_negative':sum(r['mean_delta']<0 for r in rows),
            'task_means_positive':sum(r['mean_delta']>0 for r in rows),
            'scope':'Descriptive task spread and omission sensitivity, not simultaneous confidence intervals. Negative is not universally better.'}


def factorial_block(rows, factors):
    """Marginal differences and higher-order differences of differences.

    Every row has binary ``condition`` and numeric ``values``. Missing cells
    invalidate the block; a metric missing in any cell is excluded explicitly.
    Higher-order contrasts average over factors not named in that contrast.
    """
    cells={}
    for row in rows:
        cell=tuple(row['condition'][factor] for factor in factors)
        if any(value not in (0,1) for value in cell):raise ValueError('Factors must be binary-coded')
        if cell in cells:raise ValueError('Duplicate factorial cell')
        cells[cell]=row['values']
    if set(cells)!=set(itertools.product((0,1),repeat=len(factors))):
        return {'complete':False,'observed_cells':len(cells),'required_cells':2**len(factors),'contrasts':[]}
    metrics=set.intersection(*(set(values) for values in cells.values()))
    valid={metric for metric in metrics if all(isinstance(v[metric],(int,float)) and np.isfinite(v[metric]) for v in cells.values())}
    contrasts=[]
    for order in range(1,len(factors)+1):
        for selected in itertools.combinations(range(len(factors)),order):
            for metric in sorted(valid):
                # Randomized condition order must not turn cancelling terms
                # into a spurious signed effect. Compensated summation retains
                # genuine small contrasts without an arbitrary zero threshold.
                value=math.fsum(np.prod([2*cell[i]-1 for i in selected])*values[metric] for cell,values in cells.items())/(2**(len(factors)-order))
                contrasts.append({'factors':[factors[i] for i in selected],'order':order,'metric':metric,'delta':float(value)})
    simple=[]
    if len(factors)>1:
        for index,factor in enumerate(factors):
            for cell,off in cells.items():
                if cell[index]!=0:continue
                on_cell=list(cell);on_cell[index]=1
                on=cells[tuple(on_cell)]
                fixed={factors[i]:cell[i] for i in range(len(factors)) if i!=index}
                for metric in sorted(valid):
                    simple.append({'factor':factor,'fixed':fixed,'metric':metric,
                                   'delta':float(on[metric]-off[metric])})
    return {'complete':True,'observed_cells':len(cells),'required_cells':2**len(factors),
            'contrasts':contrasts,'simple_contrasts':simple,
            'excluded_metrics':sorted(set.union(*(set(v) for v in cells.values()))-valid)}


def reset_aware_arrival(joints,fps,goal,goal_frame,reset_frames,radius=.3,hold_seconds=.5):
    """Separate eventual arrival from arrival before native failure/reset.

    Retains the registered dense contract: round(fps * hold_seconds)
    consecutive samples, not a new wall-clock observation-span definition.
    The reset-marked sample is excluded from the pre-reset segment.
    """
    xyz=np.asarray(joints,float);goal=np.asarray(goal,float)
    resets=sorted(set(int(frame) for frame in reset_frames))
    if not np.isfinite(xyz).all() or not np.isfinite(goal).all() or fps<=0 or not 0<=goal_frame<len(xyz):
        raise ValueError('Finite positions, positive FPS and an in-range goal frame required')
    if any(frame<0 or frame>=len(xyz) for frame in resets):raise ValueError('Reset outside motion window')
    hold=max(1,round(fps*hold_seconds))
    hits=np.linalg.norm(xyz[:,0,:2]-goal[:2],axis=-1)<=radius
    def first(end):
        return next((i for i in range(goal_frame,end-hold+1) if hits[i:i+hold].all()),None)
    following=next((frame for frame in resets if frame>=goal_frame),len(xyz))
    eventual=first(len(xyz));before=first(following)
    return {'reset_frames':resets,'goal_frame':int(goal_frame),'hold_samples':hold,
            'first_eventual_hold_frame':eventual,'first_hold_before_next_reset_frame':before,
            'next_reset_frame':following if following<len(xyz) else None,
            'new_goal_success_before_next_reset':int(before is not None),
            'new_goal_success_reset_free_episode':int(eventual is not None and not resets),
            'scope':'Before-next-reset excludes a reset during the goal hold; earlier resets are retained separately. Reset-free-episode rejects any reset in the full saved 20-second run, even before the new goal.'}


def root_dwell(joints,fps,goal,radius=.3,speed_limit=.1,tail_seconds=2.):
    """Final-window station keeping, distinct from transient arrival/hold."""
    root=np.asarray(joints)[:,0,:2]
    tail=max(1,min(len(root)-1,round(tail_seconds*fps)))
    distance=np.linalg.norm(root[-tail:]-np.asarray(goal)[:2],axis=-1)
    speed=np.linalg.norm(np.diff(root,axis=0)[-tail:],axis=-1)*fps
    near=distance<=radius
    return {'tail_goal_occupancy_fraction':float(near.mean()),
            'tail_stationary_at_goal_fraction':float((near&(speed<=speed_limit)).mean()),
            'tail_root_mean_speed_m_s':float(speed.mean()),
            'tail_goal_max_error_m':float(distance.max())}


def common_history_consistency(errors,history_frames,horizon,joints,chunks,common_frames=32):
    """Same recent history span at every call, despite different buffer lengths.

    Saved errors concatenate [history frame, joint] for each call after the
    initial empty-history call. All preceding samples must be present exactly.
    """
    if min(history_frames,horizon,joints,chunks,common_frames)<=0 or common_frames>history_frames:
        raise ValueError('Positive dimensions and common span within history required')
    flat=np.asarray(errors,dtype=float).reshape(-1)
    lengths=[min(step*horizon,history_frames) for step in range(1,chunks)]
    if flat.size!=sum(lengths)*joints or not np.isfinite(flat).all() or (flat<0).any():
        raise ValueError('History receipt shape, finiteness or distance invalid')
    if not lengths:raise ValueError('At least one history-bearing call required')
    offset=0;selected=[]
    for length in lengths:
        chunk=flat[offset:offset+length*joints].reshape(length,joints)
        selected.append(chunk[-common_frames:].reshape(-1));offset+=length*joints
    recent=np.concatenate(selected)
    return {'mean_m':float(recent.mean()),'samples':int(recent.size),
            'history_frames_per_call':[min(n,common_frames) for n in lengths],
            'common_frames':common_frames,'calls':len(lengths)}


def wall_observation_recovery(positions,times,goal,request_seconds=5.,radius=.3,hold_seconds=.5):
    """Hold over consecutive native completion observations, no extrapolation."""
    positions=np.asarray(positions);times=np.asarray(times,float)
    if len(positions)!=len(times) or np.any(np.diff(times)<0):raise ValueError('Ordered paired wall observations required')
    beginning=None
    for position,timestamp in zip(positions,times):
        if timestamp<request_seconds:continue
        if np.linalg.norm(position[:2]-np.asarray(goal)[:2])>radius:
            beginning=None;continue
        if beginning is None:beginning=float(timestamp)
        if timestamp-beginning>=hold_seconds:
            return {'success':True,'first_entry_after_request_seconds':beginning-request_seconds,
                    'hold_confirmed_after_request_seconds':float(timestamp-request_seconds)}
    return {'success':False,'first_entry_after_request_seconds':None,'hold_confirmed_after_request_seconds':None}
