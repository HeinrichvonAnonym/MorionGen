"""Isolated, upstream-API adapters; import heavy libraries only inside workers."""
import contextlib
import importlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from types import MethodType

import numpy as np

from .assets import find_asset
from .config import ROOT, asset_root, repo_root
from .motion import HUMANML_NAMES, Y_TO_Z, canonicalize, native_positions, save_motion
from .runtime import event, json_write


class MissingResources(RuntimeError):
    pass


class Unsupported(RuntimeError):
    pass


class ResourceGate(RuntimeError):
    pass


def require(cfg, baseline, name):
    path=find_asset(cfg,baseline,name)
    if path is None:
        raise MissingResources(f'{baseline}: missing {name}; run prepare and inspect official asset observations')
    return path


def sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Adapter:
    def __init__(self,cfg,name,run_dir,scene,seed):
        self.cfg,self.name,self.path,self.scene,self.seed=cfg,name,Path(run_dir),scene,seed
        self.spec=cfg['baselines'][name]
        self.events=self.path/'events.jsonl'
        self.stages=[]
        self.path.joinpath('raw').mkdir(exist_ok=True)

    @contextlib.contextmanager
    def timed(self,stage,**fields):
        sync(); start=time.perf_counter()
        event(self.events,stage+'_started',**fields)
        try:
            yield
        finally:
            sync(); duration=time.perf_counter()-start
            self.stages.append({'stage':stage,'seconds':duration,**fields})
            event(self.events,stage+'_finished',seconds=duration,**fields)

    def preflight(self):
        require(self.cfg,self.name,self.spec['model'])
        if self.scene['kind']=='object' and self.name!='chois':
            raise Unsupported('This baseline does not natively generate object/contact motion')
        if self.scene['kind']=='orientation' and self.name in ('motionlcm','omnicontrol'):
            raise Unsupported('Position-only conditioning does not implement wrist SO(3) commands')
        if self.scene['kind']=='online' and self.name not in ('motionlcm','ardy'):
            raise Unsupported('No online adapter for this baseline')
        if self.spec.get('variant','native') not in ('native','cached'):
            raise Unsupported('Precision/offload variants require a separate compatibility and host-memory review; this adapter currently supports native/cached text only')

    def seed_all(self,seed):
        import random
        import torch
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    def save(self,joints,names,rotations=None,filename='motion.npz',**optional):
        xyz,rot=canonicalize(joints,self.spec['axes'],rotations,units=self.spec.get('position_units','m'))
        native=getattr(self,'native_names',names)
        if len(names)==len(getattr(self,'output_native_names',[])):
            native=self.output_native_names
        skeleton=self.spec['skeleton']
        if hasattr(self,'model') and hasattr(self.model,'skeleton'):
            target=getattr(self.model,'output_skeleton',self.model.skeleton)
            if len(names)==len(target.bone_order_names_with_parents):
                skeleton=target.name
        metadata={'baseline':self.name,'model':self.spec['model'],'units':'m','axes':'right-handed Z-up',
                  'native_axes':self.spec['axes'],'skeleton':skeleton,'seed':self.seed,
                  'scenario':self.scene['name'],'native_joint_names':native,
                  'postprocess':bool(getattr(self,'postprocess',False)),
                  'representation_conversion':'world basis rotation; no retargeting'}
        save_motion(self.path/filename,xyz,names,self.fps,metadata,rot,**optional)
        return self.path/filename

    def timing_summary(self):
        result={}
        for stage in sorted({x['stage'] for x in self.stages}):
            rows=[x for x in self.stages if x['stage']==stage and not x.get('warmup',False)]
            if any(x.get('benchmark',False) for x in rows):
                rows=[x for x in rows if x.get('benchmark',False)]
            vals=[x['seconds'] for x in rows]
            if vals:
                result[stage]={'n':len(vals),'mean_seconds':float(np.mean(vals)),'p50_seconds':float(np.percentile(vals,50)),
                               'p95_seconds':float(np.percentile(vals,95)),'samples_seconds':vals}
        json_write(self.path/'timing.json',result)
        return result

    def measure_repetitions(self,generate):
        for i in range(self.cfg['timing']['warmup']):
            with self.timed('sample_end_to_end',warmup=True):
                generate(warmup=True)
        for i in range(self.cfg['timing']['repetitions']):
            self.seed_all(self.seed+i)
            with self.timed('sample_end_to_end',benchmark=True):
                generate(benchmark=True)


class MotionLCM(Adapter):
    def preflight(self):
        super().preflight()
        for name in ('Mean.npy','Std.npy','Mean_raw.npy','Std_raw.npy','finest.tar'):
            require(self.cfg,self.name,name)
        text=asset_root(self.cfg)/'text/sentence-t5-large'
        if not (text/'config.json').exists():
            raise MissingResources('sentence-t5-large text encoder not prepared')
        if not (text/'source_receipt.json').exists():
            raise MissingResources('sentence-t5-large download incomplete: pinned receipt missing')
        from .resource_review import inference_review
        if inference_review(self).get('scope')!='full MotionLCM inference':
            raise ResourceGate('Only the VAE diagnostic has a reviewed bound; full text/ControlNet loading remains unverified')

    def load(self):
        import torch
        from omegaconf import OmegaConf
        from mld.config import get_module_config
        from mld.models.modeltype.mld import MLD
        from mld.data.humanml.scripts.motion_process import recover_from_ric
        repo=repo_root(self.cfg,self.name)
        cfg=OmegaConf.load(repo/'configs/motionlcm_control_s.yaml' if self.scene['kind']!='official' else repo/'configs/motionlcm_t2m.yaml')
        cfg.model=get_module_config(cfg.model,cfg.model.target,str(repo/'configs'))
        cfg.output_dir=str(self.path)
        cfg.DATASET.NFEATS=263; cfg.DATASET.NJOINTS=22
        cfg.METRIC.TYPE=[]
        cfg.model.t5_path=str(asset_root(self.cfg)/'text/sentence-t5-large')
        evaluator=require(self.cfg,self.name,'finest.tar')
        cfg.model.t2m_path=str(evaluator.parents[3])
        for stage in ('TRAIN','VAL','TEST'):
            node=getattr(cfg,stage); node.BATCH_SIZE=1; node.NUM_WORKERS=0; node.PERSISTENT_WORKERS=False
        checkpoint=require(self.cfg,self.name,'motionlcm_humanml.ckpt' if self.scene['kind']=='official' else self.spec['model'])
        with self.timed('checkpoint_load'):
            state=torch.load(checkpoint,map_location='cpu')['state_dict']
        key='denoiser.time_embedding.cond_proj.weight'
        if key in state:
            cfg.model.denoiser.params.time_cond_proj_dim=state[key].shape[1]
        cfg.model.is_controlnet='controlnet.controlnet_cond_embedding.0.weight' in state
        mean=np.load(require(self.cfg,self.name,'Mean.npy'))
        std=np.load(require(self.cfg,self.name,'Std.npy'))
        raw_mean=np.load(require(self.cfg,self.name,'Mean_raw.npy'))
        raw_std=np.load(require(self.cfg,self.name,'Std_raw.npy'))
        def feats2joints(feats):
            return recover_from_ric(feats*torch.tensor(std).to(feats)+torch.tensor(mean).to(feats),22)
        data=SimpleNamespace(name='humanml3d',is_mm=False,feats2joints=feats2joints)
        with self.timed('model_load'):
            self.model=MLD(cfg,data)
            self.model.load_state_dict(state,strict=True)
            self.model.eval().requires_grad_(False).to('cuda')
        del state
        self.model_cfg=cfg; self.mean,self.std=mean,std
        self.raw_mean,self.raw_std=raw_mean.reshape(22,3),raw_std.reshape(22,3)
        self.fps=20; self.names=HUMANML_NAMES
        self.cache={}
        self.text_forward=self.model.text_encoder.forward
        def text_forward(texts):
            key=tuple(texts)
            use_cache=self.spec.get('variant','native')=='cached'
            with self.timed('text_encoding',cache_hit=use_cache and key in self.cache,**getattr(self,'call_flags',{})):
                if not use_cache:
                    return self.text_forward(texts)
                if key not in self.cache:
                    self.cache[key]=self.text_forward(texts).detach()
                return self.cache[key]
        self.model.text_encoder.forward=text_forward
        json_write(self.path/'upstream_config.json',OmegaConf.to_container(cfg,resolve=True))
        json_write(self.path/'parameters.json',{name:sum(p.numel() for p in module.parameters()) for name,module in self.model.named_children()})

    def batch(self,prompt,duration,constraints=None):
        import torch
        n=min(196,max(40,int(round(duration*self.fps))))
        batch={'text':[prompt],'length':[n]}
        if self.model.is_controlnet:
            hint=np.zeros((1,n,22,3),np.float32); mask=np.zeros_like(hint)
            for c in constraints or []:
                frame=int(round(c['time_seconds']*self.fps)); joint=self.names.index(c['joint'])
                if not 0<=frame<n:
                    raise ValueError('Constraint frame outside generated clip')
                pos=native_positions(c['position_m'],self.spec['axes'])
                hint[0,frame,joint]=(pos-self.raw_mean[joint])/self.raw_std[joint]
                mask[0,frame,joint]=1
            batch['hint']=torch.from_numpy(hint).cuda(); batch['hint_mask']=torch.from_numpy(mask).cuda()
        return batch

    def generate(self,prompt=None,duration=None,constraints=None,warmup=False,cold=False,benchmark=False):
        import torch
        self.call_flags={'warmup':warmup,'cold':cold,'benchmark':benchmark}
        batch=self.batch(prompt or self.scene['prompt'],duration or self.scene['duration_seconds'],self.scene.get('constraints',[]) if constraints is None else constraints)
        with self.timed('generation',**self.call_flags),torch.no_grad():
            joints=self.model(batch)[0][0].detach().cpu().numpy()
        return joints

    def run(self,benchmark=False):
        self.seed_all(self.seed)
        if self.scene['kind']=='online':
            return self.rolling()
        joints=self.generate(cold=True)
        np.save(self.path/'raw/joints.npy',joints)
        path=self.save(joints,self.names)
        if benchmark:
            self.measure_repetitions(self.generate)
        return path

    def rolling(self):
        total=int(self.scene['duration_seconds']*self.fps)
        commit=max(1,int(self.scene['commit_seconds']*self.fps)); overlap=max(1,int(self.scene['overlap_seconds']*self.fps))
        raw=[]; corrected=[]; boundaries=[]; errors=[]; prompt=self.scene['prompt']; applied=set(); previous_prediction=None
        for start in range(0,total,commit):
            for idx,update in enumerate(self.scene.get('updates',[])):
                if start/self.fps>=update['time_seconds'] and idx not in applied:
                    prompt=update['prompt']; applied.add(idx); event(self.events,'prompt_update',motion_seconds=start/self.fps,prompt=prompt)
            joints=self.generate(prompt,self.scene['horizon_seconds'],constraints=[])
            if previous_prediction is not None:
                count=min(len(previous_prediction)-commit,len(joints),overlap)
                if count>0:
                    # Two predictions for exactly the same future global frames, before any stitching.
                    errors.extend(np.linalg.norm(joints[:count]-previous_prediction[commit:commit+count],axis=-1).ravel().tolist())
            previous_prediction=joints.copy()
            take=min(commit,total-start)
            segment=joints[:take].copy(); raw.append(segment.copy())
            if corrected:
                boundaries.append(start)
                previous=np.concatenate(corrected)
                # External alignment/blending only: model receives no history condition.
                aligned=segment+(previous[-1,0]-segment[0,0])
                count=min(overlap,take,len(previous))
                alpha=np.linspace(0,1,count+1)[1:,None,None]
                aligned[:count]=(1-alpha)*previous[-1:]+alpha*aligned[:count]
                segment=aligned
            corrected.append(segment)
            event(self.events,'chunk_committed',frame=start,frames=take,external_stitch=True)
        self.save(np.concatenate(raw),self.names,filename='motion_pre_stitch.npz',chunk_boundaries=boundaries,overlap_errors_m=errors)
        np.save(self.path/'raw/rolling_clips.npy',np.concatenate(raw))
        return self.save(np.concatenate(corrected),self.names,chunk_boundaries=boundaries,overlap_errors_m=errors)


class MotionLCMVAE(Adapter):
    """Official VAE core reconstruction, separate from text/control capability claims."""
    def preflight(self):
        self.spec=dict(self.spec,model='vae_humanml.ckpt')
        super().preflight()
        for file in ('Mean.npy','Std.npy','new_joint_vecs'):
            require(self.cfg,self.name,file)
        from .resource_review import inference_review
        entry=inference_review(self)
        if entry.get('scope')!='VAE reconstruction only':
            raise ResourceGate('Wrong memory review scope for the reconstruction diagnostic')

    def load(self):
        import torch
        from omegaconf import OmegaConf
        from mld.config import get_module_config,instantiate_from_config
        from mld.data.humanml.scripts.motion_process import recover_from_ric
        repo=repo_root(self.cfg,self.name); cfg=OmegaConf.load(repo/'configs/vae.yaml')
        cfg.DATASET.NFEATS=263; cfg.DATASET.NJOINTS=22
        cfg.model=get_module_config(cfg.model,cfg.model.target,str(repo/'configs'))
        with self.timed('checkpoint_load'):
            state=torch.load(require(self.cfg,self.name,'vae_humanml.ckpt'),map_location='cpu')['state_dict']
        with self.timed('model_load'):
            self.model=instantiate_from_config(cfg.model.motion_vae)
            self.model.load_state_dict({k.removeprefix('vae.'):v for k,v in state.items() if k.startswith('vae.')},strict=True)
            self.model.eval().requires_grad_(False).cuda()
        del state
        self.mean=np.load(require(self.cfg,self.name,'Mean.npy')).astype(np.float32); self.std=np.load(require(self.cfg,self.name,'Std.npy')).astype(np.float32)
        file=sorted(require(self.cfg,self.name,'new_joint_vecs').glob('*.npy'))[0]
        features=np.load(file)[:min(196,round(self.scene['duration_seconds']*20))].astype(np.float32)
        self.reference=features; self.features=torch.from_numpy(((features-self.mean)/self.std).astype(np.float32))[None].cuda()
        self.mask=torch.ones((1,len(features)),dtype=torch.bool,device='cuda'); self.fps=20; self.names=HUMANML_NAMES; self.recover=recover_from_ric
        json_write(self.path/'upstream_config.json',OmegaConf.to_container(cfg,resolve=True))
        json_write(self.path/'diagnostic.json',{'source_motion':str(file),'scope':'VAE core encode/decode as official configs/vae.yaml; no text encoder/evaluator instantiated','parameters':sum(p.numel() for p in self.model.parameters())})

    def generate(self,warmup=False,cold=False,benchmark=False):
        import torch
        with self.timed('generation',warmup=warmup,cold=cold,benchmark=benchmark),torch.no_grad():
            latent,_=self.model.encode(self.features,self.mask)
            output=self.model.decode(latent,self.mask).detach().cpu()[0]
        if cold:
            np.save(self.path/'raw/reconstructed_features.npy',output.numpy())
            np.save(self.path/'raw/reference_features.npy',self.reference)
        with self.timed('postprocess',warmup=warmup,cold=cold,benchmark=benchmark):
            return self.recover(output*torch.tensor(self.std)+torch.tensor(self.mean),22).numpy()

    def run(self,benchmark=False):
        import torch
        self.seed_all(self.seed)
        with self.timed('sample_end_to_end',cold=True):
            joints=self.generate(cold=True)
        reference=self.recover(torch.from_numpy(self.reference),22).numpy()
        self.save(reference,self.names,filename='reference.npz')
        json_write(self.path/'reconstruction_metrics.json',{'joint_reconstruction_error_m':float(np.linalg.norm(joints-reference,axis=-1).mean()),
                'n':len(joints)*22,'scope':'stochastic VAE reconstruction, not hand control'})
        path=self.save(joints,self.names)
        if benchmark:
            self.measure_repetitions(self.generate)
        return path


class OmniControl(Adapter):
    def preflight(self):
        super().preflight()
        for name in ('Mean.npy','Std.npy','Mean_raw.npy','Std_raw.npy','SMPL_NEUTRAL.pkl'):
            require(self.cfg,self.name,name)
        require(self.cfg,self.name,'J_regressor_extra.npy')
        require(self.cfg,self.name,'ViT-B-32.pt')
        from .resource_review import inference_review
        inference_review(self)

    def load(self):
        import torch
        from argparse import Namespace
        from utils.model_util import create_model_and_diffusion,load_model_wo_clip
        from model.cfg_sampler import ClassifierFreeSampleModel
        from data_loaders.humanml.scripts.motion_process import recover_from_ric
        from model import rotation2xyz
        checkpoint=require(self.cfg,self.name,self.spec['model'])
        from utils.parser_util import add_base_options,add_data_options,add_model_options,add_diffusion_options
        from argparse import ArgumentParser
        parser=ArgumentParser()
        for add in (add_base_options,add_data_options,add_model_options,add_diffusion_options):
            add(parser)
        args=vars(parser.parse_args([]))
        if (checkpoint.parent/'args.json').exists():
            args.update(json.loads((checkpoint.parent/'args.json').read_text()))
        args.update(batch_size=1,device=0,cond_mode='both_text_spatial')
        self.args=Namespace(**args)
        # SMPL constructor is needed by upstream even for XYZ recovery. Resolve the asset path without editing baseline source.
        from model import smpl
        smpl.SMPL.__init__.__defaults__=(str(require(self.cfg,self.name,'SMPL_NEUTRAL.pkl').parent),)
        regressor=require(self.cfg,self.name,'J_regressor_extra.npy')
        smpl.JOINT_REGRESSOR_TRAIN_EXTRA=str(regressor)
        import clip
        clip_loader=clip.load
        def local_clip(name,*args,**kwargs):
            return clip_loader(str(require(self.cfg,self.name,'ViT-B-32.pt')),*args,**kwargs)
        clip.load=local_clip
        with self.timed('model_load'):
            # OmniControl hard-codes ./dataset paths inside GaussianDiffusion.
            # Resolve only its four normalization reads to receipt-backed assets
            # while keeping the pinned upstream checkout untouched.
            from diffusion import gaussian_diffusion as omni_gaussian
            original_pjoin=omni_gaussian.pjoin
            normalization={name:require(self.cfg,self.name,name) for name in
                           ('Mean.npy','Std.npy','Mean_raw.npy','Std_raw.npy')}
            def asset_pjoin(_directory,filename):
                return str(normalization[filename]) if filename in normalization else original_pjoin(_directory,filename)
            omni_gaussian.pjoin=asset_pjoin
            try:
                self.model,self.diffusion=create_model_and_diffusion(self.args,SimpleNamespace(dataset=SimpleNamespace(num_actions=1)))
            finally:
                omni_gaussian.pjoin=original_pjoin
            state=torch.load(checkpoint,map_location='cpu'); load_model_wo_clip(self.model,state); del state
            self.model=ClassifierFreeSampleModel(self.model).to('cuda').eval()
        self.mean=np.load(require(self.cfg,self.name,'Mean.npy')).astype(np.float32); self.std=np.load(require(self.cfg,self.name,'Std.npy')).astype(np.float32)
        self.raw_mean=np.load(require(self.cfg,self.name,'Mean_raw.npy')).reshape(22,3)
        self.raw_std=np.load(require(self.cfg,self.name,'Std_raw.npy')).reshape(22,3)
        self.recover=recover_from_ric; self.fps=20; self.names=HUMANML_NAMES
        self.cache={}; original=self.model.model.encode_text
        def encode_text(texts):
            key=tuple(texts); use_cache=self.spec.get('variant','native')=='cached'
            with self.timed('text_encoding',cache_hit=use_cache and key in self.cache,**getattr(self,'call_flags',{})):
                if not use_cache:
                    return original(texts)
                if key not in self.cache:
                    self.cache[key]=original(texts).detach()
                return self.cache[key]
        self.model.model.encode_text=encode_text
        json_write(self.path/'upstream_config.json',args)

    def generate(self,warmup=False,cold=False,benchmark=False):
        import torch
        self.call_flags={'warmup':warmup,'cold':cold,'benchmark':benchmark}
        n=196; hints=np.zeros((1,n,22,3),np.float32)
        for c in self.scene.get('constraints',[]):
            frame=int(round(c['time_seconds']*self.fps)); j=self.names.index(c['joint'])
            hints[0,frame,j]=(native_positions(c['position_m'],self.spec['axes'])-self.raw_mean[j])/self.raw_std[j]
        y={'text':[self.scene['prompt']],'lengths':torch.tensor([n],device='cuda'),'mask':torch.ones(1,1,1,n,dtype=torch.bool,device='cuda'),'scale':torch.tensor([2.5],device='cuda')}
        if self.scene.get('constraints'):
            y['hint']=torch.from_numpy(hints.reshape(1,n,66)).cuda()
        # Spatial guidance computes derivatives w.r.t. samples; do not globally disable autograd.
        with self.timed('generation',**self.call_flags):
            sample=self.diffusion.p_sample_loop(self.model,(1,self.model.njoints,self.model.nfeats,n),clip_denoised=False,model_kwargs={'y':y},progress=False,skip_timesteps=0,init_image=None,noise=None,const_noise=False)
        feats=sample[:,:263].detach().cpu().permute(0,2,3,1)
        if cold:
            np.save(self.path/'raw/features.npy',feats.numpy())
        joints=self.recover(feats*torch.tensor(self.std)+torch.tensor(self.mean),22).reshape(n,22,3).numpy()
        return joints

    def run(self,benchmark=False):
        self.seed_all(self.seed)
        joints=self.generate(cold=True)
        if benchmark:
            self.measure_repetitions(self.generate)
        return self.save(joints,self.names)


ALIASES={'Hips':'pelvis','Pelvis':'pelvis','Neck':'neck','Head':'head','RightArm':'right_shoulder','LeftArm':'left_shoulder',
         'RightForeArm':'right_elbow','LeftForeArm':'left_elbow','RightHand':'right_wrist','LeftHand':'left_wrist',
         'RightUpLeg':'right_hip','LeftUpLeg':'left_hip','RightLeg':'right_knee','LeftLeg':'left_knee',
         'RightFoot':'right_ankle','LeftFoot':'left_ankle','RightToeBase':'right_foot','LeftToeBase':'left_foot'}
ALIASES.update({'Neck1':'neck','LeftUpperLeg':'left_hip','RightUpperLeg':'right_hip','Spine1':'spine1','Spine2':'spine2','Chest':'spine3'})


def joint_aliases(skeleton_name):
    result=dict(ALIASES)
    if skeleton_name in ('somaskel30','somaskel77'):
        result.update({'LeftLeg':'left_hip','RightLeg':'right_hip','LeftShin':'left_knee','RightShin':'right_knee'})
    return result


class SparseJointConstraint:
    """Upstream condition-dictionary protocol, without implicit root or hand-tip commands."""
    def __init__(self,frame,joint,position,rotation=None,smooth_root_2d=None):
        self.frame,self.joint,self.position,self.rotation=frame,joint,position,rotation
        self.smooth_root_2d=smooth_root_2d

    def update_constraints(self,data,index):
        import torch
        indices=torch.tensor([[self.frame,self.joint]],device=self.position.device)
        data['global_joints_positions'].append(self.position[None])
        index['global_joints_positions'].append(indices)
        if self.smooth_root_2d is not None:
            data['smooth_root_2d'].append(self.smooth_root_2d[None])
            index['smooth_root_2d'].append(torch.tensor([self.frame],device=self.position.device))
        if self.rotation is not None:
            data['global_joints_rots'].append(self.rotation[None])
            index['global_joints_rots'].append(indices)

    def crop_move(self,start,end):
        if start<=self.frame<end:
            return SparseJointConstraint(self.frame-start,self.joint,self.position,self.rotation,self.smooth_root_2d)
        return EmptyConstraint()


class EmptyConstraint:
    """Window-local no-op matching the NVIDIA constraint protocol."""
    def update_constraints(self,data,index):
        return None

    def crop_move(self,start,end):
        return self


class TimedTextEncoder:
    def __init__(self,original,adapter):
        self.original,self.adapter,self.cache=original,adapter,{}

    def __getattr__(self,key):
        return getattr(self.original,key)

    def __call__(self,texts):
        key=tuple(texts) if not isinstance(texts,str) else (texts,)
        use_cache=self.adapter.spec.get('variant','native')=='cached'
        with self.adapter.timed('text_encoding',cache_hit=use_cache and key in self.cache,**getattr(self.adapter,'call_flags',{})):
            if not use_cache:
                return self.original(texts)
            if key not in self.cache:
                self.cache[key]=self.original(texts)
            return self.cache[key]


class Nvidia(Adapter):
    def preflight(self):
        super().preflight()
        require(self.cfg,self.name,'source_receipt.json')
        # No implicit 8B download/model allocation. Verify access in the prepare phase first.
        cache=asset_root(self.cfg)/'text/llm2vec'
        if not (cache/'local_base_receipt.json').exists():
            raise MissingResources('Official LLM2Vec/Llama text weights missing. See text_access.json; log in locally with hf auth login when access is required.')
        policy=json.loads((cache/'local_base_receipt.json').read_text())
        expected=json.loads((ROOT/'config/language_models.json').read_text())
        if policy.get('expected_base')!=expected['expected_base'] or policy.get('expected_config')!=expected['expected_config']:
            raise MissingResources('Local Llama receipt predates or disagrees with the pinned Meta-Llama-3-8B-Instruct identity; register it again')
        for item in policy.get('files',[]):
            source=Path(item['path'])
            if not source.is_file():
                raise MissingResources(f'User-supplied Llama shard disappeared: {source.name}')
            stat=source.stat()
            if stat.st_size!=item['bytes'] or stat.st_mtime_ns!=item['mtime_ns']:
                raise MissingResources(f'User-supplied Llama shard changed after receipt: {source.name}; register it again')
        for repo in (policy['mntp_repo'],policy['supervised_repo']):
            if not (cache/repo/'source_receipt.json').is_file():
                raise MissingResources(f'Official LLM2Vec adapter download incomplete: {repo}')
        from .resource_review import inference_review
        inference_review(self)

    def load(self):
        import importlib
        module=importlib.import_module(self.name+'.model.load_model')
        os.environ['TEXT_ENCODER_MODE']='local'
        os.environ['TEXT_ENCODERS_DIR']=str(asset_root(self.cfg)/'text/llm2vec')
        os.environ['CHECKPOINT_DIR']=str(asset_root(self.cfg)/self.name)
        from .local_llm import precompute_encoder
        prompts=[self.scene['prompt'],*[update['prompt'] for update in self.scene.get('updates',[])]]
        with self.timed('text_model_load_and_encode',prompts=len(set(prompts))):
            text_encoder=precompute_encoder(self.cfg,self.name,prompts)
        with self.timed('model_load'):
            self.model=module.load_model(self.spec['model'],device='cuda',text_encoder=text_encoder,checkpoints_dir=str(asset_root(self.cfg)/self.name)) if self.name=='ardy' else module.load_model(self.spec['model'],device='cuda',text_encoder=text_encoder)
        self.fps=self.model.motion_rep.fps if self.name=='ardy' else self.model.fps
        self.native_names=[x[0] for x in self.model.skeleton.bone_order_names_with_parents]
        aliases=joint_aliases(self.model.skeleton.name)
        self.names=[aliases.get(x,x) for x in self.native_names]
        self.output_native_names=[x[0] for x in getattr(self.model,'output_skeleton',self.model.skeleton).bone_order_names_with_parents]
        self.output_names=[aliases.get(x,x) for x in self.output_native_names]
        self.model.text_encoder=TimedTextEncoder(self.model.text_encoder,self)

    def constraint_list(self):
        import torch
        result=[]; skeleton=self.model.skeleton
        specifications=self.scene.get('constraints',[])
        if self.name=='kimodo' and specifications:
            if len(specifications)!=1:
                raise Unsupported('Kimodo goal-only smoke currently supports one feasible end-effector keyframe')
            c=specifications[0]
            idx=self.names.index(c['joint']); native=self.native_names[idx]
            classes={'RightHand':'RightHandConstraintSet','LeftHand':'LeftHandConstraintSet',
                     'RightFoot':'RightFootConstraintSet','LeftFoot':'LeftFootConstraintSet'}
            if native not in classes:
                raise Unsupported(f'Kimodo official post-processing does not expose {native} as an end effector')
            constraints_module=importlib.import_module('kimodo.constraints')
            constraint_class=getattr(constraints_module,classes[native])
            frame=round(c['time_seconds']*self.fps)
            duration=int(self.fps*self.scene['duration_seconds'])
            # Official end-effector constraints are feasible full-pose
            # keyframes, not isolated XYZ values. Generate a native reference
            # pose, translate it so the requested effector reaches the world
            # goal, and optionally replace that effector's global rotation.
            with self.timed('constraint_reference_generation'),torch.no_grad():
                reference=self.model([self.scene['prompt']],[duration],constraint_lst=[],
                        num_denoising_steps=100,num_samples=1,post_processing=False,return_numpy=True)
            output_lookup={name:i for i,name in enumerate(self.output_native_names)}
            subset=[output_lookup[name] for name in self.native_names]
            positions=torch.from_numpy(reference['posed_joints'][0,frame,subset]).float()
            rotations=torch.from_numpy(reference['global_rot_mats'][0,frame,subset]).float()
            target=torch.tensor(native_positions(c['position_m'],self.spec['axes']),dtype=torch.float32)
            delta=target-positions[idx]
            positions=positions+delta
            smooth=torch.from_numpy(reference['smooth_root_pos'][0,frame,[[0,2]]]).float()
            smooth=smooth+delta[[0,2]][None]
            if 'rotation_matrix' in c:
                rotations[idx]=torch.tensor(Y_TO_Z.T @ np.asarray(c['rotation_matrix']),dtype=torch.float32)
            result.append(constraint_class(skeleton,torch.tensor([frame]),positions[None],
                    rotations[None],smooth).to('cuda'))
            return result
        for c in specifications:
            idx=self.names.index(c['joint']); native=self.native_names[idx]
            pos=torch.tensor(native_positions(c['position_m'],self.spec['axes']),dtype=torch.float32,device='cuda')
            rot=None if 'rotation_matrix' not in c else torch.tensor(Y_TO_Z.T @ np.asarray(c['rotation_matrix']),dtype=torch.float32,device='cuda')
            result.append(SparseJointConstraint(round(c['time_seconds']*self.fps),idx,pos,rot))
        return result

    def initialize_state(self):
        return {'history':None,'frames':0}

    def step(self,state,prompt):
        import torch
        if self.name!='ardy':
            raise Unsupported('Kimodo is offline authoring, not ARDY streaming')
        horizon=self.model.gen_horizon_len
        with self.timed('replan'),torch.no_grad():
            history=state['history']
            n=0 if history is None else history.shape[1]
            motion=self.model.autoregressive_step(num_frames=n+horizon,init_history_sequence=history,texts=[prompt],
                    num_denoising_steps=int(self.model.diffusion.num_base_steps),cfg_weight=2.0,motion_mask=None,observed_motion=None)
        # Native API returns normalized explicit motion, including the supplied history.
        state['history']=motion[:,-160:].detach()
        state['frames']+=horizon
        decoded=self.model.motion_rep.inverse(motion,is_normalized=True)
        if history is not None:
            old=self.model.motion_rep.inverse(history,is_normalized=True)['posed_joints']
            state['overlap_errors_m']=torch.linalg.vector_norm(decoded['posed_joints'][:,:n]-old,dim=-1).detach().cpu().numpy()
        return {k:v[:,-horizon:] if torch.is_tensor(v) and v.ndim>=2 and v.shape[1]==motion.shape[1] else v for k,v in decoded.items()}

    def streaming(self):
        import torch
        state=self.initialize_state(); joints=[]; rotations=[]; boundaries=[]; errors=[]; updates=set(); prompt=self.scene['prompt']
        total=round(self.scene['duration_seconds']*self.fps)
        while state['frames']<total:
            start=state['frames']
            for i,update in enumerate(self.scene.get('updates',[])):
                if start/self.fps>=update['time_seconds'] and i not in updates:
                    prompt=update['prompt']; updates.add(i)
                    event(self.events,'prompt_update',requested_seconds=update['time_seconds'],applied_seconds=start/self.fps,prompt=prompt)
            out=self.step(state,prompt); count=min(state['frames']-start,total-start)
            if start:
                boundaries.append(start)
            joints.append(out['posed_joints'][0,:count].detach().cpu().numpy())
            rotations.append(out['global_rot_mats'][0,:count].detach().cpu().numpy())
            if 'overlap_errors_m' in state:
                errors.extend(state['overlap_errors_m'].ravel().tolist())
            event(self.events,'chunk_committed',frame=start,frames=count,external_stitch=False)
        np.save(self.path/'raw/streaming_joints.npy',np.concatenate(joints))
        return self.save(np.concatenate(joints),self.names,np.concatenate(rotations),chunk_boundaries=boundaries,overlap_errors_m=errors)

    def run(self,benchmark=False):
        import torch
        self.seed_all(self.seed)
        constraints=self.constraint_list()
        self.postprocess=self.name=='kimodo' and bool(constraints)
        if self.scene['kind']=='online':
            return self.streaming()
        output=self.generate(constraints,cold=True)
        np.savez_compressed(self.path/'raw/native.npz',**{k:v for k,v in output.items() if isinstance(v,np.ndarray)})
        path=self.save(output['posed_joints'][0],self.output_names,output['global_rot_mats'][0],foot_contacts=output.get('foot_contacts',[None])[0])
        if benchmark:
            self.measure_repetitions(lambda **flags:self.generate(constraints,**flags))
        return path

    def generate(self,constraints,warmup=False,cold=False,benchmark=False):
        import torch
        self.call_flags={'warmup':warmup,'cold':cold,'benchmark':benchmark}
        with self.timed('generation',**self.call_flags),torch.no_grad():
            if self.name=='kimodo':
                # Kimodo decodes posed_joints from rotations. Its official
                # post-processor is therefore required to project sparse
                # position/orientation commands back onto the output pose.
                output=self.model([self.scene['prompt']],[int(self.fps*self.scene['duration_seconds'])],constraint_lst=constraints,num_denoising_steps=100,num_samples=1,post_processing=bool(constraints),return_numpy=True)
            else:
                lengths=torch.tensor([int(self.fps*self.scene['duration_seconds'])],device='cuda')
                observed,mask=(None,None)
                if constraints:
                    observed,mask=self.model.motion_rep.create_conditions_from_constraints_batched(constraints,lengths,to_normalize=True,device='cuda')
                motion=self.model([self.scene['prompt']],int(lengths[0]),num_denoising_steps=int(self.model.diffusion.num_base_steps),motion_mask=mask,observed_motion=observed,cfg_weight=2.0,crop_history_length=160)
                output=self.model.motion_rep.inverse(motion,is_normalized=True)
                output={k:v.detach().cpu().numpy() if torch.is_tensor(v) else v for k,v in output.items()}
        return output


class Chois(Adapter):
    def preflight(self):
        super().preflight()
        for name in ('SMPLX_NEUTRAL.npz','SMPLX_MALE.npz','SMPLX_FEMALE.npz',
                     'cano_test_diffusion_manip_window_120_joints24.p',
                     'cano_min_max_mean_std_data_window_120_joints24.p'):
            require(self.cfg,self.name,name)
        clip=asset_root(self.cfg)/'omnicontrol/ViT-B-32.pt'
        if not clip.is_file():
            raise MissingResources('CHOIS uses OpenAI ViT-B/32; prepare the shared OmniControl CLIP asset')
        from .resource_review import inference_review
        inference_review(self)

    def load(self):
        import torch
        from .chois_compat import install_chamfer_fallback
        self.chois_chamfer_backend=install_chamfer_fallback()
        import trainer_chois as upstream

        repo=repo_root(self.cfg,self.name)
        data=require(self.cfg,self.name,'cano_test_diffusion_manip_window_120_joints24.p').parent
        checkpoint=require(self.cfg,self.name,self.spec['model'])
        import manip.data.cano_traj_dataset as chois_data
        # The release leaves the author's absolute SMPL-H path in module state.
        # Point it at the matching models included in the official data archive.
        chois_data.SMPLH_PATH=str(data/'smpl_all_models/smplh_amass')
        saved_argv=sys.argv
        try:
            sys.argv=['trainer_chois.py']
            opt=upstream.parse_opt()
        finally:
            sys.argv=saved_argv
        opt.window=120; opt.batch_size=1
        opt.data_root_folder=str(data); opt.pretrained_model=str(checkpoint)
        opt.save_res_folder=str(self.path/'raw/chois_results')
        opt.save_dir=str(self.path/'raw/chois_runtime'); opt.exp_name='chois_runtime'
        opt.input_first_human_pose=True; opt.use_random_frame_bps=True
        opt.add_language_condition=True; opt.use_object_keypoints=True
        opt.add_semantic_contact_labels=True; opt.use_guidance_in_denoising=False
        opt.loss_w_feet=1.; opt.loss_w_fk=.5; opt.loss_w_obj_pts=1.
        opt.test_sample_res=True; opt.compute_metrics=True
        repr_dim=3+9+24*3+22*6+4
        with self.timed('model_load'):
            model=upstream.ObjectCondGaussianDiffusion(opt,d_feats=repr_dim,d_model=opt.d_model,
                n_dec_layers=opt.n_dec_layers,n_head=opt.n_head,d_k=opt.d_k,d_v=opt.d_v,
                max_timesteps=opt.window+1,out_dim=repr_dim,timesteps=1000,
                objective='pred_x0',loss_type='l1',input_first_human_pose=True,
                use_object_keypoints=True).cuda().eval().requires_grad_(False)

        # Build the exact upstream inference helper without its optimizer, GradScaler,
        # training dataset/dataloader, or EMA duplicate.
        trainer=upstream.Trainer.__new__(upstream.Trainer)
        trainer.use_wandb=False; trainer.model=model
        trainer.ema=SimpleNamespace(ema_model=model); trainer.step=0
        trainer.batch_size=1; trainer.results_folder=str(self.path/'raw/chois_runtime/weights')
        trainer.vis_folder=str(self.path/'raw/chois_runtime/vis_res')
        trainer.opt=opt; trainer.window=opt.window
        for name in ('add_language_condition','use_random_frame_bps','use_object_keypoints',
                     'add_semantic_contact_labels','test_unseen_objects','test_on_train',
                     'input_first_human_pose','use_guidance_in_denoising','compute_metrics',
                     'use_long_planned_path','use_object_split'):
            setattr(trainer,name,getattr(opt,name))
        for name in ('save_res_folder','data_root_folder','loss_w_feet','loss_w_fk','loss_w_obj_pts',
                     'test_object_name','test_scene_name'):
            setattr(trainer,name,getattr(opt,name))

        val=upstream.CanoObjectTrajDataset(train=False,data_root_folder=str(data),window=opt.window,
            use_object_splits=opt.use_object_split,input_language_condition=True,
            use_random_frame_bps=True,use_object_keypoints=True)
        # The public archive omits the per-window BPS cache (upstream documents
        # that building every window takes over an hour).  Materialize only the
        # selected batch=1 window inside this run directory.
        val.dest_obj_bps_npy_folder=str(self.path/'raw/chois_bps')
        Path(val.dest_obj_bps_npy_folder).mkdir(parents=True,exist_ok=True)
        entry=val.window_data_dict[0]
        seq_name=entry['seq_name']; object_name=seq_name.split('_')[1]
        bps_path=Path(val.dest_obj_bps_npy_folder)/(seq_name+'_'+str(entry['ori_w_idx'])+'.npy')
        with self.timed('input_bps_prepare'),torch.no_grad():
            rest_verts,_=val.load_rest_pose_object_geometry(object_name)
            rotations=torch.from_numpy(entry['obj_rot_mat']).float()
            centers=torch.from_numpy(entry['window_obj_com_pos']).float()
            vertices=val.load_object_geometry_w_rest_geo(rotations,centers,torch.from_numpy(rest_verts).float())
            object_bps=val.compute_object_geo_bps(vertices,centers)
            np.save(bps_path,object_bps.detach().cpu().numpy())
        class OneSample(torch.utils.data.Dataset):
            def __len__(self): return 1
            def __getitem__(self,index): return val[0]
            def __getattr__(self,name): return getattr(val,name)
        trainer.val_ds=OneSample(); trainer.ds=trainer.val_ds; trainer.bm_dict=val.bm_dict
        clip_load=upstream.clip.load
        try:
            upstream.clip.load=lambda _name,**kwargs: clip_load(str(asset_root(self.cfg)/'omnicontrol/ViT-B-32.pt'),**kwargs)
            trainer.clip_model=trainer.load_and_freeze_clip('ViT-B/32')
        finally:
            upstream.clip.load=clip_load
        trainer.hand_vertex_idxs,trainer.left_hand_vertex_idxs,trainer.right_hand_vertex_idxs=trainer.load_hand_vertex_ids()

        def load_ema(instance,_milestone,pretrained_path=None):
            with self.timed('checkpoint_load'):
                payload=torch.load(pretrained_path or checkpoint,map_location='cpu')
            state=payload.get('ema',payload)
            if isinstance(state,dict) and isinstance(state.get('ema_model'),dict):
                state=state['ema_model']
            elif isinstance(state,dict):
                selected={k.removeprefix('ema_model.'):v for k,v in state.items() if k.startswith('ema_model.')}
                state=selected or state
            incompatible=model.load_state_dict(state,strict=False)
            if len(incompatible.missing_keys)>8:
                raise RuntimeError(f'CHOIS EMA checkpoint incompatible: {len(incompatible.missing_keys)} missing keys')
            instance.step=int(payload.get('step',0))
        trainer.load=MethodType(load_ema,trainer)
        self.model=model; self.trainer=trainer; self.fps=30
        json_write(self.path/'upstream_config.json',vars(opt))
        json_write(self.path/'allocation_contract.json',{
            'batch_size':1,'validation_samples':1,'optimizer_created':False,
            'grad_scaler_created':False,'ema_model_copy_created':False,
            'checkpoint_target':'single inference model loaded from EMA weights',
            'bps_chamfer_backend':self.chois_chamfer_backend,
            'smplh_path':chois_data.SMPLH_PATH})

    def run(self,benchmark=False):
        import torch
        self.seed_all(self.seed)
        captured=[]; original=self.trainer.gen_vis_res_generic
        original_loader=torch.utils.data.DataLoader
        def capture(*args,**kwargs):
            result=original(*args,**kwargs)
            if not kwargs.get('vis_gt',False):
                captured.append(result)
            return result
        def single_process_loader(*args,**kwargs):
            kwargs['num_workers']=0
            kwargs['persistent_workers']=False
            return original_loader(*args,**kwargs)
        self.trainer.gen_vis_res_generic=capture
        torch.utils.data.DataLoader=single_process_loader
        try:
            with self.timed('generation'),torch.no_grad():
                self.trainer.cond_sample_res()
        finally:
            torch.utils.data.DataLoader=original_loader
        if len(captured)!=1:
            raise RuntimeError(f'CHOIS batch=1 contract violated: captured {len(captured)} predictions')
        _,joint_list,_,rotation_list,object_positions,object_rotations,object_vertices,*_=captured[0]
        joints=joint_list[0].detach().cpu().numpy()
        obj_pos=object_positions[0].detach().cpu().numpy()
        obj_rot=object_rotations[0].detach().cpu().numpy()
        obj_verts=object_vertices[0].detach().cpu().numpy()
        wrist=joints[:,22:24]
        hand_dist=np.linalg.norm(wrist[:,:,None,:]-obj_verts[:,None,:,:],axis=-1).min(axis=-1)
        names=['pelvis','left_hip','right_hip','spine1','left_knee','right_knee','spine2',
               'left_ankle','right_ankle','spine3','left_foot','right_foot','neck','left_collar',
               'right_collar','head','left_shoulder','right_shoulder','left_elbow','right_elbow',
               'left_wrist_body','right_wrist_body','left_wrist','right_wrist']
        np.savez_compressed(self.path/'raw/native_prediction.npz',joints=joints,
            global_rotations=rotation_list[0].detach().cpu().numpy(),object_positions=obj_pos,
            object_rotations=obj_rot,hand_object_distances_m=hand_dist)
        return self.save(joints,names,object_positions=obj_pos,object_rotations=obj_rot,
                         hand_object_distances_m=hand_dist)


def create_adapter(cfg,name,run_dir,scene,seed):
    if name=='motionlcm' and scene['kind']=='reconstruction':
        return MotionLCMVAE(cfg,name,run_dir,scene,seed)
    return {'motionlcm':MotionLCM,'omnicontrol':OmniControl,'chois':Chois,'ardy':Nvidia,'kimodo':Nvidia}[name](cfg,name,run_dir,scene,seed)
