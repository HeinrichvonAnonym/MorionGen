"""Fail-closed short training. A reviewed bound is required, not an override flag."""
import hashlib
import json
import time
from pathlib import Path

from .adapters import ResourceGate,Unsupported,require
from .runtime import budgets,checksum,event,json_write


def preflight_training(adapter):
    if adapter.name!='motionlcm':
        raise Unsupported('This round only admits native MotionLCM frozen-backbone ControlNet training')
    evidence_path=Path(adapter.cfg['_source']).parent/'training_evidence.json'
    if not evidence_path.is_file():
        raise ResourceGate('No reviewed training memory/data evidence. Inference success is insufficient to permit backward.')
    evidence=json.loads(evidence_path.read_text())
    budget=budgets(adapter.cfg)
    required=['checkpoint_sha256','data_manifest_sha256','activation_bound_bytes','ram_bound_bytes','gpu_bound_bytes','review_basis','loss_path_verified','recovery_verified']
    if not all(x in evidence for x in required) or not evidence['loss_path_verified'] or not evidence['recovery_verified']:
        raise ResourceGate('Training evidence incomplete')
    if evidence['gpu_bound_bytes']>budget['gpu_bytes'] or evidence['ram_bound_bytes']>budget['ram_bytes']:
        raise ResourceGate('Reviewed training bound exceeds current resource budget')
    checkpoint=require(adapter.cfg,adapter.name,adapter.spec['model'])
    if checksum(checkpoint)!=evidence['checkpoint_sha256']:
        raise ResourceGate('Training evidence checkpoint digest mismatch')
    data_manifest=Path(evidence.get('data_manifest_path',''))
    if not data_manifest.is_file() or checksum(data_manifest)!=evidence['data_manifest_sha256']:
        raise ResourceGate('Training data evidence is absent/stale')
    files=json.loads(data_manifest.read_text())
    for entry in files:
        file=Path(entry['path'])
        if not file.is_file() or checksum(file)!=entry['sha256']:
            raise ResourceGate('Source motion/text/normalization data changed since review')
    return evidence


def train_smoke(adapter):
    evidence=preflight_training(adapter)
    import torch
    from mld.data.get_data import get_dataset
    from mld.utils.utils import move_batch_to_device
    if not adapter.model.is_controlnet:
        raise ResourceGate('Smoke requires loaded all-joint ControlNet checkpoint, not text-only model')
    cfg=adapter.model_cfg
    cfg.DATASET.HUMANML3D.ROOT=str(require(adapter.cfg,adapter.name,'Mean.npy').parent)
    cfg.DATASET.WORD_VERTILIZER_PATH=str(require(adapter.cfg,adapter.name,'our_vab_data.npy').parent)
    cfg.DATASET.HUMANML3D.CONTROL_ARGS.MEAN_STD_PATH=str(require(adapter.cfg,adapter.name,'Mean_raw.npy').parent)
    data=get_dataset(cfg)
    adapter.model.datamodule=data
    adapter.model.feats2joints=data.feats2joints
    loader=data.train_dataloader()
    if len(loader)==0:
        raise ResourceGate('No real training motion samples')
    model=adapter.model
    for name in ('vae','text_encoder','denoiser'):
        getattr(model,name).requires_grad_(False).eval()
    for name in ('controlnet','traj_encoder'):
        getattr(model,name).requires_grad_(True).train()
    def digest(parameters):
        value=hashlib.sha256()
        for name,param in parameters:
            value.update(name.encode()); value.update(param.detach().cpu().contiguous().numpy().tobytes())
        return value.hexdigest()
    frozen=[(n,p) for n,p in model.named_parameters() if not p.requires_grad]
    trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    frozen_before=digest(frozen); train_before=digest(trainable)
    optimizer=torch.optim.AdamW([p for _,p in trainable],lr=cfg.TRAIN.learning_rate,
                                betas=(cfg.TRAIN.adam_beta1,cfg.TRAIN.adam_beta2),eps=cfg.TRAIN.adam_epsilon,
                                weight_decay=cfg.TRAIN.adam_weight_decay)
    steps=[]; iterator=iter(loader)
    adapter.seed_all(adapter.seed)
    for step in range(20):
        try:
            batch=next(iterator)
        except StopIteration:
            iterator=iter(loader); batch=next(iterator)
        batch=move_batch_to_device(batch,torch.device('cuda'))
        torch.cuda.synchronize(); started=time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        losses=model.allsplit_step('train',batch)
        loss=losses['loss']
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite training loss')
        loss.backward()
        grads=[p.grad for _,p in trainable if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):
            raise RuntimeError('Missing/non-finite trainable gradients')
        torch.nn.utils.clip_grad_norm_([p for _,p in trainable],cfg.TRAIN.max_grad_norm)
        optimizer.step(); torch.cuda.synchronize()
        row={'step':step,'loss':float(loss.detach()),'seconds':time.perf_counter()-started}
        steps.append(row); event(adapter.events,'training_step',**row)
    frozen_after=digest(frozen); train_after=digest(trainable)
    if frozen_before!=frozen_after or train_before==train_after:
        raise RuntimeError('Frozen parameters changed or trainable parameters did not update')
    snapshot={'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),'steps':20,'evidence':evidence}
    model.on_save_checkpoint(snapshot)
    destination=adapter.path/'train_smoke.ckpt'
    torch.save(snapshot,destination)
    # Reload into the same object to avoid a second GPU-resident model.
    restored=torch.load(destination,map_location='cpu')
    incompatible=model.load_state_dict(restored['state_dict'],strict=False)
    if incompatible.unexpected_keys or any(not key.startswith('text_encoder.') for key in incompatible.missing_keys):
        raise RuntimeError(f'Unexpected checkpoint mismatch: {incompatible}')
    optimizer.load_state_dict(restored['optimizer'])
    if digest(trainable)!=train_after:
        raise RuntimeError('Checkpoint roundtrip changed trainable parameters')
    result={'status':'success','steps':steps,'frozen_unchanged':True,'trainable_updated':True,'checkpoint_roundtrip':True,
            'trainable_parameters':sum(p.numel() for _,p in trainable),'mean_step_seconds':sum(x['seconds'] for x in steps)/len(steps),
            'limitation':'resource/loss-path smoke only; repeated tiny samples do not establish convergence or quality gains',
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved()}
    json_write(adapter.path/'training_result.json',result)
    return result
