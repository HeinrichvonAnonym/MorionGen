"""CPU EMA/config compatibility check, preserving unused checkpoint keys."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import ROOT,load_config
from motiogen.runtime import checksum,json_write
from motiogen.chois_compat import install_chamfer_fallback

base=ROOT/'develop_logs/test_runs'
analysis=json.loads((base/'20261024_native_matrix_analysis/analysis.json').read_text())
rows=[r for r in analysis['rows'] if r['method']=='chois']
config_paths=[ROOT/r['run']/'upstream_config.json' for r in rows]
configs=[json.loads(p.read_text()) for p in config_paths]
keys=['window','d_model','n_dec_layers','n_head','d_k','d_v',
      'input_first_human_pose','use_object_keypoints','add_language_condition','add_semantic_contact_labels']
signatures=[{k:cfg[k] for k in keys} for cfg in configs]
if not signatures or any(s!=signatures[0] for s in signatures):
    raise ValueError('CHOIS runs do not share the checked architecture configuration')
data=Path(load_config()['data_root'])
sys.path.insert(0,str(data/'baselines/chois_release'))
install_chamfer_fallback()
from manip.model.transformer_object_motion_cond_diffusion import ObjectCondGaussianDiffusion
opt=SimpleNamespace(**configs[0])
model=ObjectCondGaussianDiffusion(opt,d_feats=220,d_model=opt.d_model,
    n_dec_layers=opt.n_dec_layers,n_head=opt.n_head,d_k=opt.d_k,d_v=opt.d_v,
    max_timesteps=opt.window+1,out_dim=220,timesteps=1000,objective='pred_x0',
    loss_type='l1',input_first_human_pose=True,use_object_keypoints=True).cpu().eval()
checkpoint=data/'outputs/assets/chois/pretrained_models/model-10.pt'
before=checksum(checkpoint)
payload=torch.load(checkpoint,map_location='cpu')
state=payload.get('ema',payload)
if isinstance(state.get('ema_model'),dict):state=state['ema_model']
else:
    selected={k.removeprefix('ema_model.'):v for k,v in state.items() if k.startswith('ema_model.')}
    state=selected or state
try:
    model.load_state_dict(state,strict=True)
    strict_probe={'passed':True,'error':None}
except RuntimeError as error:
    strict_probe={'passed':False,'error':str(error)}
# The release's own Trainer.load uses strict=False (trainer_chois.py:393-401).
# Its published checkpoint also contains five tensors for modules absent from
# this release's inference architecture. Make that difference explicit; require
# every actually instantiated parameter/buffer, rather than relaxing omissions.
expected_unused={'body_shape_encoder.0.weight','body_shape_encoder.0.bias',
    'body_shape_encoder.2.weight','body_shape_encoder.2.bias','contact_embedding.action_embedding'}
incompatible=model.load_state_dict(state,strict=False)
if incompatible.missing_keys or set(incompatible.unexpected_keys)!=expected_unused:
    raise ValueError('Missing inference tensors or a different checkpoint/release mismatch')
loaded=model.state_dict()
model.load_state_dict({k:state[k] for k in loaded},strict=True)
exact=all(torch.equal(v,state[k]) for k,v in loaded.items())
receipt={'analysis_cutoff':analysis['updated_at'],'runs':[r['run'] for r in rows],
    'configuration':signatures[0],
    'configuration_sha256':{str(p.relative_to(ROOT)):checksum(p) for p in config_paths},
    'checkpoint':str(checkpoint),'checkpoint_sha256':before,
    'missing_keys':list(incompatible.missing_keys),'unexpected_keys':list(incompatible.unexpected_keys),
    'full_checkpoint_strict_probe':strict_probe,
    'unused_checkpoint_tensors':{k:list(state[k].shape) for k in sorted(expected_unused)},
    'checkpoint_state_tensors':len(state),'loaded_state_tensors':len(loaded),'parameter_tensors':len(list(model.named_parameters())),
    'parameter_elements':sum(p.numel() for p in model.parameters()),
    'fixed_parameter_elements':sum(p.numel() for p in model.parameters() if not p.requires_grad),
    'all_loaded_values_exact':exact,
    'passed':exact and checksum(checkpoint)==before,
    'scope':'Same official inference architecture and EMA extraction as the adapter, on CPU. Full-checkpoint strict loading rejects five extra tensors for modules absent from the released inference code; official Trainer.load also uses strict=False. Every instantiated inference parameter and buffer is present and matches exactly, with the matching subset additionally loaded strictly. No training, sampling, GPU allocation or extra experimental sample. Historical per-run identities remain in the weight audit.'}
json_write(base/'20261024_chois_checkpoint_load/audit.json',receipt)
print(json.dumps({k:v for k,v in receipt.items() if k not in ['runs','configuration_sha256']}))
if not receipt['passed']:raise SystemExit(1)
