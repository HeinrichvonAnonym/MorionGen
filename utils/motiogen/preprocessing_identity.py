"""Identity of normalization and text configuration outside checkpoint tensors."""
import hashlib
import json
from .assets import find_asset
from .config import asset_root,repo_root
from .runtime import checksum
from .weight_identity import signature

def snapshot(cfg,method):
    base=asset_root(cfg);paths=[];absent=[]
    if method in ('motionlcm','omnicontrol'):
        for name in ('Mean.npy','Std.npy','Mean_raw.npy','Std_raw.npy'):
            path=find_asset(cfg,method,name)
            if path is None:raise FileNotFoundError(f'{method} normalization absent: {name}')
            paths.append(path)
    patterns={
        'motionlcm':['text/sentence-t5-large/config.json','text/sentence-t5-large/tokenizer*'],
        'omnicontrol':[],
        'chois':['chois/processed_data/cano_min_max_mean_std_data_window_120_joints24.p',
                 'chois/processed_data/cano_test_diffusion_manip_window_120_joints24.p'],
        'dart':['dart-official/data/seq_data_zero_male/mean_std_h2_f8.pkl',
                'dart-official/mvae/mvae_fps_clip/args.yaml','dart-official/mld_denoiser/mld_fps_clip_euler/args.yaml'],
        'closd':['closd-official/data/dataset/HumanML3D/Mean.npy','closd-official/data/dataset/HumanML3D/Std.npy',
                 'closd-official/checkpoints/dip/DiP_multi-target_10steps_context20_predict40/args.json',
                 'closd-distilbert/config.json','closd-distilbert/vocab.txt','closd-distilbert/tokenizer_config.json'],
        'primal':[], 'ardy':[], 'kimodo':[]}
    selected = cfg['baselines'].get(method, {}).get('preprocessing_patterns')
    if selected is None: selected = patterns[method]
    for pattern in selected:
        matches=[p for p in base.glob(pattern) if p.is_file() and not p.name.endswith(('.download.json','.verified.json'))]
        if not matches:raise FileNotFoundError('Preprocessing dependency absent: '+pattern)
        paths.extend(matches)
    if method=='omnicontrol':
        optional=base/'omnicontrol/omnicontrol_ckpt/args.json'
        if optional.is_file():paths.append(optional)
        else:absent.append(str(optional)) # Native parser defaults are versioned source.
    if method=='chois':paths.append(repo_root(cfg,method)/'bps.pt')
    if method in ('ardy','kimodo'):
        text=base/'text/llm2vec';policy=json.loads((text/'local_base_receipt.json').read_text())
        paths.append(text/'original_base/config.json')
        for name in (policy['mntp_repo'],policy['supervised_repo']):
            directory=text/name
            for path in directory.glob('*.json'):
                if not path.name.endswith(('.download.json','.verified.json')):paths.append(path)
    rows=[]
    for path in sorted(set(p.resolve(strict=True) for p in paths)):
        before=signature(path);digest=checksum(path)
        if signature(path)!=before:raise RuntimeError('Preprocessing file changed during hashing: '+str(path))
        rows.append({'path':str(path),'sha256':digest,'stat':before})
    identity=hashlib.sha256(json.dumps({'files':{r['path']:r['sha256'] for r in rows},'absent_optional':absent},sort_keys=True).encode()).hexdigest()
    return {'method':method,'identity':identity,'files':rows,'absent_optional':absent,
            'scope':'Explicit normalization, selected CHOIS input-window bank/BPS, and text/config metadata outside checkpoint tensors. PRIMAL normalization is checkpoint-contained. Not an exhaustive dynamic file-open trace.'}
