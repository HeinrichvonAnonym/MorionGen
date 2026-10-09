"""Native evaluation plans and bounded smoke-result contracts."""
import json
import os
import shutil
from pathlib import Path

from .assets import find_asset
from .config import asset_root, data_root, repo_root
from .motion import load_motion
from .runtime import checksum, json_write


NATIVE = {
    'motionlcm': {
        'entry': 'test.py',
        'command': ['python','-m','test','--cfg','configs/motionlcm_control_s.yaml'],
        'required_assets': ['motionlcm_humanml_s_all.ckpt','finest.tar','Mean.npy','Std.npy'],
        'dataset': 'Complete HumanML3D test split',
    },
    'omnicontrol': {
        'entry': 'eval_omnicontrol.sh',
        'command': ['bash','eval_omnicontrol.sh','MODEL_PATH','20','5'],
        'required_assets': ['model_humanml3d.pt','finest.tar','Mean.npy','Std.npy'],
        'dataset': 'Complete HumanML3D test split',
    },
    'chois': {
        'entry': 't2m_eval/final_evaluations.py',
        'command': ['python','t2m_eval/final_evaluations.py'],
        'required_assets': ['model-10.pt','cano_test_diffusion_manip_window_120_joints24.p'],
        'dataset': 'CHOIS/OMOMO processed evaluation split and evaluator folder',
    },
    'ardy': {
        'entry': 'scripts/generate.py',
        'command': None,
        'required_assets': ['ARDY-Core-RP-20FPS-Horizon8'],
        'dataset': None,
        'reason': 'ARDY publishes inference, not a compatible offline distribution benchmark; use online A/B/H metrics.',
    },
    'kimodo': {
        'entry': 'benchmark/evaluate_folder.py',
        'command': [
            ['python','benchmark/create_benchmark.py','TESTSUITE','--dataset','BONES_SEED_SOMA_UNIFORM'],
            ['python','benchmark/generate_eval.py','--benchmark','TESTSUITE','--output','GENERATED','--model','kimodo-soma-rp','--batch_size','1','--num_workers','0'],
            ['python','benchmark/embed_folder.py','GENERATED','--model','tmr-soma-rp'],
            ['python','benchmark/evaluate_folder.py','GENERATED'],
        ],
        'required_assets': ['Kimodo-SOMA-RP-v1.1','TMR-SOMA-RP-v1'],
        'dataset': 'nvidia/Kimodo-Motion-Gen-Benchmark plus licensed BONES-SEED soma_uniform',
    },
}


def _writable_space(path, minimum_bytes=10 * 1024**3):
    """Return a non-mutating output-space receipt for the nearest parent."""
    path=Path(path); existing=path
    while not existing.exists() and existing!=existing.parent:
        existing=existing.parent
    free=shutil.disk_usage(existing).free
    writable=os.access(existing,os.W_OK)
    return {'path':str(path),'checked_parent':str(existing),'writable':writable,
            'free_bytes':free,'minimum_bytes':minimum_bytes,
            'ready':bool(writable and free>=minimum_bytes)}


def full_preflight(cfg, name, run_dir):
    spec=NATIVE[name]; repo=repo_root(cfg,name); entry=repo/spec['entry']
    missing=[]
    if not entry.is_file():
        missing.append({'kind':'entry','name':spec['entry']})
    for item in spec['required_assets']:
        if find_asset(cfg,name,item) is None:
            missing.append({'kind':'asset','name':item})
    dataset_evidence=None
    if name=='kimodo':
        from .datasets import benchmark_metadata_root, bvh_members, source_receipts
        try:
            receipts=source_receipts(); testsuite=benchmark_metadata_root(cfg)
            full_root=data_root(cfg)/'datasets/bones-seed'; full_dataset=full_root/'soma_uniform/bvh'
            meta_count=sum(1 for _ in testsuite.rglob('meta.json')) if testsuite.is_dir() else 0
            required=bvh_members(testsuite) if testsuite.is_dir() else []
            missing_bvh=sum(not (full_root/member).is_file() for member in required)
            # Current stratified outputs are roughly 2 MiB/case before all
            # embeddings and summaries.  Reserve 64 GiB for 22,474 cases so
            # the full command cannot start against a merely writable but
            # practically undersized volume.
            output_space=_writable_space(data_root(cfg)/'benchmarks/kimodo-full',64 * 1024**3)
            dataset_evidence={'source_archive':receipts['bones_seed']['archive'],
                              'benchmark_metadata':str(testsuite),'benchmark_cases':meta_count,
                              'required_unique_bvh':len(required),'missing_required_bvh':missing_bvh,
                              'full_extracted_bvh':str(full_dataset),'output_space':output_space}
            if meta_count!=22474:
                missing.append({'kind':'dataset_metadata','name':f'Kimodo benchmark metadata: expected 22474, found {meta_count}'})
            if missing_bvh:
                missing.append({'kind':'dataset_extraction','name':f'Full BONES-SEED is missing {missing_bvh}/{len(required)} required unique BVH files; stratified extraction does not satisfy full profile'})
            if not output_space['ready']:
                missing.append({'kind':'output_space','name':f"Need writable output with >= {output_space['minimum_bytes']} free bytes at {output_space['path']}"})
        except (KeyError,FileNotFoundError) as exc:
            missing.append({'kind':'dataset_source','name':str(exc)})
    elif spec.get('dataset'):
        missing.append({'kind':'dataset_confirmation','name':spec['dataset']})
    result={'status':'ready' if not missing else 'missing_resources','profile':'full-preflight-only',
            'baseline':name,'entry':str(entry),'command':spec.get('command'),'missing':missing,
            'reason':spec.get('reason'),'dataset_evidence':dataset_evidence,'execution_started':False,
            'policy':'Full native evaluation is never launched by this profile; use the emitted command after dataset confirmation.'}
    json_write(Path(run_dir)/'native_evaluation.json',result)
    return result


def stratified_preflight(cfg):
    """Validate the bounded Kimodo suite without accepting archive presence as extracted data."""
    from .datasets import bvh_members, prepared_root, stratified_root
    root=stratified_root(cfg); manifest=root/'selection_manifest.json'; missing=[]
    if not manifest.is_file():
        missing.append({'kind':'selection_manifest','name':str(manifest)})
        data={}
    else:
        data=json.loads(manifest.read_text())
        if data.get('case_count')!=1856 or data.get('group_count')!=58 or data.get('per_group')!=32:
            missing.append({'kind':'selection_contract','name':'expected 58 groups x 32 cases = 1856'})
        if data.get('extraction_status')!='ready':
            missing.append({'kind':'dataset_extraction','name':'selected BONES-SEED BVH files'})
        suite=root/'testsuite'; prepared=prepared_root(cfg)/'bones-seed-stratified-20261008'
        required=bvh_members(suite) if suite.is_dir() else []
        absent=sum(not (prepared/member).is_file() for member in required)
        if absent:
            missing.append({'kind':'dataset_extraction','name':f'missing {absent}/{len(required)} selected BONES-SEED BVH files'})
    for item in NATIVE['kimodo']['required_assets']:
        if find_asset(cfg,'kimodo',item) is None:
            missing.append({'kind':'asset','name':item})
    output_space=_writable_space(root)
    if not output_space['ready']:
        missing.append({'kind':'output_space','name':f"Need writable output with >= {output_space['minimum_bytes']} free bytes at {root}"})
    return {'status':'ready' if not missing else 'missing_resources','profile':'stratified',
            'baseline':'kimodo','root':str(root),'manifest':str(manifest),'missing':missing,
            'selection':{k:data.get(k) for k in ('seed','group_count','per_group','case_count','selection_sha256')},
            'required_unique_bvh':len(required) if manifest.is_file() else 0,'output_space':output_space}


def smoke_result(cfg, name, motion_file, run_dir, common_metrics):
    motion=load_motion(motion_file)
    # Canonical NPZ scalar fields load as zero-dimensional ndarrays. Persist
    # only plain JSON types in the native-evaluation protocol.
    fps=float(motion['fps'])
    result={'status':'success','profile':'smoke','baseline':name,'motion':str(motion_file),
            'motion_sha256':checksum(motion_file),'frames':len(motion['joints']),'fps':fps,
            'common_metrics':common_metrics,'native_distribution_metrics':None,
            'scope':'One real native inference followed by bounded A/B/E/H metrics; not an official distribution score.'}
    if name=='ardy':
        result['native_distribution_reason']=NATIVE[name]['reason']
    elif name=='kimodo':
        result['native_distribution_reason']='TMR/FID requires paired benchmark GT; run full preflight after BONES-SEED authorization.'
    else:
        result['native_distribution_reason']='FID/R-Precision require the complete compatible reference split and evaluator.'
    if name=='chois':
        metrics_root=Path(run_dir)/'raw/chois_results/evaluation_metrics_json'
        samples=sorted(path for path in metrics_root.rglob('*.json')
                       if path.name!='evaluation_metrics_for_all_test_data.json') if metrics_root.exists() else []
        if len(samples)==1:
            result['native_sample_metrics']=json.loads(samples[0].read_text())
            result['native_sample_metrics_file']=str(samples[0])
            result['native_sample_metrics_scope']='Unmodified upstream CHOIS single-window metrics; position errors are reported in upstream millimetres and are not distribution scores.'
    json_write(Path(run_dir)/'native_evaluation.json',result)
    return result
