"""Explicit, fixed K4 candidate bank with D-only whole-window selection."""
import json
from pathlib import Path
import shutil
import numpy as np
from .config import ROOT, digest
from .motion import load
from .candidate_quality import assess, choose


def fit(run, work, mass, floor, settings):
    from .k4_fitting import fit as fit_single
    import biomech
    if settings.get('version') != 'k4-candidate-pool-1':
        raise ValueError('Unsupported candidate bank')
    candidates = settings['candidates']
    if not candidates or any(not name.isidentifier() for name in candidates):
        raise ValueError('Candidate names must be identifiers')
    if any(c.get('version') != 'k4-fit-1' or c.get('export_clock') != 'native_multiple' for c in candidates.values()):
        raise ValueError('Candidate bank requires independent single fits with exact native knots')
    motion = load(run, settings.get('method'), config=ROOT/'config/evalkit_k4_20261117/input.json')
    key = {'inputs':motion.provenance['inputs'], 'config':settings, 'mass_kg':mass, 'floor_m':floor,
           'model_sha256':digest(biomech.MODEL),
           'implementation':{str(p.relative_to(Path(__file__).parent)):digest(p) for p in sorted(Path(__file__).parent.rglob('*.py'))},
           'input_config_sha256':digest(ROOT/'config/evalkit_k4_20261117/input.json'),
           'biomech_sha256':digest(biomech.__file__)}
    out = Path(work); out.mkdir(parents=True,exist_ok=True)
    receipt_path = out/'receipt.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt['key'] != key:
            raise ValueError('K4 candidate-bank cache identity changed; use a new output directory')
        with np.load(out/'fit.npz') as z:
            return {**{k:z[k] for k in z.files},'receipt':receipt}
    qualities, failures, receipts = {}, {}, {}
    for name, config in candidates.items():
        folder = out/'candidates'/name
        try:
            candidate = dict(config, method=motion.method)
            values = fit_single(run, folder, mass, floor, candidate)
            if values['receipt']['key']['inputs'] != key['inputs']:
                raise ValueError('Candidate input identity differs')
            head = None
            if (folder/'head_neck.npz').exists():
                with np.load(folder/'head_neck.npz') as z:
                    head = z['head_relative_thorax']
            qualities[name] = assess(motion,values,head_rotation=head,thresholds=settings['selection_thresholds'],
                                     include_export_path=settings.get('selection_include_export_path',False))
            receipts[name] = values['receipt']
        except (ValueError,RuntimeError) as error:
            failures[name] = {'type':type(error).__name__,'message':str(error)}
    if not qualities:
        (out/'candidate_failures.json').write_text(json.dumps(failures,indent=2)+'\n')
        raise RuntimeError('All candidates failed; see candidate_failures.json')
    selected = choose(qualities)
    for source in (out/'candidates'/selected).iterdir():
        if source.name == 'receipt.json':
            continue
        destination = out/source.name
        if source.is_dir():
            shutil.copytree(source,destination,dirs_exist_ok=True)
        else:
            shutil.copyfile(source,destination)
    selection = {'selected':selected,'qualities':qualities,'failures':failures,
                 'rule':'lexicographic: missing supported observations, failed targets, worst normalized error, mean normalized error, declaration order',
                 'qualification':'Input-fidelity selection only; independent R acceptance remains required',
                 'all_candidates_retained':True,'per_frame_switching':False}
    (out/'candidate_selection.json').write_text(json.dumps(selection,indent=2,allow_nan=False)+'\n')
    receipt = {**receipts[selected],'key':key,'version':settings['version'],
               'selected_candidate_key':receipts[selected]['key'],'candidate_selection':selection}
    receipt_path.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    with np.load(out/'fit.npz') as z:
        return {**{k:z[k] for k in z.files},'receipt':receipt}
