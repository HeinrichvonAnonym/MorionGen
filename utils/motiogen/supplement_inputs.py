"""Read frozen accepted evidence; no historical result is edited or relabeled."""
import json
from pathlib import Path
from .config import ROOT,load_config

def accepted_inputs():
    source=ROOT/'develop_logs/test_runs/20261024_native_matrix_analysis/analysis.json'
    analysis=json.loads(source.read_text())
    for row in analysis['rows']:
        folder=ROOT/row['run']
        if row['method']=='primal':
            path=folder/'factorial'/f"{row['variant']}_canonical.npz"
        else:path=folder/'motion.npz'
        yield dict(row,motion_path=str(path),input_format='canonical',parent_analysis=str(source))
    audit=ROOT/'develop_logs/test_runs/20261024_kimodo_pair_audit/audit.json'
    base=Path(load_config()['data_root'])/'benchmarks/kimodo-stratified-20261008'
    for pair in json.loads(audit.read_text())['rows']:
        for condition,label in [(0,'raw'),(1,'postprocess')]:
            path=base/('generated_'+label)/pair['case']/'motion.npz'
            yield {'method':'kimodo','case':pair['case'],'task':pair['case'],'seed':pair['seed'],
                   'clock':'historical','condition':{'correction':condition},'run':str(path.parent),
                   'motion_path':str(path),'input_format':'kimodo_soma77','cluster':pair['cluster'],
                   'values':pair['metrics'][label],'parent_analysis':str(audit)}
