"""Timing sensitivity to interrupted cross-reboot blocks, not outcome selection."""
import json
import sys
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'utils'))
from motiogen.native_analysis import clustered_interval
from motiogen.runtime import json_write,checksum
base=ROOT/'develop_logs/test_runs'
recovery=json.loads((base/'20261025_interruption_recovery/receipt.json').read_text())
boot=datetime.fromisoformat(recovery['host_boot'])
checks=[];effects=[];cutoffs={}
for domain,relative in [('primary','20261024_native_matrix_analysis/analysis.json'),
                        ('trajectory','20261024_omnicontrol_trajectory/analysis.json')]:
    analysis=json.loads((base/relative).read_text());cutoffs[domain]=analysis['updated_at'];epochs=defaultdict(set)
    for row in analysis['rows']:
        if row['method'] not in ['motionlcm','omnicontrol']:continue
        events=[json.loads(line) for line in (ROOT/row['run']/'events.jsonl').read_text().splitlines()]
        start=datetime.fromisoformat(next(e['wall_time'] for e in events if e['event']=='process_started'))
        epochs[(row['method'],row['clock'],row['task'],row['seed'])].add('after_recovery' if start>=boot else 'before_interruption')
    for key,values in epochs.items():
        if len(values)>1:checks.append({'domain':domain,'method':key[0],'clock':key[1],'task':key[2],'seed':key[3],'epochs':sorted(values)})
    for contrast in analysis['contrasts']:
        method=contrast['method'];clock=contrast['clock']
        if contrast['metric']!='model_rtf' or method not in ['motionlcm','omnicontrol']:continue
        selected=[];removed=[]
        for block in analysis['blocks']:
            if not block['complete'] or block.get('method','omnicontrol')!=method or block.get('clock','paused')!=clock:continue
            match=next((c for c in block['contrasts'] if c['metric']==contrast['metric'] and c['factors']==contrast['factors']),None)
            if match is None:continue
            key=(method,clock,block['task'],block['seed'])
            (removed if len(epochs[key])>1 else selected).append((block['task'],match['delta']))
        if not selected:continue
        estimate=clustered_interval([v for _,v in selected],[t for t,_ in selected])
        effects.append({'domain':domain,'method':method,'clock':clock,'metric':contrast['metric'],'factors':contrast['factors'],
            'primary_paired_n':contrast['paired_n'],'primary_delta':contrast['mean_equal_group_weight'],'primary_ci95':contrast['ci95'],
            'removed_cross_epoch_blocks':len(removed),'same_epoch_paired_n':len(selected),
            'same_epoch_delta':estimate['mean_equal_group_weight'],'same_epoch_ci95':estimate['ci95']})
receipt={'updated_at':datetime.now(timezone.utc).isoformat(),'analysis_cutoffs':cutoffs,'mixed_blocks':checks,'effects':effects,
    'source_sha256':checksum(Path(__file__)),'scope':'Post-hoc operational sensitivity, eligibility based only on the documented host reboot, not measured outcomes. Retain all primary motion/quality results and timings; additionally remove complete paired timing blocks spanning the reboot. Same-epoch does not mean exclusive hardware or absence of other load.'}
json_write(base/'20261025_execution_epochs/audit.json',receipt)
print(json.dumps({'mixed_blocks':checks,'contrasts':len(effects)}))
