"""Read frozen scored evidence without recomputing metrics or changing acceptance."""
import hashlib,json
from pathlib import Path


def read_record(run, record):
    run=Path(run);key=record['window']+'_'+record['method'];task=run/'tasks'/key
    files={}
    def read(path):
        raw=path.read_bytes();files[str(path)]=hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    result=read(task/'result.json')
    if any(result.get(k)!=record[k] for k in ('window','method')):raise ValueError('Result identity differs from registered record')
    payload={k:record.get(k) for k in ('window','method','group_id','subset','recording_id','source_window_s')}
    payload.update(status=result['status'],passed=result.get('passed'))
    if result['status']!='completed':
        payload.update(error=result.get('error'),source_sha256=files);return payload
    summary=read(task/'summary.json');views={}
    if any(summary.get(k)!=record[k] for k in ('window','method')):raise ValueError('Summary identity differs')
    if summary['passed']!=result['passed']:raise ValueError('Acceptance and completed result differ')
    for name in ('native','export'):
        score=read(Path(summary['score_files'][name]))
        views[name]={}
        for clock,data in [('D_clock',score),('R_clock',score['original_clock'])]:
            views[name][clock]={k:data.get(k) for k in ('clock','comparisons','head_neck_rotation_comparisons','input_boundary_missing_frames')}
            views[name][clock]['acceptance']=summary['acceptance'][name][clock]
    selection=read(Path(summary['candidate_selection_file']))
    if selection['selected']!=summary['selected']:raise ValueError('Saved candidate choice differs from score summary')
    payload.update(selected=summary['selected'],views=views,export_audit=summary['export_audit'],
        failed_metrics=summary['failed_metrics'],selection_budget=selection.get('budget'),
        source_sha256=files,scope='Exact stored metrics, independent original R; all view/clock/stratum pairs preserved. Display does not rescore or select fits.')
    return payload
