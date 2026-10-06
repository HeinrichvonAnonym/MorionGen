"""Summarize engineering outcomes within identical source/config/scenario groups."""
import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'utils'))
from motiogen.runtime import json_write


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def summarize(root):
    groups={}
    for path in sorted(Path(root).glob('*/manifest.json')):
        data=json.loads(path.read_text())
        if data.get('action') not in ('demo','train-smoke') or data.get('status')=='running':
            continue
        folder=path.parent
        cfg=json.loads((folder/'config.json').read_text())
        code=json.loads((folder/'implementation.json').read_text())
        key=(data['baseline'],data['action'],(data.get('scenario') or {}).get('name'),
             cfg['baselines'][data['baseline']].get('variant','native'),fingerprint(cfg),fingerprint(code))
        group=groups.setdefault(key,{'baseline':key[0],'action':key[1],'scenario':key[2],'variant':key[3],
                    'config_sha256':key[4],'implementation_sha256':key[5],'runs':[]})
        started=any(json.loads(line).get('event')=='process_started' for line in (folder/'events.jsonl').read_text().splitlines())
        group['runs'].append({'path':str(folder),'seed':data.get('seed'),'status':data['status'],'started':started,'reason':data.get('reason')})
    for group in groups.values():
        rows=group['runs']; started=[r for r in rows if r['started']]
        group['requested_trials']=len(rows); group['started_trials']=len(started)
        group['status_counts']=dict(collections.Counter(r['status'] for r in rows))
        group['resource_termination_rate']={'value':sum(r['status']=='resource_terminated' for r in started)/len(started) if started else None,
                    'n':len(started),'denominator':'started workers; preflight reserve rejection listed separately'}
        group['timeout_rate']={'value':sum(r['status']=='timeout' for r in started)/len(started) if started else None,'n':len(started)}
        group['preflight_resource_rejections']=sum(not r['started'] and r['status'] in ('resource_gate','resource_terminated') for r in rows)
    return {'scope':'Engineering status only; no aggregate quality score or cross-baseline ranking. Different implementations/configs remain separate. Zero started workers yields unavailable rates.',
            'groups':list(groups.values())}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--run-root',default=str(ROOT/'develop_logs/test_runs'))
    parser.add_argument('--output',default=str(ROOT/'develop_logs/test_runs/bootstrap/run_status_summary.json'))
    args=parser.parse_args(); output=Path(args.output).resolve()
    if not output.is_relative_to(ROOT/'develop_logs/test_runs'):
        parser.error('Summary outputs belong under develop_logs/test_runs')
    result=summarize(args.run_root); json_write(output,result)
    print(json.dumps({'groups':len(result['groups']),'output':str(output)}))


if __name__=='__main__':
    main()
