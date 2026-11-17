"""Incomplete or failed work must remain in all 656 report denominators."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]


def test_pending_and_execution_failure_are_not_passes(tmp_path):
    methods=['ardy','chois','closd','dart','kimodo','motionlcm','omnicontrol','primal']
    records=[dict(window=f'{i:03d}',method=m) for i in range(82) for m in methods]
    protocol=dict(records=records,html_report=str(tmp_path/'report.html'),note_report=str(tmp_path/'report.md'))
    (tmp_path/'protocol.json').write_text(json.dumps(protocol))
    folder=tmp_path/'tasks/000_ardy';folder.mkdir(parents=True)
    (folder/'result.json').write_text(json.dumps(dict(window='000',method='ardy',status='failed',passed=False,error='</script>solver failed')))
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    subprocess.run([sys.executable,str(ROOT/'python/scripts/evalkit_k4_full_report.py'),
                    '--run',str(tmp_path),'--code-root',str(ROOT)],check=True,env=env)
    result=json.loads((tmp_path/'results_656.json').read_text())
    assert result['expected']==656 and result['passed']==0 and result['evaluated']==0
    assert result['execution_failed']==1 and result['pending']==655
    assert all(x['total']==82 for x in result['by_format'].values())
    assert result['by_format']['ardy']['execution_failed']==1
    assert (tmp_path/'report.html').read_text().count('</script>')==1
