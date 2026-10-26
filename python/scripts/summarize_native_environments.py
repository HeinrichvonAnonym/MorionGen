"""Report runtime records separately from present-day interpreter inspection."""
import json
import subprocess
import sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'utils'))
from motiogen.runtime import json_write
base=ROOT/'develop_logs/test_runs'
analysis=json.loads((base/'20261024_native_matrix_analysis/analysis.json').read_text())
groups={}
for row in analysis['rows']:
    groups.setdefault(row['method'],set()).add(row['run'])
groups['kimodo']={'develop_logs/test_runs/20261008_014115_kimodo_evaluate-native_48f8e536',
                  'develop_logs/test_runs/20261024_180737_kimodo_audit-correction_e3b0e697'}
rows=[];cache={}
probe="import sys,json,torch;print(json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,'cuda_build':torch.version.cuda,'executable':sys.executable}))"
for method,runs in groups.items():
    recorded=[];executables=set();sources=[];packages=[]
    for relative in sorted(runs):
        run=ROOT/relative
        manifest=json.loads((run/'manifest.json').read_text())
        executables.add(manifest['command'][0])
        env=run/'environment.json'
        if env.exists():
            d=json.loads(env.read_text());recorded.append((d['python'].split()[0],d['torch'],d['cuda_build'],d['device']))
            sources.append(str(env.relative_to(ROOT)))
        pkg=run/'environment_packages.json'
        if pkg.exists():
            d=json.loads(pkg.read_text());packages.append((d['python'].split()[0],d['packages'].get('torch')))
            sources.append(str(pkg.relative_to(ROOT)))
    current=[]
    for executable in sorted(executables):
        if executable not in cache:
            completed=subprocess.run([executable,'-c',probe],text=True,capture_output=True,check=True,timeout=60)
            cache[executable]=json.loads(completed.stdout.strip().splitlines()[-1])
        current.append(cache[executable])
    rows.append({'method':method,'unique_runs':len(runs),'recorded_runtime_runs':len(recorded),
        'recorded_package_runs':len(packages),'recorded_variants':sorted(set(recorded)),
        'recorded_package_variants':sorted(set(packages)), 'current_interpreters':current,
        'sources':sources,'executables':sorted(executables)})
gpu=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.total,driver_version','--format=csv,noheader'],
    check=True,text=True,capture_output=True).stdout.strip().splitlines()
receipt={'recorded_at':datetime.now(timezone.utc).isoformat(),'analysis_cutoff':analysis['updated_at'],
    'rows':rows,'current_host_gpu':gpu,'current_os_release':Path('/etc/os-release').read_text(),
    'scope':'Runtime snapshots and package snapshots are historical evidence only where actually recorded. Current interpreter probes do not retroactively prove historical package versions. Probes import torch on CPU, do not initialize CUDA or launch kernels. Kimodo includes original generation and exact correction replay, not 416 newly generated motions.'}
json_write(base/'20261024_native_environments/summary.json',receipt)
print(json.dumps({'methods':len(rows),'runtime_record_counts':{r['method']:r['recorded_runtime_runs'] for r in rows},'gpu':gpu}))
