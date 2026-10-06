"""Stage upstream builds without modifying baseline clones; C++ sources live in third_party."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
name=sys.argv[1]
if name not in ('ardy','kimodo'):
    raise SystemExit('Only bundled MotionCorrection builds are staged here')
source=ROOT/'baselines'/name
target=ROOT/'python/third_party'/name
cpp=ROOT/'third_party'/name/'MotionCorrection'
cpp.parent.mkdir(parents=True,exist_ok=True)
if not cpp.exists():
    shutil.copytree(source/'MotionCorrection',cpp,ignore=shutil.ignore_patterns('build','__pycache__'))
if not target.exists():
    shutil.copytree(source,target,ignore=shutil.ignore_patterns('.git','MotionCorrection','build','*.egg-info','__pycache__'))
    (target/'MotionCorrection').symlink_to(cpp,target_is_directory=True)
    setup=target/'setup.py'
    setup.write_text(setup.read_text().replace('build_args += ["--", "-j4"]','build_args += ["--", "-j1"]'))
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
(target/'MOTIOGEN_SOURCE.json').write_text(json.dumps({'upstream':str(source),'git_sha':commit,'cpp_source':str(cpp),
    'patch':'setup.py: use -j1 instead of -j4 to bound compiler parallelism; no inference changes'},indent=2)+'\n')
print(target)
