#!/usr/bin/env python
"""Pin CHOIS source dependencies externally and expose discipline-compliant links."""
import json
import os
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
data_root=Path(os.environ.get('MORIONGEN_DATA_ROOT',json.loads((ROOT/'config/baselines.json').read_text())['data_root'])).resolve()
registry=json.loads((ROOT/'config/dependency_sources.json').read_text())['sources']
selected={'human_body_prior','bps_torch','chamfer_distance','pytorch3d'}
stage=data_root/'staged_dependencies'
stage.mkdir(parents=True,exist_ok=True)

for item in registry:
    name=Path(item['path']).name
    if name not in selected:
        continue
    target=stage/name
    if not (target/'.git').is_dir():
        subprocess.run(['git','clone','--filter=blob:none','--no-tags',item['official_url'],str(target)],check=True)
    subprocess.run(['git','fetch','origin',item['sha'],'--depth=1'],cwd=target,check=True)
    subprocess.run(['git','checkout','--detach',item['sha']],cwd=target,check=True)
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=target,text=True).strip()
    # Editable installs may create package.egg-info beside setup.py. Pinning is
    # concerned with modifications to official tracked source; build metadata
    # is untracked and must not make an otherwise verified checkout unusable.
    dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=target,text=True).strip()
    if actual!=item['sha'] or dirty:
        raise RuntimeError(f'Pinned dependency verification failed for {name}')
    link=ROOT/item['path']
    link.parent.mkdir(parents=True,exist_ok=True)
    if link.is_symlink() and link.resolve()!=target:
        link.unlink()
    if not link.exists():
        link.symlink_to(target,target_is_directory=True)
    print(f'{name} {actual}')
