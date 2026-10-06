"""Prefer exact cached CUDA wheels; resolve only missing dependencies online."""
import importlib.metadata as metadata
import subprocess
import sys
import zipfile
from email.parser import Parser
from pathlib import Path

from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.tags import sys_tags
from pip._vendor.packaging.utils import canonicalize_name,parse_wheel_filename

wheel=Path(sys.argv[1]).resolve(); cache=wheel.parent; tags=set(sys_tags())
with zipfile.ZipFile(wheel) as archive:
    path=next(x for x in archive.namelist() if x.endswith('.dist-info/METADATA'))
    needs=Parser().parsestr(archive.read(path).decode()).get_all('Requires-Dist')
available=[]
for path in cache.glob('*.whl'):
    name,version,_,supported=parse_wheel_filename(path.name)
    if tags.intersection(supported):
        available.append((name,version,path))
local=[str(wheel)]; remote=[]
for raw in needs:
    req=Requirement(raw)
    if req.marker and not req.marker.evaluate({'extra':''}):
        continue
    try:
        if metadata.version(req.name) in req.specifier:
            continue
    except metadata.PackageNotFoundError:
        pass
    candidates=[(version,path) for name,version,path in available if name==canonicalize_name(req.name) and version in req.specifier]
    # Generic pure-Python deps resolve through pip to respect Requires-Python.
    if candidates and (req.name.startswith('nvidia-') or req.name=='triton'):
        local.append(str(max(candidates)[1]))
    else:
        remote.append(str(req).split(';')[0].strip())
subprocess.run([sys.executable,'-m','pip','install','--no-deps',*local],check=True)
if remote:
    subprocess.run([sys.executable,'-m','pip','install',*remote],check=True)
