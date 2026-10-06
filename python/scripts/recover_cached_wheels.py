"""Reuse cached wheel bytes without touching existing environments or pip cache."""
import json
import os
import re
import tarfile
import sys
import zipfile
from email.parser import Parser
from pathlib import Path

root=Path(__file__).resolve().parents[2]
target=root/'outputs/assets/wheels'; target.mkdir(parents=True,exist_ok=True)
inventory=[]
for file in (Path.home()/'.cache/pip/http-v2').rglob('*.body'):
    if not zipfile.is_zipfile(file):
        if file.stat().st_size<10*1024**2:
            with file.open('rb') as handle:
                gzip_magic=handle.read(2)==b'\x1f\x8b'
            if gzip_magic:
                try:
                    with tarfile.open(file,'r:gz') as archive:
                        info=next((x for x in archive.getmembers() if x.name.count('/')==1 and x.name.endswith('/PKG-INFO')),None)
                        if info:
                            meta=Parser().parsestr(archive.extractfile(info).read().decode())
                            name=meta['Name'].replace('-','_'); version=meta['Version']
                            if re.fullmatch(r'[A-Za-z0-9_]+',name) and re.fullmatch(r'[A-Za-z0-9.+_]+',version):
                                destination=target/f'{name}-{version}.tar.gz'
                                if not destination.exists():
                                    os.link(file,destination)
                                inventory.append({'archive':destination.name,'source_cache':str(file),'bytes':file.stat().st_size,'reused_without_source_mutation':True})
                except (tarfile.TarError,UnicodeError):
                    pass
        continue
    with zipfile.ZipFile(file) as archive:
        entries=[x for x in archive.namelist() if x.endswith('.dist-info/WHEEL')]
        if len(entries)!=1:
            continue
        folder=entries[0].rsplit('/',1)[0]
        meta=Parser().parsestr(archive.read(folder+'/METADATA').decode())
        name=meta['Name'].replace('-','_'); version=meta['Version']
        if not re.fullmatch(r'[A-Za-z0-9_]+',name) or not re.fullmatch(r'[A-Za-z0-9.+_]+',version):
            continue
        tags=Parser().parsestr(archive.read(entries[0]).decode()).get_all('Tag')
        if not tags:
            continue
        tag=tags[0]
        if 'py2-none-any' in tags and 'py3-none-any' in tags:
            tag='py2.py3-none-any'
        destination=target/f'{name}-{version}-{tag}.whl'
        if not destination.exists():
            os.link(file,destination)
        inventory.append({'wheel':destination.name,'source_cache':str(file),'bytes':file.stat().st_size,'reused_without_source_mutation':True})
(target/'cache_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
print(json.dumps({'wheels':len(inventory),'path':str(target)}))
