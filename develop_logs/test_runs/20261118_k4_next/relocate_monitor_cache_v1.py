"""Relocate owned generated caches to data disk, preserving every byte and URI."""
import hashlib,json,shutil,time
from pathlib import Path
ROOT=Path('/home/heinrich/MorionGen')
B=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k4_20261117')
origin=ROOT/'develop_logs/test_runs/20261118_k4_monitor';destination=B/'monitor_cache_20261118';destination.mkdir(exist_ok=True)
manifest=destination/'relocation_v1.json'
if manifest.exists():raise ValueError('Preserve completed migration')
def inventory(path):
    data={}
    for p in sorted(path.rglob('*')):
        if p.is_symlink():data[str(p.relative_to(path))]={'symlink':str(p.readlink())}
        elif p.is_file():
            h=hashlib.sha256()
            with p.open('rb') as f:
                for block in iter(lambda:f.read(1024**2),b''):h.update(block)
            data[str(p.relative_to(path))]={'bytes':p.stat().st_size,'sha256':h.hexdigest()}
    return data
records=[]
for name in ['pool_candidates','composed_pool_candidates']:
    source=origin/name;target=destination/name;backup=origin/(name+'.relocation_backup')
    assert source.is_dir() and not source.is_symlink() and not target.exists() and not backup.exists()
    before=inventory(source);size=sum(x.get('bytes',0) for x in before.values())
    if shutil.disk_usage(B).free-size<80*1024**3:raise RuntimeError('80GiB reserve required')
    shutil.copytree(source,target,symlinks=True)
    assert inventory(target)==before and inventory(source)==before
    source.rename(backup)
    try:
        source.symlink_to(target,target_is_directory=True)
        assert inventory(source)==before
    except Exception:
        if source.is_symlink():source.unlink()
        backup.rename(source)
        raise
    # Remove only the redundant verified copy; original logical path resolves to
    # the data-disk copy with exactly the same contents and relative filenames.
    shutil.rmtree(backup)
    records.append({'logical_path':str(source),'physical_path':str(target),'bytes':size,'files':before,'verified_identical':True})
    print(name,len(before),size,flush=True)
manifest.write_text(json.dumps({'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'records':records,'data_lost':False,'source_strings_in_scores':'unchanged; old paths resolve via symlinks'},indent=2)+'\n')
(ROOT/'develop_logs/test_runs/20261118_k4_next/monitor_cache_relocation_v1.json').write_text(json.dumps({'manifest':str(manifest),'sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'caches':[{k:r[k] for k in ('logical_path','physical_path','bytes','verified_identical')} for r in records]},indent=2)+'\n')
