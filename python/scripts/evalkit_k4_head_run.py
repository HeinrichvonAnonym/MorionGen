"""Isolated head-task selector over frozen CPU candidates, with local reuse.

The old numerical implementation is imported unchanged. Registered code hashes
are reused without a tree hash scan; hashes for newly produced task files remain
local. Cached candidate receipts still undergo the original key equality check.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--code-root',type=Path,required=True);ap.add_argument('--job',type=Path,required=True)
    args=ap.parse_args();job=json.loads(args.job.read_text())
    sys.path[:0]=[str(args.code_root/'utils'),str(args.code_root/'utils/motiogen')]
    import opensim as osim
    osim.Logger.setLevelString('error')
    from motiogen.evalkit.k import config
    manifest=json.loads(Path(job['registered_identity']).read_text())
    known=manifest['identity'];original_digest=config.digest;lookups={};local=[]
    def registered_digest(path):
        p=str(Path(path).resolve())
        if p in known:
            lookups[p]=lookups.get(p,0)+1;return known[p]
        if Path(p).is_relative_to(args.code_root):
            raise ValueError('Unregistered frozen dependency; no hash scan allowed: '+p)
        local.append(p);return original_digest(path)
    config.digest=registered_digest
    from motiogen.evalkit.k import candidate_pool
    name='motiogen.evalkit.k.head_task_selection'
    spec=importlib.util.spec_from_file_location(name,job['selector_module'])
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    candidate_pool.assess=module.assess
    out=Path(job['output']);out.mkdir(parents=True,exist_ok=True)
    settings=json.loads(Path(job['fit_config']).read_text());settings['method']=job['method']
    if settings.get('selection_scope')!=module.VERSION:raise ValueError('Head selector version mismatch')
    reused=[]
    previous=Path(job['reuse_fit']) if job.get('reuse_fit') else None
    if previous and (previous/'candidates').exists():
        destination=out/'candidates';destination.mkdir(exist_ok=True)
        for source in (previous/'candidates').iterdir():
            if not (source/'receipt.json').exists() or not (source/'fit.npz').exists():continue
            target=destination/source.name
            if not target.exists():target.symlink_to(source.resolve(),target_is_directory=True)
            reused.append(source.name)
    # Preserve incomplete attempts without discarding finished siblings.
    for folder in (out/'candidates').glob('*'):
        if folder.is_dir() and not folder.is_symlink() and not (folder/'receipt.json').exists():
            folder.rename(folder.with_name(folder.name+'_interrupted_'+str(time.time_ns())))
    started=time.time();status={}
    try:
        value=candidate_pool.fit(job['input'],out,job['mass_kg'],job['floor_m'],settings)
        status=dict(status='completed',frames=len(value['times']),selector=module.VERSION)
    except Exception as error:
        import traceback
        status=dict(status='failed',error=str(error),traceback=traceback.format_exc())
    finally:
        status.update(elapsed_s=time.time()-started,reusable_complete_candidates=reused,
                      hash_policy='registered dependency hashes reused; no source/protection tree rehash',
                      registered_hash_cache_hits=sum(lookups.values()),local_task_hash_calls=len(local))
        (out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(status),flush=True)
    return 0 if status['status']=='completed' else 1


if __name__=='__main__':sys.exit(main())
