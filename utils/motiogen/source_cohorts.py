"""Analysis-only inference source contract, distinct from repository inventory.

Whole modules are conservative: a hash change does not prove a change in a
particular branch. Use fixed-source replays instead of guessing old contents.
"""
COMMON=['utils/motiogen/motion.py','utils/motiogen/metrics.py','utils/motiogen/feedback.py']
ADAPTER=['python/scripts/baseline_worker.py','utils/motiogen/adapters.py',
         'utils/motiogen/interventions.py','utils/motiogen/local_llm.py',
         'utils/motiogen/sampler_timing.py']
SOURCES={m:COMMON+ADAPTER for m in ['motionlcm','omnicontrol','chois','ardy']}
SOURCES['ardy']=SOURCES['ardy']+['utils/motiogen/ardy_native.py','utils/motiogen/prompt_cache.py','utils/motiogen/prompt_cache_review.py']
SOURCES['chois']=SOURCES['chois']+['utils/motiogen/chois_compat.py']
SOURCES['dart']=COMMON+['python/scripts/dart_feedback_smoke.py','utils/motiogen/dart_native.py']
SOURCES['closd']=COMMON+['python/scripts/closd_native.py','utils/motiogen/closd_native.py']
SOURCES['primal']=COMMON+['python/scripts/primal_feedback_smoke.py','utils/motiogen/primal_interventions.py']

def archive_sources(root,directory,hashes):
    """Preserve actual frozen contents; never overwrite a differing archive."""
    import hashlib
    from pathlib import Path
    root=Path(root);directory=Path(directory)
    records=[]
    for relative,expected in hashes.items():
        content=(root/relative).read_bytes()
        if hashlib.sha256(content).hexdigest()!=expected:
            raise ValueError('Source changed before archival: '+relative)
        target=directory/relative
        if target.exists() and target.read_bytes()!=content:
            raise ValueError('Refusing to overwrite frozen source: '+relative)
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():target.write_bytes(content)
        records.append({'path':relative,'sha256':expected,'bytes':len(content)})
    return records
