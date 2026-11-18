"""Resume bookkeeping for frozen experiments; preserves every prior attempt."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(output, code, config, base, identity, protocol, panel):
    old = json.loads((output/'identity.json').read_text())
    for field in ('code_commit', 'config_sha256', 'new_holdout_uses'):
        if old[field] != identity[field]:
            raise ValueError('Resume identity mismatch: '+field)
    subprocess.run(['git','-C',str(code),'diff','--exit-code','HEAD','--'],check=True,stdout=subprocess.DEVNULL)
    sys.path.insert(0,str(code/'utils/motiogen'))
    import biomech
    if json.loads((output/'protocol.json').read_text()) != protocol or json.loads((output/'panel.json').read_text()) != panel:
        raise ValueError('Resume protocol/panel mismatch')
    epoch = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    manifest = dict(epoch=epoch, original_identity=old, execution_identity=identity,
                    reused=[], restarted=[], origin='Explicit resume; fitting code and scoring remain frozen')
    reused = []
    restart = []
    for row in panel['records']:
        w,m=row['window'],row['method'];key=w+'_'+m
        common=json.loads((base/'common_v1'/w/m/'receipt.json').read_text())['key']
        if sha(biomech.MODEL)!=common['model_sha256']:raise ValueError('Frozen model asset changed')
        for path,digest in common['inputs'].items():
            if sha(path)!=digest:raise ValueError('Frozen input changed: '+path)
        for condition in protocol['conditions']:
            task=output/'tasks'/condition/key;fit=output/'fits'/condition/w/m
            status=task/'status.json'
            result=json.loads(status.read_text()) if status.exists() else None
            if result and result['status']=='completed':
                saved=json.loads((fit/'receipt.json').read_text())['key']
                expected_config=json.loads((config/(condition+'.json')).read_text());expected_config['method']=m
                original=json.loads((base/'jobs'/(key+'.json')).read_text())
                if saved['config']!=expected_config or any(saved[k]!=original[k] for k in ('mass_kg','floor_m')):
                    raise ValueError('Completed fit configuration changed')
                if saved['inputs']!=common['inputs'] or saved['model_sha256']!=common['model_sha256']:
                    raise ValueError('Completed fit identity mismatch: '+key)
                for name,digest in saved['implementation'].items():
                    if sha(code/'utils/motiogen/evalkit/k'/name)!=digest:raise ValueError('Fitting implementation changed')
                if sha(code/'utils/motiogen/biomech.py')!=saved['biomech_sha256']:raise ValueError('Biomech changed')
                audit=json.loads((task/'actual_export/audit.json').read_text())
                for name,digest in audit['source_sha256'].items():
                    if sha(fit/name)!=digest:raise ValueError('Completed artifact changed: '+str(fit/name))
                for view in ('native','export'):
                    score=output/'scores'/condition/view/(key+'.json')
                    data=json.loads(score.read_text())
                    if not {'comparisons','original_clock'} <= data.keys():raise ValueError('Incomplete scoring')
                reused.append(result);manifest['reused'].append(dict(key=key,condition=condition,status_sha256=sha(status)))
            else:
                restart.append((row,condition))
    # Validate every complete record before moving any interrupted artifacts.
    for row,condition in restart:
        w,m=row['window'],row['method'];key=w+'_'+m
        paths=[output/'tasks'/condition/key,output/'fits'/condition/w/m,output/'views'/condition/w/m]
        paths += [output/'scores'/condition/view/(key+'.json') for view in ('native','export')]
        archived=[]
        for source in paths:
            if source.exists():
                target=output/'attempt_history'/epoch/source.relative_to(output)
                target.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(source),str(target))
                archived.append(str(target.relative_to(output)))
        manifest['restarted'].append(dict(key=key,condition=condition,archived=archived))
    epochs=output/'execution_epochs';epochs.mkdir(exist_ok=True)
    (epochs/(epoch+'.json')).write_text(json.dumps(manifest,indent=2)+'\n')
    return reused,restart
