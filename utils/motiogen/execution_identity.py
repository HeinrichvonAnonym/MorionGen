"""Separate inference-resume identity from reporting/analysis revisions.

The command's local Python import closure is execution code. Unrelated report
builders are not. A full per-run source inventory remains in Run provenance.
"""
import ast
import hashlib
import json
import subprocess
from pathlib import Path
from .config import ROOT,repo_root
from .runtime import checksum

REPORT_ONLY_CONTRACTS={'motiongen_survey_methods.json','survey_migration.json',
                      'survey_published_evidence.json','native_metric_contract.json'}


def framework_sources(plan,root=ROOT):
    root=Path(root).resolve()
    pending=[root/'utils/motiogen/experiment_queue.py',root/'utils/motiogen/runtime.py']
    for job in plan['jobs']:
        for argument in job['command']:
            path=Path(argument)
            if path.suffix=='.py' and '{' not in argument:
                path=path if path.is_absolute() else root/path
                if path.is_file() and path.resolve().is_relative_to(root):pending.append(path.resolve())
    visited=set()

    def candidates(module):
        path=root/'utils'/module.replace('.','/')
        return [path.with_suffix('.py'),path/'__init__.py']

    while pending:
        path=pending.pop()
        if path in visited or not path.is_file():continue
        visited.add(path)
        tree=ast.parse(path.read_text(),filename=str(path))
        for node in ast.walk(tree):
            modules=[]
            if isinstance(node,ast.Import):modules=[a.name for a in node.names if a.name.startswith('motiogen')]
            elif isinstance(node,ast.ImportFrom):
                if node.level:
                    parent=path.parent
                    for _ in range(node.level-1):parent=parent.parent
                    if not parent.is_relative_to(root/'utils'):continue
                    prefix='.'.join(parent.relative_to(root/'utils').parts)
                    module=prefix+('.'+node.module if node.module else '')
                elif node.module and node.module.startswith('motiogen'):module=node.module
                else:continue
                modules=[module]+[module+'.'+a.name for a in node.names]
            for module in modules:pending.extend(p for p in candidates(module) if p.is_file())
    return sorted(visited)


def contract_values(root=ROOT):
    contracts={}
    for path in sorted((Path(root)/'config').rglob('*.json')):
        if path.name in REPORT_ONLY_CONTRACTS:continue
        key=str(path.relative_to(root))
        if path.name=='native_component_experiments.json':
            contracts[key]={'artifact_policy':json.loads(path.read_text()).get('artifact_policy')}
        else:contracts[key]=checksum(path)
    return contracts


def identity_payload(plan,cfg,root=ROOT):
    from .weight_identity import snapshot as weight_snapshot
    from .preprocessing_identity import snapshot as preprocessing_snapshot
    native={};dependencies={}
    for job in plan['jobs']:
        method=job['method']
        if method in native:continue
        local={**cfg,'baselines':{**cfg['baselines']}}
        if method not in local['baselines']:local['baselines'][method]=job['method_spec']
        repo=repo_root(local,method)
        def git(*args):
            return subprocess.check_output(['git','-C',str(repo),*args])
        untracked=git('ls-files','--others','--exclude-standard','-z').decode().split('\0')
        source_paths=[repo/name for name in untracked if name and Path(name).suffix in {'.py','.yaml','.yml','.json','.cpp','.h','.cu'}]
        native[method]={'commit':git('rev-parse','HEAD').decode().strip(),
            'tracked_diff_sha256':hashlib.sha256(git('diff','--binary','HEAD')).hexdigest(),
            'untracked_source_sha256':{str(p.relative_to(repo)):checksum(p) for p in source_paths if p.is_file()}}
        dependencies[method]={'weights':weight_snapshot(local,method)['identity'],
                              'preprocessing':preprocessing_snapshot(local,method)['identity']}
    return {'schema':2,'plan':plan,'config':cfg,
        'sources':{str(p.relative_to(root)):checksum(p) for p in framework_sources(plan,root)},
        'contracts':contract_values(root),
        'native_source_versions':native,
        'registered_inference_dependencies':dependencies,
        'artifacts':{str(p):checksum(p) for p in map(Path,plan.get('identity_files',[]))},
        'scope':'Inference command/import closure and execution contracts; presentation and derived-analysis revisions excluded. Raw inputs, seeds, weights and inference changes still invalidate resume.'}
