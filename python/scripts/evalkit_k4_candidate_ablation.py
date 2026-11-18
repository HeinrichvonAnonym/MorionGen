"""Exhaustive frozen candidate evaluation; reuse first, D-only fitting, R-only scoring."""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor,as_completed
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('candidate_accounting',ROOT/'utils/motiogen/evalkit/reporting/candidate_ablation.py')
account=importlib.util.module_from_spec(spec);spec.loader.exec_module(account)
read,sha,atomic=account.read,account.sha,account.atomic
ENV={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}


def frozen_imports(code):
    sys.path[:0]=[str(code/'utils'),str(code/'utils/motiogen')]


def prepare(a):
    if a.output.exists():raise ValueError('Use a new immutable registration directory')
    frozen=read(a.frozen_run/'protocol.json');code=Path(frozen['fit_code_root']);pool=read(frozen['fit_config'])
    for path,h in frozen['implementation_sha256'].items():
        if sha(path)!=h:raise ValueError('Frozen implementation changed: '+path)
    frozen_imports(code)
    from motiogen.evalkit.k.motion import load
    from motiogen.evalkit.k.candidate_eligibility import assess
    impl={str(p.relative_to(code/'utils/motiogen/evalkit/k')):sha(p) for p in (code/'utils/motiogen/evalkit/k').rglob('*.py')}
    templates={};policy=pool['coordinated_export_variants']
    for name,cfg in pool['candidates'].items():
        templates[name]=dict(base=name,kind='native',config=cfg)
        if cfg.get('coordinated_path')!=policy:
            templates[name+'__coordinated']=dict(base=name,kind='export',config=cfg,policy=policy)
    routes=[]
    for row in frozen['records']:
        job=read(row['job_file'])
        if sha(row['job_file'])!=row['job_sha256']:raise ValueError('Registered job changed')
        motion=load(job['input'],job['method'],config=code/'config/evalkit_k4_20261117/input.json')
        common=read(Path(frozen['baseline_root'])/'common_v1'/row['window']/row['method']/'receipt.json')['key']
        if common['inputs']!=motion.provenance['inputs']:raise ValueError('D identity differs from frozen baseline')
        expected={k:common[k] for k in ('inputs','mass_kg','floor_m','model_sha256','input_config_sha256','biomech_sha256')}
        if expected['mass_kg']!=job['mass_kg'] or expected['floor_m']!=job['floor_m']:raise ValueError('Body conditions changed')
        cells={}
        for name,t in templates.items():
            applies,why=assess(motion,t['base'],pool['candidates'],pool.get('candidate_requirements',{}))
            cell=dict(status='applicable' if applies else 'inapplicable',reason=why)
            if why.get('reason')=='same_fixed_geometry_as_retained_base':
                alias=why['base_candidate']+('__coordinated' if t['kind']=='export' else '')
                cell.update(status='equivalent',alias=alias)
            cells[name]=cell
        routes.append(dict(window=row['window'],method=row['method'],input=job['input'],expected_key=expected,cells=cells))
    if len(routes)!=656:raise ValueError('Keep the frozen 656 denominator')
    protocol=dict(version='k4-exhaustive-candidate-ablation-1',frozen_run=str(a.frozen_run.resolve()),
        frozen_protocol_sha256=sha(a.frozen_run/'protocol.json'),code_root=str(code),
        baseline_root=frozen['baseline_root'],preservation_manifest=frozen['preservation_manifest'],
        workspace_root=frozen['workspace_root'],implementation_sha256=frozen['implementation_sha256'],
        numerical_implementation=impl,policy_sha256=sha(code/'config/evalkit_k4_20261117/strict_v1/acceptance.json'),
        templates=templates,fit_roots=[str(a.frozen_run/'fits'),*map(str,a.reuse_fit_root)],
        routes=routes,total_worker_limit=14,ablation_workers=3,minimum_free_GiB=80,
        selection_scope='Read frozen runtime selection, never rerank with R. No early acceptance in the candidate matrix.',
        equivalence_scope='Only input-proven exact fixed-geometry alternatives share evidence; numerical implementation otherwise pinned.',
        new_holdout_uses=0)
    a.output.mkdir(parents=True);atomic(a.output/'protocol.json',protocol)
    for name,cfg in pool['candidates'].items():atomic(a.output/'configs'/(name+'.json'),cfg)
    print(json.dumps(dict(output=str(a.output),routes=len(routes),declared_columns=len(templates),
        applicable_cells=sum(c['status']!='inapplicable' for r in routes for c in r['cells'].values()))),flush=True)


def variant(a):
    code=Path(a.code_root);frozen_imports(code)
    import opensim as osim
    osim.Logger.setLevelString('error')
    from motiogen.evalkit.k.motion import load
    from motiogen.evalkit.k.path_variant import build
    job=read(a.job)
    if set(job)!={'input','method','source','output','policy'}:raise ValueError('Unexpected fitting input field')
    motion=load(job['input'],job['method'],config=code/'config/evalkit_k4_20261117/input.json')
    build(motion,job['source'],job['output'],job['policy'])


class Experiment:
    def __init__(self,run):
        self.run=Path(run);self.p=read(self.run/'protocol.json');self.code=Path(self.p['code_root'])
        self.frozen=Path(self.p['frozen_run']);self.templates=self.p['templates']
        self.policy=read(self.code/'config/evalkit_k4_20261117/strict_v1/acceptance.json')
        # Import acceptance by source file, with no reference objects sent to fit children.
        s=importlib.util.spec_from_file_location('frozen_acceptance',self.code/'utils/motiogen/evalkit/k/strict_acceptance.py')
        m=importlib.util.module_from_spec(s);s.loader.exec_module(m);self.assess=m.assess
        self.states={};self.selectors={}
        self.routes={r['window']+'_'+r['method']:r for r in self.p['routes']}
        for key in self.routes:
            for path in (self.run/'cells'/key).glob('*/state.json'):
                state=read(path)
                if state.get('status')=='evaluated':
                    fit=Path(state['fit'])
                    checks={**{str(fit/n):h for n,h in state['source_sha256'].items()},
                        str(fit/'receipt.json'):state['receipt_sha256'],**state['score_sha256'],
                        state['actual_export_audit_file']:state['actual_export_audit_sha256']}
                    if state['policy_sha256']!=self.p['policy_sha256'] or any(sha(p)!=h for p,h in checks.items()):
                        raise ValueError('Saved independent candidate evidence changed: '+str(path))
                self.states[(key,path.parent.name)]=state
        self.refresh_selectors()
        self.audit_index={}
        index=self.run/'audit_reuse_index.json'
        if index.exists():
            self.audit_index=read(index)['by_artifact_digest']

    def historical_review(self,route,fit,fingerprints):
        signature=hashlib.sha256(json.dumps(fingerprints,sort_keys=True).encode()).hexdigest()
        selector_task=self.frozen/'tasks'/(route['window']+'_'+route['method'])
        if not (selector_task/'result.json').exists():return None
        if read(selector_task/'result.json').get('status')!='completed':return None
        current=read(fit/'receipt.json')
        for item in self.audit_index.get(signature,[]):
            if (item['window'],item['method'])!=(route['window'],route['method']):continue
            previous=read(item['source_receipt'])
            if previous['key']['inputs']!=current['key']['inputs'] or previous.get('clock_origin_s')!=current.get('clock_origin_s'):continue
            audit=read(item['audit_file'])
            checks={**audit['source_sha256'],**audit['score_sha256'],audit['actual_export_audit_file']:audit['actual_export_audit_sha256']}
            if any(not Path(p).exists() or sha(p)!=h for p,h in checks.items()):continue
            scores={};same_reference=True
            for path in audit['score_sha256']:
                view=Path(path).parent.name;scores[view]=Path(path)
                old=read(path);now=read(self.frozen/'scores'/view/(route['window']+'_'+route['method']+'.json'))
                if old['original_reference_file']!=now['original_reference_file']:same_reference=False
                for left,right in [(old,now),(old['original_clock'],now['original_clock'])]:
                    if left['comparisons']['D-R']!=right['comparisons']['D-R']:same_reference=False
            if not same_reference or set(scores)!={'native','export'}:continue
            result=self.score_summary(fit,fingerprints,scores,Path(audit['actual_export_audit_file']),'identical_artifacts_and_verified_historical_audit')
            result['reused_audit_file']=item['audit_file'];result['audit_equivalence_sha256']=read(self.run/'audit_reuse_index.json')['proof_sha256']
            return result

    def verify(self):
        if sha(self.frozen/'protocol.json')!=self.p['frozen_protocol_sha256']:raise ValueError('Frozen protocol changed')
        for path,h in self.p['implementation_sha256'].items():
            if sha(path)!=h:raise ValueError('Frozen implementation changed: '+path)
        manifest=read(self.p['preservation_manifest']);root=Path(self.p['workspace_root'])
        changed=[n for n,h in manifest['files'].items() if sha(root/n)!=h]
        if changed:raise ValueError('Protected files changed: '+str(changed))
        atomic(self.run/'preservation_latest.json',dict(checked=len(manifest['files']),changed=changed,new_holdout_uses=0))
        for route in self.p['routes']:
            for path,h in route['expected_key']['inputs'].items():
                if sha(path)!=h:raise ValueError('Registered D input changed: '+path)

    def refresh_selectors(self):
        for key in self.routes:
            path=self.frozen/'tasks'/key/'result.json'
            if path.exists():
                d=read(path);self.selectors[key]={k:d.get(k) for k in ('status','passed','selected','error')}

    def snapshot(self):
        self.refresh_selectors();result=account.summarize(self.p['routes'],self.states,self.selectors)
        result.update(updated_unix_s=time.time(),protocol_sha256=sha(self.run/'protocol.json'))
        atomic(self.run/'progress.json',result);return result

    def expected(self,route):return dict(route['expected_key'],implementation=self.p['numerical_implementation'])

    def sources(self,route,name):
        yield self.run/'fits'/route['window']/route['method']/name
        for root in self.p['fit_roots']:
            yield Path(root)/route['window']/route['method']/'candidates'/name

    def find_fit(self,route,name):
        t=self.templates[name];config=dict(t['config'],method=route['method']);rejections=[]
        for folder in self.sources(route,name):
            file=folder/'receipt.json'
            if not file.exists():continue
            try:
                receipt=read(file)
                ok,reason=account.compatible(receipt['key'],self.expected(route),config,t.get('policy'))
                if not ok:rejections.append(dict(source=str(folder),reason=reason));continue
                if any(not (folder/n).exists() for n in account.ARTIFACTS):continue
                return folder,rejections
            except (OSError,ValueError,KeyError):continue
        return None,rejections

    def call(self,command,dest,label):
        if shutil.disk_usage(self.run).free<80*1024**3:raise RuntimeError('80 GiB reserve reached')
        with (dest/(label+'.log')).open('w') as log:
            subprocess.run([sys.executable,*map(str,command)],env=ENV,stdout=log,stderr=subprocess.STDOUT,check=True)

    def evaluate(self,route,name,fit,dest):
        w,m=route['window'],route['method'];key=w+'_'+m
        fingerprints={n:sha(fit/n) for n in account.ARTIFACTS}
        previous=self.historical_review(route,fit,fingerprints)
        if previous is not None:return previous
        # The selected candidate already has exactly this independent acceptance.
        # Reuse requires byte-identical numerical outputs, not a matching name.
        result=self.frozen/'tasks'/key/'result.json';parent=self.frozen/'fits'/w/m
        if result.exists() and read(result).get('status')=='completed' and read(result).get('selected')==name:
            if all((parent/n).exists() and sha(parent/n)==h for n,h in fingerprints.items()):
                task=result.parent;scores={v:self.frozen/'scores'/v/(key+'.json') for v in ('native','export')}
                audit=task/'actual_export/audit.json'
                return self.score_summary(fit,fingerprints,scores,audit,'identical_frozen_selected_output')
        views=dest/'views'
        self.call([self.code/'python/scripts/evalkit_k4_native_export_views.py','--code-root',self.code,
            '--fit',fit,'--input',route['input'],'--method',m,'--window',w,'--output',views],dest,'views')
        scores={}
        for view in ('native','export'):
            self.call([self.code/'python/scripts/evalkit_k4_ablation_score.py','--fits',views/view,
                '--baseline',Path(self.p['baseline_root'])/'common_v1','--output',dest/'scores'/view,
                '--window',w,'--method',m,'--strict'],dest,'score_'+view)
            scores[view]=dest/'scores'/view/(key+'.json')
        self.call([self.code/'python/scripts/evalkit_k4_export_audit.py','--fit',fit,'--output',dest/'actual_export'],dest,'actual_export')
        if any(sha(fit/n)!=h for n,h in fingerprints.items()):raise ValueError('Source changed during independent review')
        return self.score_summary(fit,fingerprints,scores,dest/'actual_export/audit.json','new_independent_review')

    def score_summary(self,fit,fingerprints,scores,audit,reuse):
        acceptance={};failures=[];score_hashes={};degradation={}
        for view,path in scores.items():
            data=read(path);score_hashes[str(path)]=sha(path);acceptance[view]={}
            for clock,values in [('D_clock',data),('R_clock',data['original_clock'])]:
                assessed=self.assess(values,None,self.policy);acceptance[view][clock]=assessed['status']
                failures.extend(dict(view=view,clock=clock,**x) for x in assessed['required']+assessed['temporal']+assessed['full_window_coverage'] if x['status'] not in ('passed','unobservable'))
                degradation[view+'/'+clock]=hashlib.sha256(json.dumps(values['comparisons']['D-R'],sort_keys=True).encode()).hexdigest()
        export=read(audit)
        return dict(status='evaluated',passed=export['passed'] and all(v=='passed' for clocks in acceptance.values() for v in clocks.values()),
            fit=str(fit),source_sha256=fingerprints,receipt_sha256=sha(fit/'receipt.json'),score_sha256=score_hashes,
            acceptance=acceptance,failures=failures,actual_export_passed=export['passed'],
            actual_export_audit_file=str(audit),actual_export_audit_sha256=sha(audit),reuse=reuse,
            degradation_sha256=degradation,policy_sha256=self.p['policy_sha256'])

    def run_route(self,key,audit_only):
        route=self.routes[key];w,m=route['window'],route['method']
        for name,contract in route['cells'].items():
            if contract['status']=='inapplicable' or contract.get('alias'):continue
            if (self.run/'pause_requested').exists():break
            old=self.states.get((key,name),{})
            if old.get('status') in ('evaluated','execution_failed'):continue
            dest=self.run/'cells'/key/name;dest.mkdir(parents=True,exist_ok=True)
            fit,rejections=self.find_fit(route,name);t=self.templates[name]
            # Never compete with the frozen selector for a not-yet-finished route.
            if fit is None and (audit_only or (key not in self.selectors and not getattr(self,'independent_after_selector_stop',False))):continue
            attempt=dest/('attempt_'+str(time.time_ns()));attempt.mkdir()
            try:
                if fit is None:
                    if t['kind']=='native':
                        fit=self.run/'fits'/w/m/name
                        if fit.exists():
                            history=dest/'attempt_history';history.mkdir(exist_ok=True)
                            fit.rename(history/str(time.time_ns()))
                        job=dict(input=route['input'],method=m,output=str(fit),mass_kg=route['expected_key']['mass_kg'],
                            floor_m=route['expected_key']['floor_m'],fit_config=str(self.run/'configs'/(name+'.json')))
                        atomic(attempt/'runtime_job.json',job)
                        self.call([self.code/'python/scripts/evalkit_k4_run.py','--code-root',self.code,'--job',attempt/'runtime_job.json'],attempt,'fit')
                    else:
                        base,_=self.find_fit(route,t['base'])
                        if base is None:
                            if self.states.get((key,t['base']),{}).get('status')=='execution_failed':
                                raise RuntimeError('Parent native candidate execution failed; inherited export unavailable')
                            continue
                        fit=self.run/'fits'/w/m/name
                        if fit.exists():
                            history=dest/'attempt_history';history.mkdir(exist_ok=True);fit.rename(history/str(time.time_ns()))
                        atomic(attempt/'runtime_job.json',dict(input=route['input'],method=m,source=str(base),output=str(fit),policy=t['policy']))
                        self.call([__file__,'variant','--code-root',self.code,'--job',attempt/'runtime_job.json'],attempt,'export_variant')
                    verified,_=self.find_fit(route,name)
                    if verified!=fit:raise ValueError('New candidate does not satisfy registered numerical contract')
                state=self.evaluate(route,name,fit,attempt)
                # D-R must be identical across candidates in each view/clock.
                existing=next((v for (rk,_),v in list(self.states.items()) if rk==key and v.get('status')=='evaluated'),None)
                if existing and state['degradation_sha256']!=existing['degradation_sha256']:
                    raise ValueError('D-R differs between candidates')
            except Exception as error:
                import traceback
                state=dict(status='execution_failed',passed=None,error=str(error),traceback=traceback.format_exc())
            state.update(candidate=name,window=w,method=m,attempt=str(attempt),rejected_caches=rejections)
            atomic(dest/'state.json',state);self.states[(key,name)]=state
            print(key,name,state['status'],state.get('passed'),flush=True)


def index_audits(a):
    if (a.run/'identity.json').exists():raise ValueError('Register audit reuse before starting the experiment')
    target=a.run/'audit_reuse_index.json'
    if target.exists():raise ValueError('Audit reuse index is immutable')
    proof=read(a.proof);protocol=read(a.run/'protocol.json')
    for name,item in proof['files'].items():
        hashes=[sha(path) for path in item['source_sha256']]
        if ((name!='utils/motiogen/evalkit/k/k4_fitting.py' and len(set(hashes))!=1)
                or any(sha(p)!=h for p,h in item['source_sha256'].items())):
            raise ValueError('Historical audit implementation equivalence changed')
    for name in ('model_fk','write_coordinates'):
        bodies=[]
        for item in proof[name].values():
            tree=ast.parse(Path(item['source']).read_text())
            node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
            bodies.append(ast.dump(node))
        if len(set(bodies))!=1:raise ValueError('Audit FK helper differs')
    by_digest={};count=0
    for cache in a.cache:
        if read(cache/'review.json')['policy_sha256']!=protocol['policy_sha256']:raise ValueError('Old audit policy differs')
        for file in sorted(cache.glob('*/*/audit_summary.json')):
            d=read(file);sources={Path(p).name:p for p in d['source_sha256']}
            if any(n not in sources for n in (*account.ARTIFACTS,'receipt.json')):continue
            checks={**d['source_sha256'],**d['score_sha256'],d['actual_export_audit_file']:d['actual_export_audit_sha256']}
            if any(not Path(p).exists() or sha(p)!=h for p,h in checks.items()):raise ValueError('Old audit cache changed: '+str(file))
            fingerprints={n:d['source_sha256'][sources[n]] for n in account.ARTIFACTS}
            signature=hashlib.sha256(json.dumps(fingerprints,sort_keys=True).encode()).hexdigest()
            w,m=file.parent.parent.name.split('_',1)
            by_digest.setdefault(signature,[]).append(dict(window=w,method=m,audit_file=str(file),source_receipt=sources['receipt.json']))
            count+=1
    atomic(target,dict(proof=str(a.proof),proof_sha256=sha(a.proof),reviewed_historical_entries=count,
        by_artifact_digest=by_digest,scope='Scoring reuse only after exact numeric artifacts, inputs, receipt clock, frozen policy, same R path and complete D-R equality against current selector. This does not promote an old fitting implementation.'))
    print(json.dumps(dict(historical_entries=count,unique_artifact_sets=len(by_digest))),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    prepare_p=sub.add_parser('prepare');prepare_p.add_argument('--frozen-run',type=Path,required=True)
    prepare_p.add_argument('--output',type=Path,required=True);prepare_p.add_argument('--reuse-fit-root',type=Path,action='append',default=[])
    v=sub.add_parser('variant');v.add_argument('--code-root',type=Path,required=True);v.add_argument('--job',type=Path,required=True)
    i=sub.add_parser('index-audits');i.add_argument('--run',type=Path,required=True);i.add_argument('--proof',type=Path,required=True)
    i.add_argument('--cache',type=Path,action='append',required=True)
    for name in ('run','snapshot'):
        q=sub.add_parser(name);q.add_argument('--run',type=Path,required=True)
        if name=='run':
            q.add_argument('--workers',type=int,default=1);q.add_argument('--audit-only',action='store_true');q.add_argument('--once',action='store_true')
            q.add_argument('--controller-amendment',type=Path)
            q.add_argument('--independent-after-selector-stop',action='store_true')
    a=p.parse_args()
    if a.command=='prepare':return prepare(a)
    if a.command=='variant':return variant(a)
    if a.command=='index-audits':return index_audits(a)
    e=Experiment(a.run)
    if a.command=='snapshot':e.snapshot();return
    if not 1<=a.workers<=14:raise ValueError('Total numerical task budget fourteen')
    lock=(a.run/'.controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if a.independent_after_selector_stop:
        # Hold the old controller lock throughout: no competing selector restart.
        selector_lock=(e.frozen/'.controller.lock').open('a')
        fcntl.flock(selector_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if read(e.frozen/'progress.json').get('state')!='validation_stage_failed':
            raise ValueError('Independent scheduling requires the preserved failed selector gate')
        e.independent_after_selector_stop=True
    elif a.workers>3:
        raise ValueError('Expanded allocation requires exclusive stopped-selector ownership')
    e.verify()
    identity=dict(runner_sha256=sha(__file__),accounting_sha256=sha(ROOT/'utils/motiogen/evalkit/reporting/candidate_ablation.py'),protocol_sha256=sha(a.run/'protocol.json'),
        audit_reuse_index_sha256=sha(a.run/'audit_reuse_index.json') if (a.run/'audit_reuse_index.json').exists() else None)
    identity_file=a.run/'identity.json'
    previous=read(identity_file) if identity_file.exists() else None
    epochs=sorted((a.run/'execution_epochs').glob('*.json'))
    effective=read(epochs[-1])['execution_identity'] if epochs else previous
    if effective and effective!=identity:
        if a.controller_amendment is None:raise ValueError('Explicit controller-only amendment required')
        amendment=read(a.controller_amendment);old=Path(amendment['previous_runner'])
        if sha(old)!=effective['runner_sha256'] or amendment['new_runner_sha256']!=identity['runner_sha256']:
            raise ValueError('Controller amendment identity mismatch')
        if {k:v for k,v in effective.items() if k!='runner_sha256'}!={k:v for k,v in identity.items() if k!='runner_sha256'}:
            raise ValueError('Controller amendment cannot change protocol, accounting or reuse contracts')
        def without_main(path):
            tree=ast.parse(Path(path).read_text())
            tree.body=[n for n in tree.body if not(isinstance(n,ast.FunctionDef) and n.name=='main')]
            # The sole candidate-loop change releases scheduling dependency;
            # fitting, cache validation, scoring and acceptance remain exact.
            before=ast.dump(ast.parse('fit is None and (audit_only or key not in self.selectors)',mode='eval').body)
            after=ast.parse("fit is None and (audit_only or (key not in self.selectors and not getattr(self,'independent_after_selector_stop',False)))",mode='eval').body
            class SchedulingPredicate(ast.NodeTransformer):
                def visit_If(self,node):
                    if ast.dump(node.test)==before:node.test=after
                    return self.generic_visit(node)
            tree=SchedulingPredicate().visit(tree)
            return ast.dump(tree)
        if without_main(old)!=without_main(__file__):raise ValueError('Controller amendment changed candidate execution or scoring')
        atomic(a.run/'execution_epochs'/(str(time.time_ns())+'.json'),dict(original_identity=previous,
            execution_identity=identity,amendment=amendment,workers=a.workers,
            scope='Scheduling allocation and exclusive stopped-selector independence; fitting/reuse/scoring unchanged.'))
    elif previous is None:atomic(identity_file,identity)
    while True:
        e.snapshot()
        # Existing candidate bodies are reviewed first; no new fit required.
        keys=list(e.routes);keys.sort(key=lambda k:(not(e.selectors.get(k,{}).get('passed') is False),k not in e.selectors,k))
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            from concurrent.futures import wait,FIRST_COMPLETED
            pending={pool.submit(e.run_route,key,a.audit_only) for key in keys}
            published=-1;published_at=0
            while pending:
                done,pending=wait(pending,timeout=5,return_when=FIRST_COMPLETED)
                for f in done:f.result()
                if (len(e.states)!=published and time.time()-published_at>=5) or time.time()-published_at>=30:
                    e.snapshot();published=len(e.states);published_at=time.time()
        s=e.snapshot()
        if s['fully_evaluated_routes']==656:
            e.verify();atomic(a.run/'completion.json',dict(fully_evaluated_routes=656,all_candidates_passed_routes=s['all_candidates_passed_routes'],new_holdout_uses=0));break
        if s['pending_cells']==0 and s['execution_failed_cells']:
            atomic(a.run/'execution_failures.json',dict(count=s['execution_failed_cells'],scope='Preserved numerical or scoring execution failures require diagnosis; not accuracy failures.'))
            break
        if a.once or a.audit_only or (a.run/'pause_requested').exists():break
        time.sleep(15)


if __name__=='__main__':main()
