import datetime,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
import numpy as np
sys.path[:0]=['utils','utils/motiogen']
from motiogen.evalkit.k.config import implementation_key
from motiogen.evalkit.k.fitting import fit
ROOT=Path.cwd();data=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k_arms_20261116');out=ROOT/'config/evalkit_k_arms_20261116';out.mkdir(exist_ok=True)
read=lambda p:json.loads(Path(p).read_text())
hashfile=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
key=implementation_key();fits=[]
for folder,count in [('development/k2_adaptive',36),('development_first/k2_adaptive',30),('regression_k2_adaptive',8)]:
 receipts=list((data/folder).rglob('work/receipt.json'));assert len(receipts)==count,(folder,len(receipts))
 for p in receipts:
  r=read(p);assert r['k_implementation']==key,p
  fits.append({'path':str(p),'sha256':hashfile(p),'calibration':r['calibration'],'arm_selection':r.get('arm_selection'),'arm_quality':r['arm_quality']})
cache=[]
for r in read(data/'regression_k2_adaptive/summary.json'):
 assert r['status']=='computed' and r['residual_interface_finite']
 work=data/'regression_k2_adaptive'/r['method']/'work';before=(work/'coordinates.mot').stat().st_mtime_ns
 f=fit(r['source'],work,75.,0.);assert (work/'coordinates.mot').stat().st_mtime_ns==before
 with np.load(work.parent/'frames.npz') as frames:assert all('arm_'+s+'_fit_valid' in frames for s in 'lr')
 cache.append({'method':r['method'],'reused':True,'calibration_valid':f['calibration_valid']})
protected=[]
for prefix in ['config/evalkit_k_validation_20261116','config/gold_k_20261115','config/evalkit_k_v1.json']:
 names=subprocess.check_output(['git','ls-tree','-r','--name-only','2edc628','--',prefix],text=True).splitlines()
 for name in names:
  original=subprocess.check_output(['git','show','2edc628:'+name]);assert original==(ROOT/name).read_bytes(),name
  protected.append({'path':name,'sha256':hashfile(ROOT/name)})
tests=[]
for name,count in [('tests_final.log',30),('ergo_tests.log',8)]:
 p=ROOT/'develop_logs/test_runs/20261116_k_arms'/name;s=p.read_text();assert f'Ran {count} tests' in s and '\nOK\n' in s
 tests.append({'path':str(p),'tests':count,'sha256':hashfile(p),'passed':True})
cohorts={}
for folder in ['development','development_first']:
 manifest=read(data/folder/'manifest.json');assert all(r['partition']=='development' for r in manifest['records'])
 for version in ['k1','k2_adaptive']:
  rows=read(data/folder/(version+'_summary.json'));assert len(rows)==len(manifest['records']) and all(r['status']=='computed' for r in rows)
 compact={**manifest,'records':[{k:v for k,v in r.items() if k!='retarget_provenance'} for r in manifest['records']]}
 cohorts[folder]={'manifest':compact,'manifest_sha256':hashfile(data/folder/'manifest.json')}
sizes={p.name:sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) for p in data.iterdir() if p.is_dir()}
files=['utils/motiogen/evalkit/run.py','utils/motiogen/evalkit/ergo.py','utils/motiogen/upper_body.py','utils/motiogen/model_resources.py','config/biomechanics_models.json','python/scripts/evalkit_k.py','python/scripts/evalkit_k_arm_audit.py','python/scripts/evalkit_k_arm_report.py','python/tests/test_evalkit_k_arms.py']
record={'status':'passed','recorded_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'version':'k-2-arms-development','implementation':key,
 'integration_and_tools':{p:hashfile(p) for p in files},'tests':tests,'fit_cache':cache,'verified_fits':fits,'cohorts':cohorts,'protected_original_files':protected,
 'final_holdout_used_this_round':False,'baseline_checkout':str(data/'frozen_k1'),'baseline_commit':'acb48ac8cbd656a24b722a4b6fa2a01e0404bf9c',
 'fbls':read(data/'fbls_final_check.json'),'directory_bytes':sizes,'total_output_bytes':sum(sizes.values()),'disk_free_bytes':shutil.disk_usage(data).free,
 'history':'Final results are k2_adaptive only. Earlier range/registration, direction-only, calibration, and warm-start experiments are retained as superseded development iterations. Original frozen artifacts are byte-for-byte unchanged.'}
(out/'verification.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
for name,src in [('diagnosis.json',ROOT/'develop_logs/test_runs/20261116_k_arms/diagnosis.json'),('ablation.json',data/'ablation_v2/ablation.json'),('direction_experiment.json',ROOT/'develop_logs/test_runs/20261116_k_arms/directions_multistart.json')]:shutil.copy2(src,out/name)
print('PASS',len(fits),'fits',len(cache),'cache checks',sum(t['tests'] for t in tests),'tests','GB',sum(sizes.values())/1e9)
