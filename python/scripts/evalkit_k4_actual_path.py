"""Fixed-native export-path check on a registered real generated motion (F-D only)."""
import argparse,hashlib,json,os,shutil,subprocess,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',type=Path,required=True);a=parser.parse_args()
    p=json.loads(a.protocol.read_text());out=Path(p['output']);code=Path(p['code_root'])
    if subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD']).decode().strip()!=p['code_commit']:raise ValueError('Frozen runtime identity changed')
    sources={**p['source_sha256'],p['policy']:p['policy_sha256']}
    if any(digest(path)!=value for path,value in sources.items()):raise ValueError('Source changed before fixed-native experiment')
    if shutil.disk_usage(out.parent).free<80*2**30:raise RuntimeError('80GiB reserve required')
    out.mkdir(exist_ok=False);shutil.copyfile(a.protocol,out/'protocol.json')
    scorescript=ROOT/'python/scripts/evalkit_k4_real_fidelity.py'
    (out/'identity.json').write_text(json.dumps(dict(protocol_sha256=digest(a.protocol),runner_sha256=digest(__file__),score_script_sha256=digest(scorescript)),indent=2)+'\n')
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    def call(script,options,name):
        with (out/(name+'.log')).open('w') as f:subprocess.run([sys.executable,str(script),*map(str,options)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    result=dict(scope=p['scope'],status='failed',promotion='none')
    try:
        call(code/'python/scripts/evalkit_k4_export_variant.py',['--input',p['input'],'--method',p['method'],'--fit',p['source_fit'],'--policy',p['policy'],'--output',out/'fit'],'path')
        call(code/'python/scripts/evalkit_k4_native_export_views.py',['--code-root',code,'--fit',out/'fit','--input',p['input'],'--method',p['method'],'--window','actual','--output',out/'views'],'views')
        call(scorescript,['score','--input',p['input'],'--method',p['method'],'--views',out/'views','--output',out/'fidelity.json'],'fidelity')
        call(code/'python/scripts/evalkit_k4_export_audit.py',['--fit',out/'fit','--output',out/'actual_export'],'roundtrip')
        old=json.loads((Path(p['source_baseline'])/'fidelity.json').read_text());new=json.loads((out/'fidelity.json').read_text())
        if old['scoring_implementation']!=new['scoring_implementation']:raise ValueError('Paired fidelity implementation changed')
        if old['source_input_hashes']!=new['source_input_hashes']:raise ValueError('Paired actual input changed')
        if old['views']['native']!=new['views']['native']:raise ValueError('Native scores changed in an inserted-path-only experiment')
        if any(digest(path)!=value for path,value in sources.items()):raise ValueError('Source modified during experiment')
        receipt=json.loads((out/'fit/receipt.json').read_text())
        result.update(status='completed',native_score_exactly_unchanged=True,source_unchanged=True,native_fk_max_abs_difference_m=receipt['export_variant']['native_fk_max_abs_difference_m'])
    except Exception as error:result.update(error=str(error),traceback=traceback.format_exc())
    (out/'completion.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    if result['status']=='failed':raise SystemExit(1)
if __name__=='__main__':main()
