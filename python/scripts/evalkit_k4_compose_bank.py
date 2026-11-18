"""Reuse all frozen D-native candidates and select among export compositions.

Run the declared native producer first. This entry never reads R, independent
acceptance, or chooses which native results to import based on their accuracy.
"""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('input','native-bank','native-contract','config','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--method',required=True)
    p.add_argument('--mass',type=float,required=True);p.add_argument('--floor',type=float,required=True)
    a=p.parse_args();sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
    from motiogen.evalkit.k.motion import load
    from motiogen.evalkit.k.native_cache import FrozenNativeBank
    from motiogen.evalkit.k.candidate_pool import fit
    settings=json.loads(a.config.read_text());contract=json.loads(a.native_contract.read_text())
    settings['method']=a.method
    provider=FrozenNativeBank(a.native_bank,contract,settings)
    motion=load(a.input,a.method,config=ROOT/'config/evalkit_k4_20261117/input.json')
    if provider.receipt['key']['inputs']!=motion.provenance['inputs']:
        raise ValueError('Frozen bank belongs to a different declared input')
    result=fit(a.input,a.output,a.mass,a.floor,settings,native_provider=provider)
    print(json.dumps({'selected':result['receipt']['candidate_selection']['selected'],
                      'native_fit_recomputed':False,'all_native_candidates_retained':True}))


if __name__=='__main__':main()
