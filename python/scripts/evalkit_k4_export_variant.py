"""D-only export composition; independent reference scoring is a separate entry."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('input','fit','policy','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--method',required=True)
    a=p.parse_args()
    sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
    from motiogen.evalkit.k.motion import load
    from motiogen.evalkit.k.path_variant import build
    motion=load(a.input,a.method,config=ROOT/'config/evalkit_k4_20261117/input.json')
    result=build(motion,a.fit,a.output,json.loads(a.policy.read_text()))
    print(json.dumps(result['receipt']['export_variant'],indent=2))


if __name__=='__main__':main()
