"""Fit existing baseline motion fields with an explicit kinematic-only model."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from motiogen.evalkit.k.domain_fitting import fit_kinematic_domain

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--method');p.add_argument('--mass',type=float,default=75.);p.add_argument('--floor',type=float,default=0.);p.add_argument('--config',default='config/evalkit_k36_20261117/kinematic_config.json');a=p.parse_args()
 import opensim as osim
 osim.Logger.setLevelString('error')
 r=fit_kinematic_domain(a.input,a.out,a.mass,a.floor,config=a.config,method=a.method)
 print(json.dumps({'output_kind':r['output_kind'],'force_eligible':r['force_eligible'],'model':r['model_path'],'coverage':{k:float(v.mean()) for k,v in r['region_angle_valid'].items()}},indent=2))
if __name__=='__main__':main()
