#!/usr/bin/env python3
"""Separate upstream native conventions from the delivered archive and K internal basis."""
import json,subprocess,sys,ast
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen'),str(ROOT/'python/scripts')]
from evalkit_k_regression import sample_path
from motiogen.evalkit.k import load
from motiogen.evalkit.k.config import digest
from motiogen.evalkit.k.export import target_template


def main():
    folder=ROOT/'config/evalkit_k3_20261116';registry=json.loads((folder/'conventions.json').read_text())
    samples=json.loads((ROOT/'config/gold_k_20261115/evidence.json').read_text())['baseline_samples'];rows={}
    for method,items in samples.items():
        source=sample_path(items[0]['path']);motion=load(source,method,config='config/evalkit_k_v3.json')
        native=Path(registry['records'][method]['sources'][0]['path']);upstream=None
        git=subprocess.run(['git','-C',str(native.parent),'rev-parse','HEAD'],capture_output=True,text=True)
        if git.returncode==0:upstream=git.stdout.strip()
        rows[method]={'sample':str(source),'upstream_revision':upstream,'source_file_sha256':digest(native),
            'native_definition':registry['records'][method]['rotation'],
            'delivered_position_contract':motion.provenance['source_position_contract'],
            'internal_position_contract':{'axes':motion.provenance['axes'],'units':'m','classification':'项目自行约定'},
            'world_conversion':{'value':'left multiply rotations and rotate position vectors by the same proper Y/Z basis transform; preserve local axes',
                                'classification':'按来源公式确定性转换'},
            'source_asset_basis':{'status':motion.provenance['calibration']['status'],
                                 'rest_geometry':motion.provenance['calibration'].get('rest_geometry'),
                                 'bone_direction_p95_deg':motion.provenance['calibration'].get('bone_direction_p95_deg'),
                                 'anatomical_correspondence':'尚未确定'},
            'native_rotation':motion.rotations is not None,'input_sha256':motion.provenance['inputs']}
        names,rest,parents=target_template(method);asset='primal' if method in ['motionlcm','omnicontrol','chois'] else method
        rows[method]['controlled_fixture']={'names':names,'parents':parents,'asset':f'config/evalkit_k_calibration/{asset}.json',
            'asset_sha256':digest(ROOT/f'config/evalkit_k_calibration/{asset}.json'),
            'classification':'项目自行约定' if method in ['motionlcm','omnicontrol','chois'] else '来源资产与项目节点映射',
            'scope':'fixed controlled-test body; actual generated body lengths are observed separately'}
        rows[method]['rotation_parameter_units']='axis-angle radians' if method=='primal' else 'xyzw unit quaternion' if method=='closd' else 'dimensionless matrices' if motion.rotations is not None else 'no retained independent segment rotations'
        rows[method]['metric_nodes']=json.loads((ROOT/'config/evalkit_k_v3.json').read_text())['profiles'][method]
        if method in ['motionlcm','omnicontrol']:
            # Upstream HumanML chain is checked independently of the SMPL fixture.
            prefix=Path(registry['records'][method]['sources'][0]['path']).parents[1]
            topology=prefix/'utils/paramUtil.py';tree=ast.parse(topology.read_text())
            chains=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='t2m_kinematic_chain' for t in n.targets))
            upstream_parents=[-1]*22
            for chain in chains:
                for parent,child in zip(chain,chain[1:]):upstream_parents[child]=parent
            if upstream_parents!=parents:raise ValueError('HumanML topology mismatch: '+method)
            rows[method]['upstream_hierarchy_check']={'path':str(topology),'sha256':digest(topology),'parents':upstream_parents,'matches_fixture_topology':True}
    result={'scope':'registered actual versions only; schema metadata belongs to project adapters when canonicalized',
        'classification_policy':{'source':'来源明确给出','conversion':'按来源公式确定性转换','estimation':'从几何或动作估计',
                                 'project':'项目自行约定','unknown':'尚未确定'},
        'handedness_scope':'canonical schema explicitly declares right handed; algebraic basis transforms have determinant +1; this does not define anatomical segment axes',
        'records':rows}
    (folder/'source_contract_checks.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print('actual source contracts',len(rows))

if __name__=='__main__':main()
