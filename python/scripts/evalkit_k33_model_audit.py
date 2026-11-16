"""Trace lower-limb ranges/locks to local and official model assets; probe lock effect."""
import json,sys,hashlib,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
import biomech,opensim as osim
osim.Logger.setLevelString('error')
LOG=ROOT/'develop_logs/test_runs/20261117_k_conversion'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def definitions(path):
    tree=ET.parse(path);result={}
    for joint in tree.findall('.//JointSet/objects/*'):
        for c in joint.findall('./coordinates/Coordinate'):
            name=c.attrib['name']
            if not name.startswith(('hip_','knee_','ankle_','subtalar_','mtp_')):continue
            result[name]={'range_deg':(np.fromstring(c.findtext('range'),sep=' ')*180/np.pi).tolist(),'locked':c.findtext('locked')=='true',
                'clamped':c.findtext('clamped')=='true','joint':joint.attrib['name'],'joint_type':joint.tag,
                'default':float(c.findtext('default_value','0')),
                'transforms':[{'axis':x.attrib.get('name'),'coordinate':x.findtext('coordinates'),'function':[(y.tag,y.findtext('coefficients')) for y in x if y.tag.endswith('Function')]} for x in joint.findall('./SpatialTransform/TransformAxis')]}
    return result
local=definitions(biomech.MODEL);upstream_path=LOG/'upstream_RajagopalLaiUhlrich2023.osim';upstream=definitions(upstream_path)
baseline=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k31_full_20261116/074/kimodo/model.osim')
scaled=definitions(baseline)
rows={n:{'local':v,'official':upstream.get(n),'fitted_model':scaled.get(n),
        'matches_official':v==upstream.get(n),
        'fitted_properties_unchanged':all(v[k]==scaled[n][k] for k in ['range_deg','locked','clamped'])} for n,v in local.items()}
probe=[]
for coordinate,joint in [('subtalar_angle_l','mtp_l'),('mtp_angle_l','mtp_l')]:
    m=osim.Model(str(biomech.MODEL));s=m.initSystem();cs=m.getCoordinateSet();c=cs.get(coordinate);m.realizePosition(s)
    a=biomech._child_origin(m,s,joint);c.setValue(s,.3,False);m.realizePosition(s);locked_value=float(c.getValue(s));b=biomech._child_origin(m,s,joint)
    c.setLocked(s,False);c.setValue(s,.3,False);m.realizePosition(s);d=biomech._child_origin(m,s,joint)
    probe.append({'coordinate':coordinate,'request_rad':.3,'locked_value_rad':locked_value,'locked_endpoint_delta_m':float(np.linalg.norm(b-a)),
        'unlocked_value_rad':float(c.getValue(s)),'unlocked_endpoint_delta_m':float(np.linalg.norm(d-a)),
        'endpoint':joint,'scope':'mtp origin is a toe-base point; toe flexion changes distal toe orientation, not this point'})
result={'model':str(biomech.MODEL),'model_sha256':sha(biomech.MODEL),'official_file_sha256':sha(upstream_path),
 'official_source':'https://github.com/opensim-org/opensim-models/blob/master/Models/Rajagopal/RajagopalLaiUhlrich2023.osim',
 'official_readme':'https://github.com/opensim-org/opensim-models/blob/master/Models/Rajagopal/README.txt',
 'readme_sha256':sha(LOG/'upstream_README.txt'),'coordinates':rows,'lock_probe':probe,
 'classification':{'model_ranges':'author-specified model domain; not universal physiological bounds',
 'subtalar_and_mtp_locks':'present in official model asset; inherited modeling simplification, not a claim of biological immobility',
 'solver_box_bounds':'project applies model ranges as hard least-squares bounds',
 'locked_subtalar_optimization':'project includes an ineffective optimization variable; OpenSim refuses its value changes while locked',
 'expanded_knee175_hip_ankle':'prior project diagnostic hypothesis, not official calibrated domain',
 'toe_observability':'toe-base position alone does not independently observe MTP flexion',
 'muscle_stage_unlock':'project config/evalkit_v11_20261107.json sets unlock_subtalar=true; static_optimization unlocks its separate model while consuming the supplied coordinate trajectory; this is not an observed K degree of freedom'},
 'decision':'preserve authored ranges and locks by default; explicitly report inactive variables; unlock subtalar only in separate diagnostic with direction coverage; retain MTP lock without distal toe orientation observations'}
(LOG/'model_audit.json').write_text(json.dumps(result,indent=2));print('coordinates',len(rows),'all official',all(v['matches_official'] for v in rows.values()),'all fitted',all(v['fitted_properties_unchanged'] for v in rows.values()));print(probe)
