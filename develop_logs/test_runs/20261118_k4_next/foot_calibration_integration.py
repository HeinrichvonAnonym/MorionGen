import sys,pathlib,json,shutil,xml.etree.ElementTree as ET
r=pathlib.Path('/home/heinrich/MorionGen');sys.path[:0]=[str(r/'utils'),str(r/'utils/motiogen')]
from motiogen.evalkit.k.motion import load
from motiogen.evalkit.k.position_foot_geometry import apply
import opensim as osim
osim.Logger.setLevelString('error')
b=pathlib.Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k4_20261117');out=b/'position_foot_calibration_integration_v1';out.mkdir(exist_ok=False)
for method in ('motionlcm','omnicontrol'):
 src=b/f'closure_formats_20261118_v1/fits/geometry_domain/002/{method}/model.osim';dest=out/(method+'.osim');shutil.copyfile(src,dest)
 motion=load(b/f'runtime_inputs/002/{method}/candidate/motion.npz',method,config=r/'config/evalkit_k4_20261117/input.json')
 result=apply(dest,motion,dict(version='input_polar_envelope_v1',length_tolerance_m=.02))
 before,after=ET.parse(src),ET.parse(dest)
 def normalized(tree):
  for suffix in ('l','r'):
   joint=tree.find(f".//JointSet/objects/*[@name='mtp_{suffix}']")
   parent=joint.findtext('socket_parent_frame').split('/')[-1]
   joint.find(f"./frames/PhysicalOffsetFrame[@name='{parent}']/translation").text='ALLOWED_FIXED_ENDPOINT'
  return ET.tostring(tree.getroot())
 assert normalized(before)==normalized(after),'Unexpected model change beyond fixed endpoint'
 result['xml_only_mtp_translation_changed']=True
 (out/(method+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(method,{side:(d['status'],d.get('supported')) for side,d in result['changes'].items()},flush=True)
