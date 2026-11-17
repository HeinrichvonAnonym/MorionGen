"""Offline full-window baseline/candidate/selected overlays with identical R and D."""
import argparse,json
from pathlib import Path
from evalkit_k_animation import case,ROOT,NAMES,PARENTS,SHA

def main():
 p=argparse.ArgumentParser();p.add_argument('results',type=Path);p.add_argument('--out',type=Path,default=ROOT/'docs/20261117_k39_motion_comparison.html');a=p.parse_args()
 cases=[]
 for key,method in [('074','kimodo'),('031','motionlcm'),('038','omnicontrol'),('077','kimodo'),('051','ardy')]:
  folder=a.results/key/method;path=folder/'result.json'
  if not path.exists():continue
  r=json.loads(path.read_text())
  if r['status']!='computed':continue
  for variant in ['baseline','candidate','selected']:
   description=f'同一退化输入、完整开发窗口。当前显示 {variant}；运行时最终选择：{r["selection"]["selected"]}。原始动作只用于评分。'
   cases.append(case(folder/'comparison.npz',method,key,'K3.9 · '+variant,f'{method} · {key} · {variant}','torso',description,key='F:'+variant))
   metrics=r['scores'][variant]['spatial_directions']['trunk_axis']
   cases[-1]['metrics']={label:metrics[stage]['p95_abs'] for label,stage in [('degradation','degradation'),('fit','fit_deviation'),('end','end_to_end')]}
   cases[-1]['metric_description']='骨盆中点至肩中点的三维方向误差 P95；完整窗口，三个误差分别统计。独立胸段旋转和内部腰椎坐标另行定义。'
 payload={'footer':f'{len(cases)//3} 个开发窗口 × 基准／候选／最终选择；完整窗口、同一输入，离线显示。最终选择可以回退到基准。快捷键：空格播放／暂停，← →逐帧。','cases':cases,'names':NAMES,'parents':PARENTS,'scope':'K3.9 development only; full windows; same inputs; candidate and fallback separate',
  'display':'Whole trunk axis and joint-centre landmarks; no independent anatomical neck qualification'}
 template=(ROOT/'utils/motiogen/evalkit/visualization/motion_viewer.html').read_text()
 a.out.write_text(template.replace('__PAYLOAD__',json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')))
 (a.results/'visualization.json').write_text(json.dumps({'html':str(a.out),'sha256':SHA(a.out),'cases':len(cases),'full_frames':sum(c['frames'] for c in cases)},indent=2))
 print(a.out,len(cases))
if __name__=='__main__':main()
