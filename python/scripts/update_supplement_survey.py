"""Merge supplement evidence into the existing authored survey, keep old queries."""
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'utils'))
from motiogen.runtime import checksum
base=ROOT/'develop_logs/test_runs/20261028_supplement'
target=ROOT/'develop_logs/notes/motiongen_survey_revision/src/data.json'
data=json.loads(target.read_text())
def query(name,rows,files,caveats):
    data['queries'][name]={'rows':rows,'source':{'files':[str(p.relative_to(ROOT)) for p in files],
        'sha256':{str(p.relative_to(ROOT)):checksum(p) for p in files},
        'analysisCutoff':datetime.now(timezone.utc).isoformat(),'caveats':caveats}}
gen=base/'generation_evaluation_with_references.json'
g=json.loads(gen.read_text())
query('supplement_generation',g['rows'],[gen],['Eight source texts, two seeds; system comparison, not backbone causal effect. Native lengths are retained.'])
query('supplement_reference',g['references'],[gen,base/'reference_recovery/audit.json'],['Official processed test cache; not full raw AMASS.'])
query('supplement_fid',g['distribution_diagnostics'],[gen],['Eight-source small-sample diagnostic only, not benchmark quality ranking.'])
runtime=base/'runtime_pilot_audit.json';r=json.loads(runtime.read_text())
query('supplement_runtime',r['rows'],[runtime],['Queue success is separate from response success. Paused clocks are diagnostic only.'])
query('supplement_runtime_pairs',r['pairs'],[runtime],['Common generated prefix and trajectory length differences are both retained.'])
posture_files=[base/'posture/analysis.json',base/'posture_rotations/analysis.json',base/'posture_primal_rotations_v2/analysis.json']
merged={}
for file in posture_files:
    for row in json.loads(file.read_text())['rows']:
        if row['status'].endswith('_computed'):
            normalized=dict(row)
            if 'left' in normalized.get('ergonomic',{}):
                normalized['ergonomic']={side+'_'+method:values for side,methods in row['ergonomic'].items() for method,values in methods.items()}
            merged[row['motion_path']]=dict(merged.get(row['motion_path'],{}),**normalized)
# Runtime pose exports have different evidence channels from old position metrics.
for method,relative in [('dart','dart_native_pose_posture/analysis.json'),('closd','closd_execution_posture/analysis.json')]:
    file=base/relative
    if not file.exists():continue
    posture_files.append(file)
    for row in json.loads(file.read_text())['rows']:
        if row.get('status','rotation_components_computed')!='rotation_components_computed':continue
        actual_method=row.get('method',method)
        item=dict(row,method=actual_method,task=('tracking:' if actual_method=='phc_tracker' else 'runtime:')+row['job'],seed='见原始运行',condition={'channel':row['channel']},
                  motion_path=row['run']+('/execution_trace.npz' if actual_method=='phc_tracker' else '/motion.npz'),status='rotation_components_computed')
        merged[item['motion_path']]=item
generation_posture=base/'generation_posture/analysis.json'
posture_files.append(generation_posture)
for row in json.loads(generation_posture.read_text())['rows']:
    merged[row['motion_path']]=row
uhc_posture=base/'uhc_native_posture/analysis.json'
if uhc_posture.exists():
    posture_files.append(uhc_posture)
    for row in json.loads(uhc_posture.read_text())['rows']:
        if row['status'].endswith('_computed'):merged[row['motion_path']]=row
query('supplement_posture',list(merged.values()),posture_files,['Scores are unknown-input ranges, not validated formal totals. Human ratings have not been performed. Bounds are not confidence intervals.'])
query('supplement_acceptance',[
    {'requirement':'共同参考数据','status':'8 条源文本全部接上参考动作','evidence':'官方缓存 4,648 条；5 条与独立文件数值相同'},
    {'requirement':'自动姿态与评分范围','status':str(len(merged))+' 条动作/姿态通道已重分析','evidence':'3,232 条旧结果保留；其中 2,392 条恢复旋转。新通道独立标注，缺项继续保留'},
    {'requirement':'独立人工评分','status':'尚无人评分','evidence':'384 项盲评包及浏览器交互检查已完成；不等于评分有效性核验'},
    {'requirement':'真实在线事件链','status':'三方法各 64 条主条件已完成','evidence':'200 条过程记录：含原 ARDY 8 条资源条件记录；4 次保护终止保留。主条件 ARDY 1 次实时响应失败'},
    {'requirement':'共同参考跟踪','status':'八条共同参考、24 个执行条件已完成','evidence':'PHC 7/8 完整执行；UHC RFC 开 8/8、关 0/8。终止阈值不同，不能直接当跌倒排名'},
    {'requirement':'正式统计覆盖','status':'未达到正式比较覆盖','evidence':'当前独立文本／动作数量仅用于第一阶段接口与吞吐试跑'}
],posture_files+[runtime],['Separate acceptance gates; no overall research-pass status.'])
tracking=base/'common_tracking_analysis.json'
t=json.loads(tracking.read_text())
query('supplement_tracking',t['rows'],[tracking],t['scope'])
query('supplement_tracking_pairs',t['paired_rfc'],[tracking],['Matched post-action prefix excludes initial state; failed runs retained.'])
for name,filename,key in [('supplement_tracking_previews','tracking_previews.json','rows'),('supplement_tracking_timing','phc_timing_replay_analysis.json','rows'),('supplement_tradeoff_summary','joint_tradeoff_summary.json','rows'),('supplement_online_summary','online_pilot_summary.json','rows'),('supplement_online_effects','online_pilot_summary.json','paired_goal_effects'),('supplement_capacity','phase1_capacity_and_proposal.json','rows')]:
    file=base/filename;payload=json.loads(file.read_text())
    query(name,payload[key],[file],[payload.get('scope','See source receipt')])
joint=base/'joint_correction_tradeoffs.json'
j=json.loads(joint.read_text())
tradeoffs=[{k:r[k] for k in ['method','task','seed','clock','component','other_conditions','off_motion','on_motion','metrics','failed_task_conditions']} | {'score_ranges':{k:v.get('conservative_mean_score_difference_range') for k,v in r['ergonomics'].items()}} for r in j['pairs']]
query('supplement_tradeoffs',tradeoffs,[joint],j['scope'])
curves=base/'posture_score_curves.json'
if curves.exists():
    c=json.loads(curves.read_text())
    query('supplement_posture_curves',c['rows'],[curves],[c['scope']])
if 'historical_candidate_methods' not in data['queries']:
    data['queries']['historical_candidate_methods']=data['queries']['candidate_methods']
candidates=json.loads(json.dumps(data['queries']['historical_candidate_methods']['rows']))
updates={
 'MotionGPT':('已完成16条原生生成及共同评价','Flan-T5-base编码/解码器各12层、768d；512运动词汇，VQ解码。原生长度不修改。',['官方预训练权重严格加载','8条源文本×2seed','原生生成、共同参考评价、姿态分项','不具有本轮原生空间目标接口']),
 'UHC':('八条参考×残余外力开/关已完成','固定MCP策略：8个primitive，MLP 512/256；composer 300/200/8。外力开均完成，关均提前终止；不等同重训对照。',['同一八条原生AMASS参考','分别导出主体与执行误差','残余外力、接触和成本记录','与PHC的终止阈值/几何差异单列']),
 'PhysDiff':('记录的官方来源未取得可执行代码/权重','项目页及指定仓库API响应归档；没有启动推理，没有自行重实现替代原方法。',['官方项目页、论文、发布物审计','没有取得原生投影入口','无本地投影性能结论','不是物理投影路线性能失败']),
 'PHC_MJX':('没有取得兼容原生预训练策略','隐藏MLP宽度与CLoSD相同，但原生加载字段不兼容，默认观测796d对现有权重574d。',['已克隆固定源码','已检查权重字段与观测维度','450Hz仿真/30Hz控制默认配置','未训练或默默移植策略']),
 'SMPLSim':('12组人体/体形/步长基础设施检查通过','三种SMPL性别模型×两种体形×两个步长；数值有限、轴/限位/坐标/接触检查，不作为生成性能。',['授权SMPL人体资产','原生人体XML与模型构造','两秒无控制器自由动力学','自然跌倒不判数值失败'])}
repos={r['name']:r for r in json.loads((base/'source_audit.json').read_text())['repositories']}
source_files={'MotionGPT':['mGPT/archs/mgpt_lm.py','mGPT/archs/mgpt_vq.py'], 'UHC':['config/release/uhc_implicit.yml','uhc/models/policy_mcp.py','uhc/envs/humanoid_im.py'], 'PHC_MJX':['phc_mjx/cfg/learning/im_mlp.yaml','phc_mjx/envs/humanoid_im.py'], 'SMPLSim':['smpl_sim/envs/humanoid_env.py','smpl_sim/smpllib/smpl_local_robot.py']}
for row in candidates:
    row['status'],row['finding'],row['data']=updates[row['name']]
    if row['name'] in repos:
        repo=repos[row['name']];row['source_commit']=repo['commit'];row['source_repo']=repo['url'].removeprefix('https://github.com/').removesuffix('.git');row['source_files']=source_files[row['name']]
    if row['name']=='SMPLSim':row['loop']='本机完成零控制下原生step/状态与接触检查；无生成器或跟踪策略'
    if row['name']=='MotionGPT':row['loop']='已运行原生自回归token生成；无物理执行反馈或外挂目标控制'
candidate_receipt=base/'candidate_status_update.json'
candidate_receipt.write_text(json.dumps(candidates,ensure_ascii=False,indent=2))
query('candidate_methods',candidates,[candidate_receipt,base/'candidate_release_audit_with_interfaces.json',base/'phc_mjx_interface_audit.json',base/'common_tracking_analysis.json',base/'smplsim_geometry/checks.json',gen],['Different system roles; publication availability is not a performance score.'])
online=base/'online_motion_response.json'
o=json.loads(online.read_text())
query('supplement_online_motion',o['rows'],[online],[o['scope']])
data['buildStatus']='updating'
target.write_text(json.dumps(data,ensure_ascii=False,separators=(',',':')))
print(target)
