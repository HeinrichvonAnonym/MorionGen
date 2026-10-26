"""Create a report snapshot from accepted local receipts, not withdrawn prose."""
import json
import argparse
import base64
import hashlib
import re
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import ROOT
from motiogen.config import load_config
from motiogen.runtime import json_write
from motiogen.survey_motion import reviewed_motion_rows,accepted_pilot_entries
from motiogen.seed_sensitivity import annotate_seed_sensitivity
from motiogen.request_latency import recorded_requests

parser=argparse.ArgumentParser()
parser.add_argument('--complete',action='store_true',help='Mark completed authored content; requires full accepted matrix, not merely drained queues')
args=parser.parse_args()

def write_snapshot(path,value):
    # Whitespace only: retain every reviewed row, floating value and media byte.
    # Large motion arrays otherwise spend most exported bytes on indentation.
    encoded=json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n'
    if json.loads(encoded)!=value:raise ValueError('Snapshot serialization changed reviewed values')
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(encoded)
    temporary.replace(path)

pilot=ROOT/'develop_logs/test_runs/20261024_native_pilots/audited_pilots.json'
reviewed=json.loads(pilot.read_text())
registry=json.loads((ROOT/'config/native_component_experiments.json').read_text())
catalog=json.loads((ROOT/'config/motiongen_survey_methods.json').read_text())
guide=json.loads((ROOT/'config/survey_migration.json').read_text())
for method,spec in registry['methods'].items():
    source=next(r for r in catalog if r['name'].lower()==method)
    spec.update(family=source['family'],language_role=source['llm'],native_physics=source['physics'],
                source_repo=source['repo'],source_commit=source['commit'],source_files=source['files'])
    if method=='omnicontrol':
        spec['evaluation_domain']+=' 另以独立补漏域测试 12 条九时点腕部轨迹，5 seed、2×2 干预；不与原单点任务合并。'
rows=[]
for row in reviewed['rows']:
    for name,metric in row['metrics'].items():
        rows.append({'method':row['method'],'seed':row['seed'],'condition':json.dumps(row['condition']),
                     'metric':name,'value':metric.get('value'),'unit':metric.get('unit'),
                     'status':metric['status'],'reason':metric.get('reason'),'run':row['run']})
snapshot={'surface':'report','title':'MotionGen Survey：预训练组件与运动闭环',
          'status':'本机实验持续更新','buildStatus':'creating','filters':[],
          'queries':{'native_pilots':{'rows':rows,'source':{'type':'local experiment receipts',
              'files':[str(pilot.relative_to(ROOT))],
              'caveats':['首轮单个配对块不是总体置信区间','不同任务、骨架和数据域不得汇总排名'],
              'evidenceFlow':[{'title':'原始运行','detail':'每条记录指向 manifest、motion、metrics 与 intervention；资源拒绝不进入模型失败分母'},
                              {'title':'转换','detail':'audit_native_pilots.py 检查有限值与配对长度；build_native_survey_snapshot.py 展开指标'}]}},
          'native_methods':{'rows':[{'method':method,**spec} for method,spec in registry['methods'].items()],
                            'source':{'type':'reviewed local source','files':['config/native_component_experiments.json','config/motiongen_survey_methods.json']}}},
          'pilot_checks':reviewed['checks']}
snapshot['queries']['candidate_methods']={'rows':[r for r in catalog if r['name'].lower() not in registry['methods']],
    'source':{'type':'official repositories and project pages','files':['config/motiongen_survey_methods.json'],
              'caveats':['候选方法不并入本机八方法实验结果；基础设施和 tracker 不等同于文本运动生成器']}}
snapshot['queries']['execution_contract']={'rows':[
    {'layer':'方法与任务登记','files':['config/baselines.json','config/native_component_experiments.json','config/native_weight_bindings.json'],
     'contract':'方法运行规格、原生接口/干预范围与权重依赖分开登记；新增原生入口的 method_spec 随冻结任务清单保存。不是通过一个通用控制器补齐原生能力。'},
    {'layer':'统一调度与隔离','files':['python/scripts/experiments.py','utils/motiogen/experiment_queue.py','utils/motiogen/runtime.py'],
     'contract':'plan/run/status/resume；每物理GPU一把进程锁，子进程使用对应CUDA_VISIBLE_DEVICES；单任务退出不终止其他队列任务。'},
    {'layer':'复跑身份','files':['utils/motiogen/execution_identity.py','utils/motiogen/weight_identity.py','utils/motiogen/preprocessing_identity.py'],
     'contract':'绑定完整任务/seed/输入、推理Python导入闭包、执行配置、上游commit及本地diff、权重和预处理依赖。动态加载文件需显式加入identity_files。报告与派生分析修订不冒充推理变更。'},
    {'layer':'故障与期限','files':['utils/motiogen/experiment_queue.py','config/survey_revision.json'],
     'contract':'只有声明为可恢复的终态最多重试一次；资源门限/确定依赖缺失不盲重试。所有队列共享固定自然截止，提前4小时停止接收预计不能完成的新重任务。'},
    {'layer':'观测与验收','files':['python/scripts/native_execution_status.py','python/scripts/watch_native_experiments.py','python/scripts/audit_native_acceptance.py'],
     'contract':'PID创建时间、心跳、当前任务、日志/资源推进、失败历史及GPU占用分开保存；CPU监督测试不能替代GPU分配审计或实验科学验收。'}],
    'source':{'files':['python/scripts/experiments.py','utils/motiogen/experiment_queue.py','utils/motiogen/runtime.py',
                      'utils/motiogen/execution_identity.py','python/scripts/native_execution_status.py','config/survey_revision.json'],
              'caveats':['Describes the inspected engineering contracts, not new experimental results. Frozen historical job manifests remain authoritative for their runs.']}}
execution_status=ROOT/'develop_logs/test_runs/20261024_native_execution/status.json'
cohort_path=ROOT/'develop_logs/test_runs/20261025_source_replays/audit.json'
if cohort_path.exists():
    cohort=json.loads(cohort_path.read_text())
    source={'files':[str(cohort_path.relative_to(ROOT)),'develop_logs/test_runs/20261025_source_replays/source_archive.json','python/scripts/audit_native_source_cohorts.py','utils/motiogen/source_cohorts.py'],
            'analysisCutoffs':cohort['analysis_cutoffs'],'caveats':[cohort['scope']]}
    snapshot['queries']['source_replay_methods']={'rows':cohort['rows'],'source':source}
    snapshot['queries']['source_replay_status']={'rows':[{k:cohort[k] for k in ['passed','complete_groups','expected_groups','completed_processes','expected_processes','frozen_sources_still_current','frozen_source_archive_matches','frozen_source_file_count']}],'source':source}
    snapshot['queries']['source_replay_comparisons']={'rows':[{**{k:v for k,v in r.items() if k!='comparisons'},**c} for r in cohort['replays'] for c in r['comparisons']],'source':source}
epoch_path=ROOT/'develop_logs/test_runs/20261025_execution_epochs/audit.json'
if epoch_path.exists():
    epoch=json.loads(epoch_path.read_text())
    snapshot['queries']['execution_epoch_sensitivity']={'rows':epoch['effects'],
        'source':{'files':[str(epoch_path.relative_to(ROOT)),'python/scripts/audit_native_execution_epochs.py'],
                  'analysisCutoffs':epoch['analysis_cutoffs'],'mixedBlocks':epoch['mixed_blocks'],
                  'caveats':[epoch['scope']]}}
if execution_status.exists():
    status=json.loads(execution_status.read_text())
    snapshot['queries']['execution_budget']={'rows':[{
        'recorded_at':status['timestamp'],'deadline':status['deadline'],
        'wall_elapsed_hours':status['wall_elapsed_hours'],'wall_remaining_hours':status['wall_remaining_hours'],
        'report_reserve_hours':status['report_reserve_hours'],
        'gpu0_lease_hours':status['supervised_gpu_lease_hours'].get('0',0),
        'gpu1_lease_hours':status['supervised_gpu_lease_hours'].get('1',0),
        'gpu0_observed_lower_hours':status.get('observed_gpu_lease_hours_lower_bound',status['supervised_gpu_lease_hours']).get('0',0),
        'gpu1_observed_lower_hours':status.get('observed_gpu_lease_hours_lower_bound',status['supervised_gpu_lease_hours']).get('1',0),
        'remaining_projected_lease_hours':status['workload_projection']['estimated_gpu_lease_hours'],
        'scope':status['ledger_scope']}],
        'source':{'files':[str(execution_status.relative_to(ROOT))],
            'caveats':['Static resource snapshot, not live process telemetry. 36 elapsed wall hours and 72 GPU-hours are different budgets. Lease time includes loading and host work, not integrated kernel utilization.']}}
snapshot['queries']['method_uses']={'rows':guide['uses'],'source':{'files':['config/survey_migration.json','config/native_component_experiments.json','config/motiongen_survey_methods.json'],
    'caveats':['接口设计适配，不是跨方法性能排名；实测能力另见逐 case 与配对统计']}}
environment_path=ROOT/'develop_logs/test_runs/20261024_native_environments/summary.json'
if environment_path.exists():
    environment=json.loads(environment_path.read_text())
    snapshot['queries']['native_environments']={'rows':environment['rows'],
        'source':{'files':[str(environment_path.relative_to(ROOT)),'python/scripts/summarize_native_environments.py'],
                  'recordedAt':environment['recorded_at'],'caveats':[environment['scope']]}}
snapshot['queries']['report_migration']={'rows':guide['sections'],'source':{'files':['config/survey_migration.json',guide['source']],
    'caveats':['迁移清单不替代内容与浏览器验收；原报告按历史日期保留，错误或过时结论以上方纠正为准']}}
chois_audit_path=ROOT/'develop_logs/test_runs/20261024_chois_native_metrics/audit.json'
ardy_history_path=ROOT/'develop_logs/test_runs/20261024_ardy_history_audit/audit.json'
if ardy_history_path.exists():
    history_audit=json.loads(ardy_history_path.read_text())
    snapshot['queries']['ardy_history_audit']={'rows':history_audit['summary'],
        'source':{'files':[str(ardy_history_path.relative_to(ROOT)),'python/scripts/audit_ardy_history.py'],
                  'analysisCutoff':history_audit['analysis_cutoff'],'caveats':[history_audit['scope']]}}
chois_contact_path=ROOT/'develop_logs/test_runs/20261024_chois_contact_audit/audit.json'
if chois_contact_path.exists():
    contact_audit=json.loads(chois_contact_path.read_text())
    snapshot['queries']['chois_contact_audit']={'rows':contact_audit['summary'],
        'source':{'files':[str(chois_contact_path.relative_to(ROOT)),'python/scripts/audit_chois_contact.py'],
                  'analysisCutoff':contact_audit['analysis_cutoff'],'caveats':[contact_audit['scope']]}}
chois_load_path=ROOT/'develop_logs/test_runs/20261024_chois_checkpoint_load/audit.json'
if chois_load_path.exists():
    load_audit=json.loads(chois_load_path.read_text())
    snapshot['queries']['chois_checkpoint_load']={'rows':[{k:v for k,v in load_audit.items() if k not in ['runs','configuration_sha256']}],
        'source':{'files':[str(chois_load_path.relative_to(ROOT)),'python/scripts/audit_chois_checkpoint_load.py'],
                  'caveats':[load_audit['scope']]}}
clock_audit_path=ROOT/'develop_logs/test_runs/20261024_native_clock_observations/audit.json'
marker_path=ROOT/'develop_logs/test_runs/20261024_closd_marker_probe/analysis.json'
if marker_path.exists():
    marker=json.loads(marker_path.read_text())
    for query,key in [('closd_marker_cases','cases'),('closd_marker_runs','rows'),('closd_marker_comparisons','comparisons'),('closd_marker_clock_cases','cross_clock_cases')]:
        snapshot['queries'][query]={'rows':marker[key], 'source':{
            'files':[str(marker_path.relative_to(ROOT)),'python/scripts/closd_marker_probe.py','python/scripts/analyze_closd_marker_probe.py'],
            'analysisCutoff':marker['updated_at'],'caveats':[marker['scope']]}}
if clock_audit_path.exists():
    clock_audit=json.loads(clock_audit_path.read_text())
    snapshot['queries']['native_clock_observation_bounds']={'rows':clock_audit['summary'],
        'source':{'files':[str(clock_audit_path.relative_to(ROOT)),'python/scripts/audit_native_clock_observations.py'],
                  'analysisCutoff':clock_audit['analysis_cutoff'],'caveats':[clock_audit['scope']]}}
if chois_audit_path.exists():
    chois_audit=json.loads(chois_audit_path.read_text())
    snapshot['queries']['chois_native_metrics']={'rows':chois_audit['summary'],
        'source':{'files':[str(chois_audit_path.relative_to(ROOT)),'python/scripts/audit_chois_native_metrics.py'],
                  'analysisCutoff':chois_audit['analysis_cutoff'],'caveats':[chois_audit['scope']]}}
published=json.loads((ROOT/'config/survey_published_evidence.json').read_text())
snapshot['queries']['published_architecture']={'rows':published,'source':{'files':['config/survey_published_evidence.json'],
    'urls':[r['source'] for r in published],'caveats':['Author-retrained architecture ablations; not local component interventions or local hardware measurements']}}
for query,relative,key in [('native_invariants','20261024_native_invariants/audit.json','summary'),
                           ('dart_body_compatibility','20261024_dart_body_compatibility/audit.json','checks')]:
    path=ROOT/'develop_logs/test_runs'/relative
    if path.exists():
        audit=json.loads(path.read_text())
        snapshot['queries'][query]={'rows':audit[key], 'source':{
            'files':[str(path.relative_to(ROOT))], 'caveats':[audit['scope']],
            'analysisCutoff':audit.get('analysis_cutoff')}}
preprocessing=ROOT/'develop_logs/test_runs/20261024_native_preprocessing/audit.json'
if preprocessing.exists():
    audit=json.loads(preprocessing.read_text())
    snapshot['queries']['native_preprocessing']={'rows':[{'method':method,
        'files':len(identity['files']),'checked_runs':sum(c['method']==method for c in audit['checks']),
        'passed_runs':sum(c['method']==method and c['passed'] for c in audit['checks'])}
        for method,identity in audit['identities'].items()],
        'source':{'files':[str(preprocessing.relative_to(ROOT))],'caveats':[audit['scope']]}}
reconstruction=ROOT/'develop_logs/test_runs/20261024_native_metric_reconstruction/audit.json'
if reconstruction.exists():
    audit=json.loads(reconstruction.read_text())
    methods=sorted({c['method'] for c in audit['checks']})
    snapshot['queries']['native_metric_reconstruction']={'rows':[{
        'method':method,'condition_cells':sum(c['method']==method for c in audit['checks']),
        'metric_checks':sum(len(c['checks']) for c in audit['checks'] if c['method']==method),
        'passed_cells':sum(c['method']==method and c['passed'] for c in audit['checks'])}
        for method in methods], 'source':{'files':[str(reconstruction.relative_to(ROOT))],
        'analysisCutoff':audit['analysis_cutoff'],'caveats':[audit['scope']]}}
coordinate_path=ROOT/'develop_logs/test_runs/20261024_spatial_coordinates/audit.json'
if coordinate_path.exists():
    audit=json.loads(coordinate_path.read_text())
    snapshot['queries']['spatial_coordinates']={'rows':[{
        'tasks':len(audit['packing']), 'runs':len(audit['runs']), 'passed':audit['passed'],
        'official_sparse_constructor_matches':sum(r['official_sparse_constructor_exact'] for r in audit['packing']),
        'target_roundtrip_max_m':max(r['max_target_roundtrip_error_m'] for r in audit['packing']),
        'native_metric_max_difference_m':max(r['metric_difference_m'] for r in audit['runs'])}],
        'source':{'files':[str(coordinate_path.relative_to(ROOT))],
                  'analysisCutoff':audit['analysis_cutoff'],'caveats':[audit['scope']]}}
visual_entries=list(reviewed['rows'])
current_matrix=ROOT/'develop_logs/test_runs/20261024_native_matrix_analysis/analysis.json'
if current_matrix.exists():
    visual_entries=accepted_pilot_entries(json.loads(current_matrix.read_text())['rows'],reviewed['rows'])
kimodo_visual=ROOT/'develop_logs/test_runs/20261024_kimodo_survey_cases/entries.json'
if kimodo_visual.exists():visual_entries.extend(json.loads(kimodo_visual.read_text()))
diagnostic_entries=ROOT/'develop_logs/test_runs/20261024_native_diagnostic_cases/entries.json'
if diagnostic_entries.exists():visual_entries.extend(json.loads(diagnostic_entries.read_text()))
contact_counterexample=ROOT/'develop_logs/test_runs/20261024_chois_contact_counterexample/entries.json'
if contact_counterexample.exists():visual_entries.extend(json.loads(contact_counterexample.read_text()))
trajectory_path=ROOT/'develop_logs/test_runs/20261024_omnicontrol_trajectory/analysis.json'
trajectory=None
if trajectory_path.exists():
    trajectory=json.loads(trajectory_path.read_text())
    trajectory_source={'files':[str(trajectory_path.relative_to(ROOT)),'config/native_trajectory_experiments.json','python/scripts/analyze_native_trajectories.py'],
                       'analysisCutoff':trajectory['updated_at'],'caveats':[trajectory['scope']]}
    for query,key in [('trajectory_runs','rows'),('trajectory_effects','contrasts'),('trajectory_simple_effects','simple_effects')]:
        snapshot['queries'][query]={'rows':trajectory[key],'source':trajectory_source}
    snapshot['queries']['trajectory_coverage']={'rows':[{k:trajectory[k] for k in ['accepted_cells','expected_cells','complete_blocks','complete_tasks','pilot_passed','negative_control_groups','negative_control_compared_groups','negative_controls_passed']}],
        'source':trajectory_source}
    complete={(b['task'],b['seed']) for b in trajectory['blocks'] if b['complete']}
    for row in trajectory['rows']:
        if row['seed']==20 and (row['task'],row['seed']) in complete:
            visual_entries.append({**row,'scene':row['task_spec'],'case':'trajectory_task'+str(row['task'])+'_seed20'})
matrix_path=ROOT/'develop_logs/test_runs/20261024_native_matrix_analysis/analysis.json'
if matrix_path.exists():
    matrix_rows=json.loads(matrix_path.read_text())['rows']
    for row in matrix_rows:
        if row['method']=='primal' and row['clock']=='paused' and row['seed']==20 and row['task'] in [0,1,6]:
            visual_entries.append({**row,'scene':{'constraints':[]},'case':'initial'+str(row['task'])+'_seed20',
                'motion_path':row['run']+'/factorial/'+row['variant']+'_canonical.npz'})
        if row['method']=='closd' and row['seed']==20 and row['task']==0 and row['clock']=='paused':
            visual_entries.append({**row,'scene':{'constraints':[]},'case':'native_physics_seed20'})
motions,curves=reviewed_motion_rows(visual_entries)
snapshot['queries']['motion_cases']={'rows':motions,'payloadColumns':['joints'],
    'source':{'files':[r.get('motion_path',r['run']+'/motion.npz') for r in visual_entries]+['develop_logs/test_runs/20261024_chois_survey_geometry/geometry.json']+(['develop_logs/test_runs/20261024_native_diagnostic_cases/selection.json'] if diagnostic_entries.exists() else [])+(['develop_logs/test_runs/20261024_chois_contact_counterexample/selection.json'] if contact_counterexample.exists() else []),
        'caveats':['CHOIS display uses 320 sampled mesh vertices; native full-mesh hand distances are reconstructed within 2e-6 m. Native quaternion projection precedes object geometry, exactly as trainer_chois.py:2618-2620.',
                   'Diagnostic examples select a declared tail metric within complete blocks and expose that rule. They are not random representative cases and do not change full-matrix estimates.'],
        'metricDefinitions':{'paired-motion-player':'Canonical metres, Z-up; skeleton rendering rounded to 0.1mm only; same selected spatial scale in both panels'}}}
snapshot['queries']['motion_curves']={'rows':curves,
    'source':{'files':[r.get('motion_path',r['run']+'/motion.npz') for r in visual_entries]+['utils/motiogen/survey_motion.py'],
        'metricDefinitions':{'paired-motion-curve':'Pelvis-neck angle to vertical; consecutive pelvis displacement times native FPS; distance to supplied target. Distance outside constraint time is descriptive, not constraint score. Jerk curve is raw coordinate third-difference times FPS cubed, RMS across joint XYZ components, indexed at the four-frame window end. Its squared temporal mean equals squared whole-clip component RMS; first three frames are null.'}}}
archive=[]
for path in sorted((ROOT/'develop_logs/notes').glob('*')):
    if path.suffix not in ['.md','.html']: continue
    content=path.read_text();assets={};images={}
    if path.name=='20261007_abeh_baseline_interactive_report.html':
        block=re.search(r'const cases = \[(.*?)\n\s*\];',content,re.S)
        if block is None:raise ValueError('Original 1007 case registry absent')
        identifiers=sorted(set(re.findall(r'2026\d{4}_\d{6}_[A-Za-z0-9_-]+_[a-f0-9]{8}',block.group(1))))
        if len(identifiers)!=20:raise ValueError('Expected the reviewed twenty original case videos')
        for identifier in identifiers:
            relative=f'../test_runs/{identifier}/motion.mp4'
            media=path.parent/relative;payload=media.read_bytes()
            assets[relative]={'mime':'video/mp4','sha256':hashlib.sha256(payload).hexdigest(),
                'bytes':len(payload),'base64':base64.b64encode(payload).decode('ascii'),
                'source_path':str(media.resolve())}
        case_ids=re.findall(r"\[\s*'[^']+'\s*,\s*'([^']+)'",block.group(1))
        image_paths={x for x in re.findall(r'<img[^>]+src="([^"]+)"',content) if '{' not in x}
        image_paths.update(f'../test_runs/20261007_abeh_fixed_scale_gallery/{case}.png' for case in case_ids)
        if len(image_paths)!=24:raise ValueError('Expected twenty case images and four GT/raw/correction images')
        for relative in sorted(image_paths):
            media=path.parent/relative;payload=media.read_bytes()
            images[relative]={'mime':'image/png','sha256':hashlib.sha256(payload).hexdigest(),
                'bytes':len(payload),'base64':base64.b64encode(payload).decode('ascii'),'source_path':str(media.resolve())}
    archive.append({'name':path.name,'format':path.suffix[1:],'content':content,'video_assets':assets,'image_assets':images,
                    'role':'historical original, not a replacement for current native-component conclusions'})
snapshot['queries']['historical_reports']={'rows':archive,'payloadColumns':['content','video_assets','image_assets'],
    'source':{'type':'original local reports','files':[str(Path('develop_logs/notes')/r['name']) for r in archive],
              'caveats':['原始脚本、case、视频控件完整保留；历史项目包装成绩不并入原生能力表',
                         'Twenty original MP4s and twenty-four PNGs embedded byte-for-byte with SHA256 for sandbox-safe file:// viewing; no media regeneration. Raw evidence links still point to repository files.']}}
kimodo=ROOT/'develop_logs/test_runs/20261024_kimodo_pair_audit/audit.json'
if kimodo.exists():
    audited=json.loads(kimodo.read_text())
    snapshot['queries']['kimodo_correction']={'rows':audited['summary'],
        'source':{'type':'audited paired native inference','files':[str(kimodo.relative_to(ROOT))],
            'metricDefinitions':{'kimodo-correction-table':'Postprocess minus raw; equal source-BVH weight; source-BVH bootstrap 95% intervals'},
            'caveats':['416 pairs from 385 source BVHs; source crops grouped together, not independent frames',
                       'Conditional on existing benchmark selection; not a retrained architecture comparison']}}
kimodo_weights=ROOT/'develop_logs/test_runs/20261024_kimodo_weight_audit/audit.json'
if kimodo_weights.exists():
    weights=json.loads(kimodo_weights.read_text())
    snapshot['queries']['kimodo_weights']={'rows':weights['checks'],'source':{'files':[str(kimodo_weights.relative_to(ROOT))],
        'caveats':[weights['scope']]}}
replay_state=ROOT/'develop_logs/test_runs/20261024_kimodo_recorrection/queue/state.json'
if replay_state.exists():
    job=json.loads(replay_state.read_text())['jobs']['kimodo_official_correction_replay']
    if job['status']=='success':
        replay_path=Path(job['run_dir'])/'replay.json'
        replay=json.loads(replay_path.read_text())
        cpu_path=ROOT/'develop_logs/test_runs/20261024_kimodo_recorrection/audit.json'
        cpu=json.loads(cpu_path.read_text())
        fields=['root_positions','local_rot_mats','global_rot_mats','posed_joints','foot_contacts']
        snapshot['queries']['kimodo_recorrection']={'rows':[{'field':field,
            'unit':'m' if field in ['root_positions','posed_joints'] else ('boolean mismatch' if field=='foot_contacts' else 'matrix element (dimensionless)'),
            'pairs':replay['checked_pairs'],
            'gpu_max_abs_difference':max(f['max_abs_difference'] for c in replay['checks'] for f in c['fields'] if f['field']==field),
            'cpu_max_abs_difference':max(f['max_abs_difference'] for c in cpu['checks'] for f in c['fields'] if f['field']==field),
            'gpu_passed_pairs':sum(next(f['passed'] for f in c['fields'] if f['field']==field) for c in replay['checks'])}
            for field in fields], 'source':{'files':[str(replay_path.relative_to(ROOT)),str(cpu_path.relative_to(ROOT)),
                'python/scripts/audit_kimodo_recorrection.py'],'caveats':[replay['scope'],
                'CPU precision discrepancies are numerical replay diagnostics, not model capability failures.']}}
destination=ROOT/'develop_logs/test_runs/20261024_native_pilots/report_snapshot.json'
matrix=ROOT/'develop_logs/test_runs/20261024_native_matrix_analysis/analysis.json'
if matrix.exists():
    analyzed=json.loads(matrix.read_text())
    snapshot['queries']['native_clock_pairs']={'rows':[
        {'method':method,'paired_conditions':len(pairs),
         'same_assets':sum(r['same_assets'] for r in pairs),
         'same_shape':sum(r['same_shape'] for r in pairs),
         'exact_same_coordinates':sum(r['exact_same_coordinates'] for r in pairs),
         'different_coordinates':sum(not r['exact_same_coordinates'] for r in pairs)}
        for method in ['ardy','dart','primal','closd']
        for pairs in [[r for r in analyzed['clock_pairs'] if r['method']==method]]],
        'source':{'files':[str(matrix.relative_to(ROOT))],'analysisCutoff':analyzed['updated_at'],
                  'caveats':['Same task, seed and intervention across clocks. No pooling as independent samples, even when coordinates differ. Trajectory equality does not imply equal timing or hard-real-time guarantees.']}}
    snapshot['queries']['native_effects']={'rows':analyzed['contrasts'],
        'source':{'type':'complete paired factorial blocks','files':[str(matrix.relative_to(ROOT))],
                  'caveats':[analyzed['interpretation'],'登记场景覆盖已完成，不按结果方向挑样本' if args.complete else '进行中的中期统计；最终结论须等待预登记场景覆盖，不按中期结果挑样本']}}
    snapshot['queries']['native_runs']={'rows':analyzed['rows'],'source':{'type':'local native run receipts',
        'files':[str(matrix.relative_to(ROOT))],'caveats':['Foot contact by height is a kinematic proxy, not force contact',
        'Cold text/model setup, native sampling, physics-step time and complete worker time have different scopes; single offline calls do not establish steady latency distributions',
        'CLoSD execution_wall_seconds includes pacing and terminal weight verification, motion/contact export and metric writing after the last env_step; not pure execution or the complete worker lifetime. Its cold_start_seconds already includes the first PHC action. Direct env_step, DiP and PHC timers are separate and must not be blindly added to cold start.']}}
    snapshot['queries']['native_coverage']={'rows':[{'method':m,**counts} for m,counts in analyzed['coverage'].items()],
        'source':{'files':[str(matrix.relative_to(ROOT))],'caveats':['任务进程状态不是模型能力成败；复用首跑另有证据入口']}}
    text_setup=[]
    for clock in ['paused','realtime']:
        for profile in ['native_precompute','required_hit_cache']:
            selected=[r for r in analyzed['rows'] if r['method']=='ardy' and r['clock']==clock and r.get('text_execution_profile')==profile]
            if not selected:continue
            entry={'clock':clock,'profile':profile,'runs':len(selected)}
            for metric in ['text_setup_seconds','model_load_seconds']:
                values=[r['values'][metric] for r in selected if metric in r['values']]
                entry[metric+'_n']=len(values)
                entry[metric+'_median']=float(np.median(values)) if values else None
                entry[metric+'_p95']=float(np.quantile(values,.95)) if values else None
            text_setup.append(entry)
    snapshot['queries']['ardy_text_setup']={'rows':text_setup,'source':{
        'files':[str(matrix.relative_to(ROOT))],'analysisCutoff':analyzed['updated_at'],
        'caveats':['Observed process-start distributions, not matched cache-speedup effects. Both finite prompts are encoded before the timed input change. text_model_load_and_encode and model_load are sequential, disjoint timers in Nvidia.load; model_load receives the prepared text encoder and measures motion-model loading. Their per-run sum omits other worker startup work; sums of marginal quantiles are not total-startup quantiles.']}}
    snapshot['queries']['native_failures']={'rows':analyzed['failures'],'source':{'files':[str(matrix.relative_to(ROOT))]}}
    snapshot['queries']['native_capability_rates']={'rows':analyzed['capability_rates'],'source':{'files':[str(matrix.relative_to(ROOT))],
        'caveats':['完成运行中的能力结果；未完成任务不填失败，进程失败单列；各方法成功阈值和任务不同']}}
    snapshot['queries']['native_simple_effects']={'rows':analyzed.get('simple_effects',[]),
        'source':{'files':[str(matrix.relative_to(ROOT))], 'analysisCutoff':analyzed['updated_at'],
                  'caveats':['Other switches held at stated values; within complete factorial blocks only. Same task/parent-cluster bootstrap, no new samples or retraining. Pointwise intervals, not multiplicity-adjusted.']}}
    snapshot['queries']['native_blocks']={'rows':analyzed['blocks'],'source':{'files':[str(matrix.relative_to(ROOT))],
        'caveats':['仅完整且配对身份一致的块；每个点是同一任务、seed、时钟中的组件差值']}}
    snapshot['queries']['matrix_acceptance']={'rows':analyzed['matrix_acceptance'],'source':{'files':[str(matrix.relative_to(ROOT))],
        'caveats':['Scientific cells, not queued processes. PRIMAL has eight cells per worker. Clocks shown separately and never pooled as independent motion samples.',
                   'Kimodo reuses 416 historical pairs and is reported separately, not forced into a new 12x5 matrix.']}}
    for query in ['native_effects','native_runs','native_coverage','native_failures',
                  'native_capability_rates','native_blocks','matrix_acceptance']:
        snapshot['queries'][query]['source']['analysisCutoff']=analyzed['updated_at']
        snapshot['queries'][query]['source']['analysisSourceSha256']=analyzed.get('analysis_source_sha256',{})
historical_audit=ROOT/'develop_logs/test_runs/20261024_historical_survey_audit/audit.json'
if historical_audit.exists():
    history=json.loads(historical_audit.read_text())
    source={'files':[str(historical_audit.relative_to(ROOT))], 'caveats':['历史预训练推理证据，按原始任务、日期和计时范围保留，不混入本轮新增矩阵']}
    snapshot['queries']['historical_raw_ability']={'rows':history['absolute'],'source':source}
    snapshot['queries']['historical_kimodo_timing']={'rows':[{'metric':k,**v} for k,v in history['timing_delta'].items()],
        'source':{**source,'caveats':[history['timing_scope']]}}
    snapshot['queries']['historical_kit_reference']={'rows':[{'metric':k,**v} for k,v in history['kit']['metrics'].items()],
        'source':{**source,'caveats':['KIT-ML 12.5 FPS 真实动作参考，不是跨域排名或临床工效阈值']}}
    tmr_path=Path(load_config()['data_root'])/'benchmarks/kimodo-stratified-20261008/raw_summary_rows.json'
    tmr=json.loads(tmr_path.read_text())
    snapshot['queries']['historical_tmr']={'rows':[{'split':r['split'],'category':r['category'],'n':r['num_motions'],**r['tmr']} for r in tmr['rows'] if r.get('tmr')],
        'source':{'files':[str(tmr_path)],'caveats':['保留官方叶组评估后聚合的原始 TMR；没有重算全体 1856 条的单一 FID，不能将 FID 平均值当全局 FID']}}
timing_path=ROOT/'develop_logs/test_runs/20261024_native_timing/analysis.json'
if timing_path.exists():
    timing=json.loads(timing_path.read_text())
    snapshot['queries']['primal_primary_timing']={'rows':timing['primal_primary_decomposition'],
        'source':{'files':[str(timing_path.relative_to(ROOT))],'caveats':[timing['primal_primary_scope']]}}
    snapshot['queries']['native_timing']={'rows':timing['rows'],'source':{'files':[str(timing_path.relative_to(ROOT))],'caveats':[timing['interpretation']]}}
    if timing.get('primal_clock_decomposition'):
        snapshot['queries']['primal_timing']={'rows':timing['primal_clock_decomposition'],
            'source':{'files':[str(timing_path.relative_to(ROOT))],'caveats':[timing['primal_scope']]}}
for query, evidence in [('native_effects', analyzed), ('trajectory_effects', trajectory)]:
    if evidence is not None and query in snapshot['queries']:
        annotate_seed_sensitivity(snapshot['queries'][query]['rows'], evidence['blocks'])
        snapshot['queries'][query]['source'].setdefault('files', []).append('utils/motiogen/seed_sensitivity.py')
for row in snapshot['queries'].get('native_runs',{}).get('rows',[]):
    row['request_latency']=recorded_requests(row['method'],row['run'])
snapshot['queries']['native_runs']['source']['files'].append('utils/motiogen/request_latency.py')
project=ROOT/'develop_logs/notes/motiongen_survey_revision/src/data.json'
if project.exists():
    existing=json.loads(project.read_text())
    snapshot['id']=existing['id']
    snapshot['buildStatus']='updating'
if args.complete:
    if not cohort_path.exists() or not cohort['passed']:
        raise ValueError('Frozen-source replay and input/weight comparison must finish before final delivery')
    if trajectory is None or trajectory['accepted_cells']!=240 or trajectory['complete_blocks']!=60 or trajectory['complete_tasks']!=12:
        raise ValueError('Supplementary native trajectory coverage must be complete before final delivery')
    coverage=snapshot['queries'].get('matrix_acceptance',{}).get('rows',[])
    if len(coverage)!=11 or any(r['accepted_cells']!=r['expected_cells'] or r['complete_blocks']!=60 or r['complete_tasks']!=12 for r in coverage):
        raise ValueError('Cannot mark authored report complete before registered matrix coverage')
    if analyzed['rejected']:raise ValueError('Rejected scientific blocks require reconciliation')
    snapshot['buildStatus']='complete'
    snapshot['status']='本机登记矩阵与报告整合完成'
if project.exists():write_snapshot(project,snapshot)
write_snapshot(destination,snapshot)
print(destination)
