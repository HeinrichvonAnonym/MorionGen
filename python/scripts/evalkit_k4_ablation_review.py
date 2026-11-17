"""Summarize paired causal contrasts; quantile differences are not additive contributions."""
import argparse
import json
from pathlib import Path
import statistics


METRICS={
    '髋线方向_deg':('directions_deg','pelvis_line'),
    '躯干方向_deg':('directions_deg','trunk_axis'),
    '肩线方向_deg':('directions_deg','shoulder_line'),
    '左肩位置_cm':('positions_cm','shoulders','left_shoulder','world'),
    '右肩位置_cm':('positions_cm','shoulders','right_shoulder','world'),
    '左膝角_deg':('joint_angles_deg','left_knee'),
    '右膝角_deg':('joint_angles_deg','right_knee'),
    '右大腿方向_deg':('directions_deg','right_thigh'),
    '右小腿方向_deg':('directions_deg','right_shank'),
    '左膝位置_cm':('positions_cm','knees','left_knee','world'),
    '右膝位置_cm':('positions_cm','knees','right_knee','world'),
    '左足方向_deg':('directions_deg','left_foot_axis'),
    '右足方向_deg':('directions_deg','right_foot_axis'),
    '左踝位置_cm':('positions_cm','feet','left_ankle','world'),
    '右踝位置_cm':('positions_cm','feet','right_ankle','world'),
    '右足端位置_cm':('positions_cm','feet','right_foot','world'),
}


def extract(score,comparison,path):
    value=score['paired_baseline_clock']['comparisons'][comparison]['full']
    for key in path:value=value[key]
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    panel=json.loads((a.run/'fixed_panel.json').read_text())['records']
    repo=Path(__file__).resolve().parents[2]
    windows={r['window_key']:r for r in json.loads((repo/'config/k_degradation_20261117/windows.json').read_text())['records']}
    completion=json.loads((a.run/'completion.json').read_text())
    contrasts=[('约束数值处理','legacy_common/before','legacy_common/filtered_direct_before'),
               ('滤波与重采样','legacy_common/filtered_direct_before','g0_s0/before'),
               ('固定几何','g0_s0/before','g1_s0/before'),
               ('姿态细化_原几何','g0_s0/before','g0_s1/before'),
               ('姿态细化_独立几何','g1_s0/before','g1_s1/before')]
    contrasts += [('髋线目标_原关节域','g1_s1/before','g1_s1_pelvis/before'),
                  ('距下解锁_原目标','g1_s1/before','g1_s1_subtalar/before'),
                  ('髋线目标_距下解锁','g1_s1_subtalar/before','g1_s1_pelvis_subtalar/before'),
                  ('距下解锁_含髋线目标','g1_s1_pelvis/before','g1_s1_pelvis_subtalar/before')]
    conditions=['g0_s0','g1_s0','g0_s1','g1_s1','g1_s1_pelvis','g1_s1_subtalar','g1_s1_pelvis_subtalar']
    contrasts += [('root_'+c,c+'/before',c+'/after') for c in conditions]
    records=[];summaries=[]
    for label,left,right in contrasts:
        for row in panel:
            name=f"{row['window']}_{row['method']}.json"
            lf=a.run/'scores'/left/name;rf=a.run/'scores'/right/name
            if not lf.exists() or not rf.exists():continue
            old=json.loads(lf.read_text());new=json.loads(rf.read_text())
            for comparison in ('F-D','F-R'):
                for metric,path in METRICS.items():
                    x,y=extract(old,comparison,path),extract(new,comparison,path)
                    if x['p95'] is None or y['p95'] is None:continue
                    records.append({'contrast':label,'comparison':comparison,'metric':metric,'window':row['window'],
                        'method':row['method'],'participant_group':windows[row['window']]['group_id'],'subset':windows[row['window']]['subset'],
                        'before_p95':x['p95'],'after_p95':y['p95'],'delta_p95':y['p95']-x['p95'],
                        'before_n':x['n'],'after_n':y['n'],'denominator':x['denominator']})
        for method in sorted({r['method'] for r in panel}):
            for comparison in ('F-D','F-R'):
                for metric in METRICS:
                    rows=[r for r in records if r['contrast']==label and r['method']==method and r['comparison']==comparison and r['metric']==metric]
                    if rows:
                        groups={g:statistics.median(r['delta_p95'] for r in rows if r['participant_group']==g) for g in {r['participant_group'] for r in rows}}
                        summaries.append({'contrast':label,'method':method,'comparison':comparison,'metric':metric,
                            'paired_windows':len(rows),'participant_group_median_delta_p95':groups,
                            'median_window_delta_p95':statistics.median(r['delta_p95'] for r in rows),
                            'worst_increase_p95':max(r['delta_p95'] for r in rows)})
    result={'completion':completion,'panel_windows':len(panel),'contrasts':summaries,'records':records,
            'interpretation':'Differences of paired window P95; no additive variance attribution or contribution percentage. Fixed common-baseline time intersection; full D and original-clock results retained in score files.',
            'promotion':'none','head_neck':'native readout separate; model F has no independent head joint'}
    (a.run/'review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    failed=[r for r in completion['results'] if r['status']!='completed']
    gates=json.loads((a.run/'acceptance.json').read_text())['comparisons']
    blocked=sum(r['paired']['status']=='regression_blocked' for r in gates)
    lines=['# K4 固定面板消融数据摘要','',f"固定面板 {len(panel)} 个格式—窗口组合；应有 {completion['expected']} 个处理任务，失败 {len(failed)} 个。",'',
           '以下列出同窗口 P95 的差值，负值代表下降。比较固定几何、姿态细化、数值处理和 root 平移；分位数差值不解释为可相加的误差贡献。完整 D 时钟、原始时钟和失败分母保留。',
           '',f'共检查 {len(gates)} 个条件—窗口—root 结果，{blocked} 个触发配对回归门槛。其余仅表示未触发规定回归，还需检查改善、目标达标和适用范围。',
           '','原动作 R 用于独立验收；运行时拟合、root 及模型扫描只读取 D 和声明资产。头胸原生读出单列，当前拟合模型没有独立头关节。',
           '',f'完整逐记录证据：`{a.run}/review.json`；逐指标接受／拒绝依据：`{a.run}/acceptance.json`。',
           '','<details><summary>按格式展开配对误差变化</summary>','',
           '| 对照 | 格式 | 指标 | 配对窗口数 | F−R 窗口P95差值中位数 | 最大增加 |','|---|---|---|---:|---:|---:|']
    for r in summaries:
        if r['comparison']=='F-R':lines.append(f"|{r['contrast']}|{r['method']}|{r['metric']}|{r['paired_windows']}|{r['median_window_delta_p95']:.3f}|{r['worst_increase_p95']:.3f}|")
    lines+=['','</details>','','下一步结合逐阶段 FK 和关节域扫描复核误差转移，完成局部配置调整后再进行完整开发回归；本摘要不自动授予配置推荐资格。']
    content='\n'.join(lines)+'\n';a.report.parent.mkdir(parents=True,exist_ok=True)
    if a.report.exists() and a.report.read_text()!=content:raise ValueError('Preserve prior report; choose a new report filename')
    a.report.write_text(content)


if __name__=='__main__':main()
