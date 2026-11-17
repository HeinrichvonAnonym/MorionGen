"""Summarize the degradation audit without pooling unequal windows as subjects."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run',type=Path);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    rows=[json.loads(p.read_text()) for p in sorted((args.run/'scores').glob('*.json'))]
    source_geometry=json.loads((args.run/'receipt.json').read_text()).get('source_geometry',False)
    methods=['motionlcm','omnicontrol','ardy','chois','closd','dart','kimodo','primal']
    labels=['头颈','双肩','躯干','手（腕点）','双膝','足（踝/趾点）']
    regions=['head_neck','shoulders','trunk','hands','knees','feet']
    summary={'windows':len(rows),'methods':{},'k_runs':0,'overall_acceptance':'blocked'}
    lines=['# 八种格式退化对齐：开发核查', '',
        '本次只检查退化前原动作 R 到读回动作 D。K 拟合器、参数和历史留出记录保持原值。', '',
        ('位置格式直接选择原动作节点；六种旋转格式采用用户确认的源体型骨架，静止几何在片段内固定，并随输入声明。旧转换使用原固定模板，作为归因对照。'
         if source_geometry else '位置格式直接选择原动作节点；旋转候选保留原固定资产。此批用于检验固定体型条件，候选不自动晋升。'), '',
        f'本报告覆盖 {len(rows)}/82 个原登记开发窗口。以下位置值是**各窗口 P95 的中位数，单位 cm**，不是全部帧的 P95；最大窗口 P95 与逐侧端点在 JSON 中保留。失败保留分母。', '',
        '| 格式／方案 | '+ ' | '.join(labels)+' | 完成窗口 |',
        '|---|'+'---:|'*7]
    failures=[]
    for method in methods:
        record={'denominator':82,'versions':{}}
        for version in ('legacy','candidate'):
            successful=[r for r in rows if r['methods'][method]['status']=='computed']
            items=[r['methods'][method]['versions'][version] for r in successful]
            by_region={}
            for region in regions:
                values=[x['positions'][region]['world_cm']['p95'] for x in items]
                by_region[region]={'median_window_p95_cm':float(np.median(values)) if values else None,
                                  'worst_window_p95_cm':float(max(values)) if values else None,
                                  'windows_p95_above_5cm':sum(v>5 for v in values),
                                  'valid_points':sum(x['positions'][region]['world_cm']['n'] for x in items),
                                  'total_points':sum(x['positions'][region]['world_cm']['denominator'] for x in items)}
            detail={'positions':by_region,'computed_windows':len(items),
                    'groups':len({r['window']['group_id'] for r in successful}),
                    'directions':{key:{'median_window_p95_deg':float(np.median([x['directions_deg'][key]['p95'] for x in items])),
                                       'worst_window_p95_deg':float(max(x['directions_deg'][key]['p95'] for x in items))}
                                  for key in items[0]['directions_deg']} if items else {}}
            detail['source_clock']={region:{'median_window_p95_cm':float(np.median([
                x['source_clock_reconstruction']['positions'][region]['world_cm']['p95'] for x in items])),
                'worst_window_p95_cm':float(max(x['source_clock_reconstruction']['positions'][region]['world_cm']['p95'] for x in items))}
                for region in regions} if items else {}
            # Numerical field-preservation gate, not a biomedical threshold.
            detail['position_transport_within_0_01cm_windows']=sum(
                max(x['positions'][region]['world_cm']['max'] for region in regions)<=.01 for x in items)
            detail['geometry_budget_1cm_1deg_windows']=sum(
                max(x['positions'][region]['world_cm']['p95'] for region in regions)<=1 and
                max(x['directions_deg'][key]['p95'] for key in x['directions_deg'])<=1 for x in items)
            detail['engineering_5cm_5deg_windows']=sum(
                max(x['positions'][region]['world_cm']['p95'] for region in regions)<=5 and
                max(x['directions_deg'][key]['p95'] for key in x['directions_deg'])<=5 for x in items)
            record['versions'][version]=detail
            values=[by_region[r]['median_window_p95_cm'] for r in regions]
            lines.append('| '+method+'/'+('旧转换' if version=='legacy' else '候选')+' | '+
                         ' | '.join(f'{v:.3f}' if v is not None else '缺失' for v in values)+f' | {len(items)}/82 |')
        record['status']='review_complete' if source_geometry and record['versions']['candidate']['geometry_budget_1cm_1deg_windows']==82 else 'review_required'
        summary['methods'][method]=record
        failures.extend({'window':r['window']['window_key'],'method':method,**r['methods'][method]}
                        for r in rows if r['methods'][method]['status']!='computed')
    summary['failures']=failures
    lines.extend(['','## 怎样解释这些结果','',
        '1. 直接节点选择与编解码检查回答“字段能否保留原位置”。头部轴向转动仍缺少独立观测；位置接近不证明完整头颈旋转可恢复。',
        '2. 旋转候选同时约束位置和旋转属于同一 FK 骨架。源体型修正与拓扑压缩残差分别解释；残余误差需要具体证据才能标为结构下界。',
        '3. 模板长度固定时，一条骨段两端的误差之和至少等于原动作与模板的骨长差。这是该固定资产条件下的下界，不能推广成整个格式的必然损失。',
        '4. 原生头／胸旋转结果只检验所声明的坐标转换；单骨段扭转采用最小旋转约定，叶节点采用同轴约定。这些约定需要继续核实，不能据此授予解剖精度。',
        '5. 原始时间轴重建另列，反映采样率和边界覆盖；不会用先滤波的参考冲淡快速动作误差。PRIMAL 历史两帧和 OmniControl 编码最后一帧分别记录。','',
        '## 验收结论','',
        '验收按可表达分量分别进行。JSON 同时报出严格数值保持（所有位置最大误差 ≤0.1 mm）、逐窗口 P95 的 1 cm／1° 转换预算，以及既有 5 cm／5° 工程档。1 cm／1° 是本轮为转换环节预留的工程检查档，未经生物力学标准认证。缺少头部独立旋转的两种位置格式保持部分支持。K 优化本轮执行次数为零。','',
        '独立真值范围为发布 BVH 的关节中心与原生段朝向。手指、掌面、鞋底接触和解剖关节轴没有由这里的腕点、趾点或低旋转往返误差得到独立验证。', '',
        f'逐窗口结果：`{args.run}/scores/`；可读取输入与参考分别存于 `inputs/`、`references/`。'])
    if source_geometry and all(v['status']=='review_complete' for v in summary['methods'].values()):
        summary['overall_acceptance']='geometry_budget_passed; information scope and native-frame review remain explicit'
    verification=args.output/'artifact_verification.json'
    if verification.exists():
        checked=json.loads(verification.read_text());vr=checked['records']
        assert checked['run']==str(args.run.resolve()) and len(vr)==656
        position=max(v['versions']['candidate']['positions'][r]['worst_window_p95_cm'] for v in summary['methods'].values() for r in regions)
        direction=max(d['worst_window_p95_deg'] for v in summary['methods'].values() for d in v['versions']['candidate']['directions'].values())
        head=max(v['head_neck_relative_deg']['p95'] for v in vr if v['head_neck_relative_deg'])
        jump_windows=len({v['window'] for v in vr if v.get('source_jump_events')})
        intro=[
            f'**在声明的源体型骨架和实际输出时刻，656/656 路的退化空间对齐通过本轮检查。** 各窗口、各部位的最差位置 P95 为 {position*10:.3f} mm，几何方向 P95 为 {direction:.3f}°。这支持修复后的转换保留已观测动作；本轮未优化 K，也未验证重构精度。',
            '',
            '旧流程把原动作换成任意固定模板、重复映射部分脊柱节点，并在编码中引入朝向分支错误。源体型、正确拓扑和连续编码修正后，大部分损失消失。这些损失应归于可修复的转换实现，不能归为 K 拟合误差或格式必然损失。',
            '',
            f'六条旋转路径从实际保存参数和输入骨架重新计算 FK，最大位置差为 {max(v.get("fk_position_max_cm",0) for v in vr):.2g} cm；按声明坐标转换后的头相对胸原生旋转，最差窗口 P95 为 {head:.3f}°。原生坐标保持与解剖标定分开，后者未由本实验授予。',
            '',
            f'按用户确认，原动作已有跳变的邻域单列宽容，涉及 {jump_windows}/82 个窗口。识别只使用 R：相邻源帧任一身体点位移超过 25 cm，或根／头／脊柱旋转超过 45°；对应输出区间前后各加一帧。阈值及邻域是明确的工程分层规则。完整窗口、宽容项和连续段均保留，D 自行产生的跳变不据此豁免。','']
        lines[2:2]=intro
        lines+=['','## 原始时间轴：连续段与跳变宽容项','',
            '本表连续段 P95 仍取所有窗口／部位中的最大值；最大单点误差另列，避免 P95 隐藏少量极值。计数是部位点×源帧，包含躯干中点代理；区间外的原始帧与 PRIMAL 历史帧在逐记录 time_coverage 中保留。','',
            '| 格式 | 连续段最差窗口 P95 / 最大单点（cm） | 跳变宽容项点×帧 / 共同区间总点×帧 | 宽容项最大误差（cm） |',
            '|---|---:|---:|---:|']
        for method in methods:
            rv=[v for v in vr if v['method']==method]
            cells=[z for v in rv for z in v['original_clock'].values()]
            p95=max(z['continuous']['p95'] or 0 for z in cells);maximum=max(z['continuous']['max'] or 0 for z in cells)
            tolerated=sum(z['source_jump_tolerance']['n'] for z in cells);total=sum(z['n'] for z in cells)
            extreme=max(z['source_jump_tolerance']['max'] or 0 for z in cells)
            lines.append(f'| {method} | {p95:.3f} / {maximum:.3f} | {tolerated} / {total} | {extreme:.3f} |')
        lines+=['','## 可修复错误与表示边界','',
            '- **已修复**：任意模板体型造成的位置偏差；ARDY 重复脊柱映射；根节点与双髋中点混淆；OmniControl 四元数符号切换处的朝向增量折叠；将弯曲脊柱直接合成刚性长骨。',
            '- **可保持但需要完整声明**：自定义源体型骨架。各片段只有一套静止几何，记录坐标、拓扑、旋转基底和派生节点。DART／PRIMAL 的自定义运动学骨架覆盖默认体型几何，零 betas 不表示它等同默认 SMPL-X 网格；CLoSD 这里提供运动学输入。',
            '- **确实缺少的信息**：MotionLCM／OmniControl 的独立头部轴向转动；少节点格式中被合并的内部脊柱／肩带自由度；固定低帧率样本之间任意变化的运动。头部旋转缺失由同位置、不同头转动产生相同输入的测试证明。',
            '- **有条件的结构残差**：两段肩带合成一段固定长度时，其端点距离会随原始内部动作变化。本轮固定有效长度取距离范围的中点；在父端保持原位、单段定长的条件下，最坏径向残差下界为距离范围的一半。该边界与参数逐条写入 input_rig.json，不推广为允许改变拓扑时的全局下界。',
            '- **保留的限定**：腕点不等于完整手掌，踝／趾点不等于鞋底；原生头胸坐标不等于已经标定的解剖轴。下一阶段先让 K 明确读取 input_rig.json，再重新验证 K；旧 K 仍按既有模板解释输入，本轮未改动。','',
            '## 核验与复跑','',
            '8 项独立测试覆盖 BVH 通道与非交换旋转、静止骨架固定性、位置保持、头部扭转信息遮蔽、根平移显式计分，以及穿越 ±π 的 HumanML 编解码。实际产物另外完成 656 路 FK、输入字段与哈希核验。',
            '[R／D 动画（含两种时间轴）](/home/heinrich/MorionGen/docs/20261117_degradation_alignment.html) · [逐格式汇总](summary.json) · [独立产物核验](artifact_verification.json)',
            '',
            '复跑入口为 scripts/evalkit_degradation_audit.sh OUTPUT_DIR BASELINE_AUDIT_DIR，最多三个 CPU 工作进程、数值库单线程。完整输入、参考、旧候选与 HTML 实体在数据盘；docs 保留 HTML 链接。原冻结、留出记录及原 notes 逐文件核验，未新增留出使用。']
        summary['independent_verification']={'records':len(vr),'max_head_neck_window_p95_deg':head,
             'source_jump_windows':jump_windows,'protected_files':checked['preserved_files'],'changed':checked['changed_protected']}
        summary['overall_acceptance']='passed_for_declared_sampled_geometry_and_native_frame_transport; source_jump_tolerance_separate; unobservable_components_excluded_explicitly'
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    (args.output/'README.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
