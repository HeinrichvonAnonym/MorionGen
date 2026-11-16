#!/usr/bin/env python3
"""Render the completed development evidence as a concise research report."""
import argparse,json,sys,collections
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'python/scripts'))
from evalkit_k3_report import record_score,participant_mean
LABEL={'legacy_raw':'原拟合／原输入','legacy_projected':'原拟合／投影输入','optimized_raw':'方向优化／原输入','optimized_projected':'方向优化／投影输入'}
METHODS=['motionlcm','omnicontrol','ardy','chois','closd','dart','kimodo','primal']
REGIONS={'pelvis':'骨盆线','torso':'躯干','left_arm':'左臂','right_arm':'右臂','left_leg':'左腿／足','right_leg':'右腿／足'}
fmt=lambda x:'—' if x is None else f'{x:.2f}'
pct=lambda x:'—' if x is None else f'{100*x:.1f}%'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();cfg=ROOT/'config/evalkit_k3_20261116'
    summary=json.loads((cfg/'results_summary.json').read_text());methods=summary['methods']
    frozen=json.loads((cfg/'experiment_freeze.json').read_text())['commit']
    if summary.get('partial'):raise ValueError('A partial experiment cannot become the completed report')
    actual={r['method']:r for r in json.loads((a.out/'actual_summary.json').read_text())}
    spatial=json.loads((cfg/'spatial_effects_summary.json').read_text())
    revision=json.loads((cfg/'scoring_revision.json').read_text())
    figures=json.loads((ROOT/'develop_logs/notes/20261116_k3_figures/index.json').read_text())
    tails=json.loads((cfg/'tail_diagnostics.json').read_text())['methods']
    selection=json.loads((cfg/'selection.json').read_text())['records'];rows=[json.loads(Path(p).read_text()) for p in summary['result_paths']]
    lines=['# Evalkit K3：来源约定、统一骨长对照与姿态优化','',
      '本轮检验的是：原始动作经过八种格式退化，再由 evalkit 拟合后，姿态信息保留到什么程度；骨长变化和姿态拟合各自带来了什么误差。验收参考始终来自退化前的原始 BVH。Andy/Han 的独立光学证据限定在头胸，身体模板保留先验属性。',
      '', '首要发现是原始参考的读取约定：CoLLaboratE 的平移通道已提供局部位置，原实现再次加上 OFFSET，使典型站立记录的关节点垂直跨度从约 1.53 米变为 3.06 米。K3 已按独立导入器支持的通道语义修正，并重新生成全部受控输入。此前较大的位置差异包含读取误差，原先主要归为骨长差异的解释需要修正。',
      '', '本报告中的求解改善均比较修正后同一批输入上的原拟合与方向优化。旧冻结结果、原留出使用记录及此次被替代的 166 个结果文件全部保留；参考修正与求解改善分别登记。旧 CoLLaboratE 数值继续归属原读取约定，K3 的验收采用修正后的开发对照。[来源修正证据](/home/heinrich/MorionGen/config/evalkit_k3_20261116/reference_translation_correction.json)',
      '', '**开发结论：方向优化降低了八种格式的全身姿态误差，改善集中在双臂和双腿。** LAB 按人汇总的端到端角度 P95 从 5.02–8.09° 降到 1.51–2.93°；工业迁移从 7.37–9.50° 降到 2.50–4.18°。八路最终均选择保留原输入。该结论来自原始参考、相同输入和固定评分窗口，覆盖修正另行统计。',
      '', '**剩余误差需要沿转换与拟合两段解释。** 一部分姿态偏差已在目标骨架转换中产生：例如 MotionLCM 躯干的降级损失为 6.89°，最终拟合相对输入的偏差为 1.21°，端到端仍为 6.61°。拟合阶段还要同时观察局部姿态失败和角度优先带来的空间代价：Kimodo 的双足低于共同参考面占比从 6.7% 增至 36.2%。因此本版支持姿态几何评估，并继续用标定与位置条件限制力学入口。',
      '', '**局部高误差继续作为受限结果保留。** Kimodo 的一个跪姿窗口，右腿误差汇总从 58.57° 增到 97.35°，该窗口右腿的输入方向支持为 0%。全身均值下降无法替代逐部位验收；下文同时列出尾部记录、方向支持、模型约束和专门叠加图。',
      '', '**统一骨长有助于解释位置差异；此次受控输入的投影消融主要检验了数值稳定性。** 受控动作由固定骨架 FK 构造，投影前后关节位置几乎相同。真实 MotionLCM、OmniControl、DART 样例具有更明显的逐帧骨长变化，投影可引起厘米级手脚位移。两类证据分别报告，当前开发结果支持本版采用原输入，投影在更广泛真实输出中的适用性仍需更多样例。',
      '',f'已登记全量轻量检查 1,670 条开发资料；固定 {len(selection)} 个原始动作窗口、{len({r["group_id"] for r in selection})} 个参与者组，覆盖八种格式与四种拟合设置。本轮完成 {summary["completed"]}/{summary["expected_exports"]} 个窗口×格式结果，共 2,624 次拟合，运行失败 {summary["failed"]} 个。计算完成与姿态精度分别验收，局部失败解保留在误差统计中。',
      '', '配置选择以按人聚合的端到端几何角 P95 为主，包括骨盆线与躯干倾斜、抬臂、肘膝屈曲、髋躯干夹角及足部链条角。每条记录先计算各角度 P95，再按骨盆、躯干、左右臂、左右腿六个部位等权汇总，最后按人等权。表中的“角度 P95 汇总”采用这一口径，区别于把所有帧混合后取一次 P95。三维骨方向和独立段旋转另列，保留对朝向变化的检查。',
      '', '| 格式 | 配对窗口 | 选定设置 | K2 角度 P95 汇总 → K3 / ° | 留人折基准 → 选型 / ° | 工业迁移基准 → K3 / ° |',
      '| --- | ---: | --- | ---: | ---: | ---: |']
    for m in METHODS:
        e=methods[m];base=e['variants']['legacy_raw'];new=e['variants'][e['selected']]
        lines.append(f'| {m} | {e["paired_records"]} | {LABEL[e["selected"]]} | {fmt(base["body_pose_score"])} → {fmt(new["body_pose_score"])} | {fmt(e["fold_mean_legacy_score"])} → {fmt(e["fold_mean_score"])} | {fmt(base["industrial_pose_score"])} → {fmt(new["industrial_pose_score"])} |')
    lines+=['','十个 LAB 留人开发折用于检查跨人稳定性；最终配置用全部 LAB 开发窗口确定。Airplane、Glass、Mastic、Silk 的迁移结果单列。原冻结版本及已使用的 Andy／电视留出记录保留，本轮继续属于开发验证。',
      '', '![八种格式的角度改善与误差来源](/home/heinrich/MorionGen/develop_logs/notes/20261116_k3_figures/overview.png)','',
      '## 误差从哪里产生','',
      'R 是原始参考，D 是退化文件读回，P 是拟合输入投影，F 是最终 OpenSim 坐标经 FK 得到的关节中心。D−R 衡量格式退化；F−D 衡量拟合相对输入的偏差；F−R 决定端到端验收。P−D 与 F−P 进一步说明投影做了什么。各项使用相同物理时间与共同可观测窗口，支持掩码另列。',
      '', '本实验的降级链包含项目规定的目标骨架转换、降采样前滤波和格式序列化。降采样时沿用四阶 Butterworth 双向滤波，截止频率取 6 Hz 与目标采样率 0.4 倍中的较小值；矩阵滤波后恢复合法旋转矩阵，再插值到目标时钟。这属于受控输入构造约定。因此 D−R 的差异需要结合骨架、滤波和采样率解释，八条分支在此检验的是格式接入路径。真实模型已生成动作的保真度在后表独立核查。',
      '', '| 格式 | 双臂：D−R / F−D / F−R，° | 双腿：D−R / F−D / F−R，° | 躯干：D−R / F−D / F−R，° |',
      '| --- | ---: | ---: | ---: |']
    for m in METHODS:
        e=methods[m];v=e['variants'][e['selected']];parts=[]
        for regs in [['left_arm','right_arm'],['left_leg','right_leg'],['torso']]:
            parts.append(' / '.join(fmt(float(np.mean([v['regions'][r][s] for r in regs]))) for s in ['degradation','fit_deviation','end_to_end']))
        lines.append('| '+m+' | '+' | '.join(parts)+' |')
    lines+=['','双侧在上表等权汇总；逐侧、骨盆、足部及每条记录的中位数、P95、系统偏差和 5°／10°／20°覆盖均保存在结果中。三个工程精度档用于解释表示误差，生物力学解释还依赖各指标的观测与轴定义。范数、P95 和旋转距离分别统计；误差向量可能相互抵消，因而不将三个 P95 相加。',
      '', '三维骨方向的夹角保留世界朝向差异，补充仅由屈曲量或倾斜量构成的几何角。下表为端到端方向 P95 汇总，左右取较大值；一条骨方向提供两个姿态分量，绕该骨自身的轴向转动由独立段旋转证据限定。',
      '', '| 格式 | 躯干 / ° | 上臂 / ° | 前臂 / ° | 大腿 / ° | 小腿 / ° | 足部 / ° |',
      '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for m in METHODS:
        e=methods[m];d=e['variants'][e['selected']]['directions']
        values=[d['shoulder_mid']['end_to_end']['p95_abs']]
        values += [max(d[s+'_'+joint]['end_to_end']['p95_abs'] for s in ['left','right']) for joint in ['elbow','wrist','knee','ankle','foot']]
        lines.append('| '+m+' | '+' | '.join(fmt(x) for x in values)+' |')
    lines+=['', '## 骨长：评分归一与输入投影分别回答问题','',
      '评分端把 R、D、F 按各自骨方向重建到原始采样窗口的共同分段中位长度，左右侧分别保留，同一窗口的八种格式共用同一组骨长。公共评价骨架以双髋中点为根，保持各自根运动；另报去根平移的位置误差。参考骨长只进入评分器。输入投影则只使用 D 的逐段长度中位数，按原生层级建立固定骨长视图，头胸原生旋转继续来自原输入。',
      '', f'评分复查将骨长统计范围从各格式的重采样参考窗口统一为原始采样窗口，最大分段长度修正为 {revision["maximum_length_change_m"]*1000:.3f} 毫米。原四组拟合及初始评分文件保留，{revision["completed"]} 组重评分中未经支持掩码筛选的原始几何指标逐项一致，配置选择依据保持原值。CHOIS 的观测标记修正单列在下文。[评分修订记录](/home/heinrich/MorionGen/config/evalkit_k3_20261116/scoring_revision.json)',
      '', '| 格式 | 原拟合：投影关 → 开，° | 优化拟合：投影关 → 开，° | 投影造成的手部世界位移 P95 / cm |',
      '| --- | ---: | ---: | ---: |']
    for m in METHODS:
        e=methods[m];v=e['variants'];dis=max(v['optimized_projected']['positions'][s+'_wrist']['raw']['projection'] for s in ['left','right'])
        lines.append(f'| {m} | {fmt(v["legacy_raw"]["body_pose_score"])} → {fmt(v["legacy_projected"]["body_pose_score"])} | {fmt(v["optimized_raw"]["body_pose_score"])} → {fmt(v["optimized_projected"]["body_pose_score"])} | {fmt(dis*100)} |')
    lines+=['', '这里的 0.00 cm 有明确含义：七路受控输入的投影位移约为浮点舍入量级；OmniControl 经编码读回后，按人汇总的手脚最大位移 P95 约为 1.35×10⁻⁷ m。如此小的输入变化仍能改变少数求解结果，体现初值和非线性求解分支的数值敏感性，不能归因为消除了实质骨长变化。PRIMAL 有一个留人折选择投影，其余 79 个格式×留人折选择原输入；最终全 LAB 选型均为原输入。',
        '', '因此，受控投影开关的结果属于“近恒等输入下的稳定性对照”；真实输出表才直接显示随帧骨长变化被投影后的影响。四组轨迹和逐记录差异全部保留。']
    lines+=['','下表给出双手中较大的世界位置误差；每格依次为 D−R／F−D／F−R。统一骨长后仍保留的世界位置差异包含根运动、方向及关节中心对应误差；去根平移列进一步分离整体平移。',
      '', '| 格式 | 原几何 / cm | 统一骨长 / cm | 统一骨长且去根平移 F−R / cm |',
      '| --- | ---: | ---: | ---: |']
    for m in METHODS:
        e=methods[m];p=e['variants'][e['selected']]['positions'];spaces=[]
        for space in ['raw','normalized']:
            spaces.append(' / '.join(fmt(100*max(p[s+'_wrist'][space][stage] for s in ['left','right'])) for stage in ['degradation','fit_deviation','end_to_end']))
        rel=max(p[s+'_wrist']['normalized_root_relative_end_to_end'] for s in ['left','right'])
        lines.append(f'| {m} | {spaces[0]} | {spaces[1]} | {fmt(rel*100)} |')
    lines+=['','纯评分长度归一保持公共骨架的骨方向和肘膝夹角。输入投影作用于原生层级；跨多个原生骨段定义的躯干轴、虚拟中点等量仍可能变化，P−D 已保留这些变化。骨长归因保存精确的“原始误差向量减去归一误差向量”，其范数单列，避免将两个 P95 的差解释为贡献百分比。OpenSim 现有分段缩放仍采用左右平均长度，评分端保留左右差异，因此两侧位置代价继续可见。',
      '', '## 实际修改与支持范围','',
      '新增方向优化从固定的 K2 初始化流程出发，用输入骨段方向约束骨盆、躯干、双臂和双腿，同时保留较弱的位置项、连续性项和初值先验。优化在 20 Hz 上求解并连续插值到 100 Hz；输出由实际 OpenSim 坐标和模型 FK 得到。方向、位置、连续性权重分别为 1、0.05、0.002，冻结代码中的初值先验权重为 0.001，单次求解上限为 35 次函数评估。这些是本轮检验的工程设置，未作为推导得到的生物力学参数；运行时只按输入残差选择初值和解。未观测的轴向转动保留初值先验。模型原有的距下关节锁定保持原状，该变量的列名不代表新增了可自由拟合的足部转动。',
      '', '下表的角度支持是对输入骨方向的保真检查，原始参考精度由 F−R 单独确定。配准角度支持还要求片段满足直立标定条件；配准位置支持还检查端点位移。三种覆盖各有用途。',
      '', f'CHOIS 的额外手点与 body wrist 使用不同观测来源。原拟合视图只转换了位置名称，额外手点的先验标记随后误落到 body wrist 上，过度收紧了手臂支持。本轮同步转换名称与观测标记，{revision["support_changed_records"]} 条受控记录的支持统计因此更新。四种设置都按同一规则重算；这项覆盖修正与姿态拟合改善分别归因，完整评分窗口保持原值。',
      '', '| 格式 | 角度支持，六部位均值 | 配准角度支持 | 配准位置支持 | 相对基准退化 >1° 的记录 |',
      '| --- | ---: | ---: | ---: | ---: |']
    for m in METHODS:
        e=methods[m];c=e['variants'][e['selected']]['coverage']
        vals=[float(np.mean([v[k] for v in c.values()])) for k in ['angle','registered_angle','registered_position']]
        lines.append(f'| {m} | {pct(vals[0])} | {pct(vals[1])} | {pct(vals[2])} | {len(e["regressions"])} / {e["paired_records"]} |')
    lines+=['', '上述全身汇总中没有增加超过 1° 的记录，局部部位仍有显著退化。下表按同一原始参考列出各部位增加超过 1° 的记录数，顺序为骨盆／躯干／左臂／右臂／左腿／右腿，每种格式分母均为 82；右列列出本格式残留误差最高的部位及其原输入方向支持。',
        '', '| 格式 | 六部位退化记录数 | 局部高误差记录／部位 | 该部位基准 → K3 P95 汇总 / ° | 输入方向支持 |',
        '| --- | --- | --- | ---: | ---: |']
    for m in METHODS:
        e=tails[m];reg=max(e['regions'],key=lambda k:e['regions'][k]['highest_remaining_error']['regions'][k]['selected'])
        case=e['regions'][reg]['highest_remaining_error'];value=case['regions'][reg]
        counts=' / '.join(str(len(e['regions'][r]['regressed_records'])) for r in REGIONS)
        support=e['worst_region_constraints']['input_direction_support']['share']
        lines.append(f'| {m} | {counts} | {Path(case["id"]).stem}／{REGIONS[reg]} | {fmt(value["baseline"])} → {fmt(value["selected"])} | {pct(support)} |')
    lines+=['', '这些案例把残差进一步定位到求解阶段。Kimodo 跪姿峰值处，输入右膝几何屈曲约 97.9°，输出接近伸直；髋屈、内收、轴转与踝坐标同时触及范围边界。CHOIS／PRIMAL 的铆接案例则同时触及肩内收和轴转边界。边界、初值和多项方向目标的相互作用值得后续单独消融；当前证据能够确认失败解和低支持范围，尚不足以将全部误差归因于某一个约束。',
        '', '近恒等投影也暴露了求解敏感性：部分窗口仅有浮点量级输入变化，角度汇总仍变化约 4.8°；OmniControl 的一个工业窗口约变化 9.3°。这些记录的初始手臂候选类型保持相同，故候选类别本身不能解释全部差异。该结果按数值稳定性局限登记，保留原输入策略及逐部位保真门。[逐记录尾部、投影敏感性与约束诊断](/home/heinrich/MorionGen/config/evalkit_k3_20261116/tail_diagnostics.json)',
        '', '![Kimodo 跪姿右腿的局部高误差](/home/heinrich/MorionGen/develop_logs/notes/20261116_k3_figures/tail_kimodo_'+next(r['window_key'] for r in tails['kimodo']['records'] if r['id']==tails['kimodo']['regions']['right_leg']['highest_remaining_error']['id'])+'.png)']
    lines+=['','姿态优化的空间代价继续进入下游：K3 保存逐部位掩码，力学入口要求全身角度与位置配准条件同时满足；下肢肌肉主结果另与残差判定取交集，全帧强制结果保留。足部近地、低于参考面和近地滑动速度按共同输入参考面报告，它们是关节点接触代理，实测反力与场景支撑仍需后续独立验证。',
      '', '| 格式 | 基准 → K3：足部低于参考面占比 | 基准 → K3：近地水平速度 / m/s |',
      '| --- | ---: | ---: |']
    for m in METHODS:
        e=methods[m];vals=[]
        for field in ['below_floor_share','speed_when_near_floor_mps']:
            pair=[]
            for v in ['legacy_raw',e['selected']]:
                numbers=[e['variants'][v]['contact'][s][field] for s in ['left','right']];numbers=[x for x in numbers if x is not None]
                pair.append(float(np.mean(numbers)) if numbers else None)
            render=pct if field=='below_floor_share' else fmt;vals.append(' → '.join(render(x) for x in pair))
        lines.append('| '+m+' | '+' | '.join(vals)+' |')
    lines+=['','角度选型同时记录手脚的世界位置代价。下表取左右较大值，均相对原始参考；这使角度改善与末端位置变化可以分别验收。',
      '', '| 格式 | 手：基准 → K3，原几何 P95 / cm | 足：基准 → K3，原几何 P95 / cm | 足：K3 统一骨长且去根平移 P95 / cm |',
      '| --- | ---: | ---: | ---: |']
    for m in METHODS:
        e=methods[m];cells=[]
        for joint in ['wrist','foot']:
            cells.append(' → '.join(fmt(100*max(e['variants'][v]['positions'][s+'_'+joint]['raw']['end_to_end'] for s in ['left','right'])) for v in ['legacy_raw',e['selected']]))
        foot=100*max(e['variants'][e['selected']]['positions'][s+'_foot']['normalized_root_relative_end_to_end'] for s in ['left','right'])
        lines.append(f'| {m} | {cells[0]} | {cells[1]} | {fmt(foot)} |')
    lines+=['','## 真实基线输出单独验收拟合保真度','',
      '这八份已生成动作没有配对原始动作 R，因此只检验生成输出 G 与拟合 F 的几何保真度。下表的肘、膝误差取左右较大的单条 P95；结果不并入受控退化的原始参考精度。',
      '', '| 格式 | 状态 | 肘：基准 → 选定设置 / ° | 膝：基准 → 选定设置 / ° |',
      '| --- | --- | ---: | ---: |']
    for m in METHODS:
        r=actual[m];v=methods[m]['selected'];cells=[]
        if r['status']=='computed' and v in r['scores']:
            for joint in ['elbow','knee']:
                cells.append(' → '.join(fmt(max(r['scores'][strategy]['raw']['angles'][s+'_'+joint+'_flexion']['fit_fidelity']['p95_abs'] for s in ['left','right'])) for strategy in ['legacy_raw',v]))
        else:cells=['—','—']
        lines.append(f'| {m} | {r["status"]} | {cells[0]} | {cells[1]} |')
    lines+=['','真实输出中的骨长随帧变化，与固定骨架 FK 构造的受控输入有不同分布。投影消融同时保留在这八份样例中；下表比较方向优化的投影开关，并给出输入本身的位移，结果用于说明接入代价。',
      '', '| 格式 | 手部 G→P 位移 P95 / cm | 足部 G→P 位移 P95 / cm | 肘保真 P95：投影关 → 开 / ° | 膝保真 P95：投影关 → 开 / ° |',
      '| --- | ---: | ---: | ---: | ---: |']
    for m in METHODS:
        r=actual[m];s=r['scores'];cells=[]
        if all(v in s for v in ['optimized_raw','optimized_projected']):
            for joint in ['wrist','foot']:
                cells.append(fmt(100*max(s['optimized_projected']['raw']['positions'][side+'_'+joint]['input_projection']['world']['p95_abs'] for side in ['left','right'])))
            for joint in ['elbow','knee']:
                cells.append(' → '.join(fmt(max(s[v]['raw']['angles'][side+'_'+joint+'_flexion']['fit_fidelity']['p95_abs'] for side in ['left','right'])) for v in ['optimized_raw','optimized_projected']))
        else:cells=['—']*4
        lines.append('| '+m+' | '+' | '.join(cells)+' |')
    lines+=['','输入投影引起的速度变化在相同时间网上计算，下表给出速度向量差的 P95；接触变化使用原输入的同一参考面。该指标描述表示变换的代价，肌肉与外力精度仍需独立力学证据。',
      '', '| 格式 | 手部速度变化 P95 / m/s | 足部速度变化 P95 / m/s | 足部近地判定切换占比 |',
      '| --- | ---: | ---: | ---: |']
    for m in METHODS:
        p=spatial['actual_output_fidelity'][m]['variants']['optimized_projected']['projection']
        hand=max(p[s+'_wrist']['velocity_difference_mps'] for s in ['left','right'])
        foot=max(p[s+'_foot']['velocity_difference_mps'] for s in ['left','right'])
        switch=max(p[s+'_foot']['near_plane_switch_share'] for s in ['left','right'])
        lines.append(f'| {m} | {fmt(hand)} | {fmt(foot)} | {pct(switch)} |')
    lines+=['','方向优化未投影样例的残留腿部误差还与模型允许的运动有关。OmniControl、CLoSD、PRIMAL 的一些膝误差峰值出现在模型接近伸直时；Kimodo 的一个峰值同时达到髋轴转动上界。距下关节（踝下负责足部侧向转动的关节）保持锁定，足部与膝部的方向目标需共同适配这一自由度限制。[逐侧模型约束诊断](/home/heinrich/MorionGen/config/evalkit_k3_20261116/constraint_diagnostics.json)给出边界和峰值事实；各约束的独立误差贡献还需另行消融。']
    lines+=['','## 轴定义、头胸与既有冻结证据','',
      '来源核查区分了原生模型、项目公共归档和 K 内部坐标。已明确的轴转换继续固定使用；本轮未通过重新估计轴来降低误差。Kimodo 的 SOMA77 资产复用已由原生文件哈希确认；当前 ARDY 27 节点与 Kimodo 77 节点评价骨架仍分别处理。CHOIS 当前真实样例采用逐输入逆 FK 静止几何回推，作为估计来源单列。Andy/Han 的标记框架与 CoLLaboratE 共用胸段静止基底属于项目几何定义，解剖对应保留独立假设。详见[来源审计](/home/heinrich/MorionGen/develop_logs/notes/20261116_evalkit_k3_sources.md)。',
      '', 'MotionLCM、OmniControl 的位置输入支持几何方向及胸段代理；独立头朝向与完整颈部旋转继续不可可靠评价。其余六路保留原生旋转，并在明确的受控段约定下对照原始旋转。该分支直接走 K 旋转路径；HYOID 保持原状。身体轴向转动、完整解剖关节角和解剖段坐标对应仍按证据范围保留部分支持。',
      '', '| 格式 | 头 / 胸 / 颈旋转端到端 P95 / ° | 颈旋转有效帧 / 总帧 |',
      '| --- | ---: | ---: |']
    for m in METHODS:
        h=methods[m]['head_thorax'];n=h['neck']
        lines.append(f'| {m} | '+ ' / '.join(fmt(h[k]['p95_deg']) for k in ['head','thorax','neck'])+f' | {n["valid_frames"]} / {n["total_frames"]} |')
    lines+=['', '头部位置与颈头连线另走源姿态几何通路。下表比较退化读回 D 与原始 R；当前全身 OpenSim 拟合没有独立头部位置输出，故该表标记为读取通路验收。颈头连线提供空间方向，头部绕连线的转动仍由独立旋转观测决定。长度归一使用原窗口的骨盆中点—肩中点—颈节点—头节点链，并保留各自根运动。',
        '', '| 格式 | 颈头连线方向 D−R P95 / ° | 头位置 D−R P95：原几何 / 统一骨长且去根 / cm | 连线有效覆盖 |',
        '| --- | ---: | ---: | ---: |']
    for m in METHODS:
        h=methods[m]['source_head_geometry'];d=h['raw']['neck_head_direction']
        lines.append(f'| {m} | {fmt(d["p95_abs"])} | {fmt(h["raw"]["head_position"]["world"]*100)} / {fmt(h["normalized"]["head_position"]["root_relative"]*100)} | {pct(d["coverage"])} |')
    lines+=['', 'Kimodo 的颈头连线还包含目标骨架的中间节点假设：`neck→Neck2→head` 中的 Neck2 在受控原始 BVH 中没有对应观测，冻结退化器采用父旋转先验；直接子节点标定未跨过这一节点。该路线的头部独立旋转与颈头连线因此具有不同误差，长度归一也保留方向差异。[头部链条定义核查](/home/heinrich/MorionGen/config/evalkit_k3_20261116/head_chain_contract.json)记录节点、静止几何和适用边界。本轮保持冻结输入，将其作为降级链中的局限列出。']
    fast=figures['fast_head'];jump=figures['source_jump'];interval=jump['peak_interval_s']
    lines+=['',f'时序检查按原始窗口的头部角速度选择案例。持续高角速度示例 `{Path(fast["id"]).stem}` 的角速度 P95 为 {fast["original_head_p95_deg_s"]:.1f}°/s；六路旋转曲线和读回误差分别绘制。另一个极端峰值来自 `{Path(jump["id"]).stem}` 的窗口首个时间间隔：{interval[1]-interval[0]:.4f} 秒内头部世界旋转变化约 {jump["original_head_peak_deg_s"]*(interval[1]-interval[0]):.1f}°，根位置同时变化 {jump["root_displacement_in_peak_interval_m"]*100:.1f} cm。原始 BVH 前两帧也含有这次变化，属于源记录边界突变；历史窗口和评分掩码继续保留。',
      '', '![高角速度窗口的头颈旋转读回](/home/heinrich/MorionGen/develop_logs/notes/20261116_k3_figures/head_fast_turn.png)',
      '', '[源记录边界跳变的独立图](/home/heinrich/MorionGen/develop_logs/notes/20261116_k3_figures/head_source_jump.png)与图索引保存选取规则和所有候选角速度。这些检查说明重采样及边界数据的影响，段轴的解剖含义继续采用前述证据范围。']
    lines+=['','7 条丝织记录的原始根平移异常，进入共同 float32 格式后肢体尺度的位置差异会丢失。它们保留在全量 1,291 条 CoLLaboratE 的登记与失败列表中；本轮分层拟合使用其余可表达的记录。局部姿态可通过原生层级去根平移保留，世界轨迹与接触范围仍需源数据修复。',
      '', '<details><summary>展开逐部位工程精度与覆盖</summary>','',
      '下表使用全部共同可观测窗口，采用按记录、部位内指标、参与者逐级等权的统计。中位数与 P95 均是各指标统计量的汇总；5°／10°／20°为对应误差档内比例。有效／总数按“指标×帧”计数，同一帧可贡献多个腿部几何指标。骨盆线倾斜只描述双髋连线相对竖直的角度；轴向转动和完整解剖关节角另受观测约束。',
      '', '| 格式 | 部位 | 中位数 / ° | P95 / ° | ≤5° / ≤10° / ≤20° | 有效指标帧 / 总指标帧 |',
      '| --- | --- | ---: | ---: | ---: | ---: |']
    for m in METHODS:
        e=methods[m]
        for reg,label in REGIONS.items():
            v=e['variants'][e['selected']]['region_precision'][reg];n=v['metric_frames']
            tiers=' / '.join(pct(v['within_'+str(k)]) for k in [5,10,20])
            lines.append(f'| {m} | {label} | {fmt(v["median_abs"])} | {fmt(v["p95_abs"])} | {tiers} | {n["valid"]} / {n["total"]} |')
    lines+=['','</details>','']
    old=json.loads((ROOT/'config/evalkit_k_validation_20261116/development_summary.json').read_text())['datasets']
    lines+=['','Andy 120 条、Han 259 个分段（178 条原记录）的头胸可观测性已全量复查。独立标记参考及短缺测误差沿用已冻结的开发结果；本轮的全身方向优化不进入该参考构建。历史报告中 Andy/Han 的独立参考重建 P95 均值分别为 '+fmt(old['andy']['reference']['record_macro_p95_deg'])+'°／'+fmt(old['han']['reference']['record_macro_p95_deg'])+'°，对应旧参考有效覆盖分别为 '+pct(old['andy']['reference']['frame_coverage'])+'／'+pct(old['han']['reference']['frame_coverage'])+'。这部分明确标记为历史证据复用。',
      '', '## 可复跑资料','',
      '- [全量轻量检查](/home/heinrich/MorionGen/config/evalkit_k3_20261116/light_diagnostics.json)：Andy 8 人、Han 7 人的直接标记框架覆盖；1,284 条可表达 CoLLaboratE 均完成八路三姿态转换探针。直接标记框架统计与冗余点恢复后的历史参考覆盖分别标注。',
      '- [固定开发名单](/home/heinrich/MorionGen/config/evalkit_k3_20261116/selection.json)：记录、参与者组、窗口及选择依据。',
      f'- [四组实验配置](/home/heinrich/MorionGen/config/evalkit_k3_20261116/experiment_config.json)与[最终 K3 配置](/home/heinrich/MorionGen/config/evalkit_k_v3.json)：修正参考后的实验冻结检出为 `{frozen}`，K2 对照检出为 `780dda7`；原 `49ad7d0` 批次因参考读取问题停止并保留。',
      '- [逐格式汇总](/home/heinrich/MorionGen/config/evalkit_k3_20261116/results_summary.json)：逐侧误差、覆盖、留人折和退化记录名单；其中 `result_paths` 指向数据盘逐记录结果。',
      '- [投影与优化的空间变化](/home/heinrich/MorionGen/config/evalkit_k3_20261116/spatial_effects_summary.json)：位移、速度向量变化和近地／低于参考面判定切换；受控试次按人汇总，真实输出单列。',
      '- 原始／退化／投影／拟合叠加图：`develop_logs/notes/20261116_k3_figures`，同时保留典型、高基准误差、退化及任务覆盖案例。',
      f'- 大产物目录：`{a.out}`。CPU 工作进程最多 3 个、数值库单线程；磁盘余量底线 80 GiB。',
      '', '单条入口：`python python/scripts/evalkit_k.py --source INPUT --method METHOD --out OUTPUT --fit --config config/evalkit_k_v3.json`。统一 evalkit Python 入口传 `k_config`；批处理 manifest 每行可指定 `"k_config":"config/evalkit_k_v3.json"`。既有 K2 默认入口和历史结果保留。',
      '', '复跑顺序：`evalkit_k3_audit.py light/prepare` → 冻结检出的 `evalkit_k3_experiment.py` → `evalkit_k3_rescore.py` → `evalkit_k3_report.py` → `evalkit_k3_select.py` → `evalkit_k3_actual.py` / `evalkit_k3_regression.py`。`evalkit_k3_finish.py` 串行执行拟合完成后的重评分、选型、归一检查、空间变化、回归与绘图。复跑已有实验应使用其冻结检出和独立产物目录，避免把当前实现写入旧缓存。','']
    verification=cfg/'verification.json'
    if verification.exists():
        check=json.loads(verification.read_text());resource=check['resource'];reuse=check['actual_fidelity_reuse']
        visual='39 张最终图已人工检查布局、曲线和失败姿态' if check['visual_review']['status']=='reviewed' else '最终图的人工核查正在登记'
        lines+=['## 验证与交付记录','',
          f'{check["tests"]["passed"]} 项自动检查通过；八路原拟合在修正后的同一输入上与冻结 K2 坐标逐项一致。K3 的八路工效与外力接口回归通过，下肢肌肉入口完成 11 帧有限值检查。这些回归检验接口与运行一致性，肌肉及外力的实验精度由后续独立观测评价。{visual}。',
          '', f'真实样例复用结果与当前运行入口再次配对核查：最大关节位置差 {reuse["maximum_fitted_position_difference_m"]*1000:.4f} 毫米，几何角保真 P95 的最大变化 {reuse["maximum_fidelity_p95_difference_deg"]:.3g}°，均小于本报告显示精度。独立坐标仍按各自可观测性解释，完整差异见[真实样例复用核查](/home/heinrich/MorionGen/config/evalkit_k3_20261116/actual_reuse_parity.json)。',
          '', f'冻结资料与留出使用记录共 {check["protected_files_unchanged"]} 个受保护文件逐项哈希一致；本轮新增留出运行 {check["new_holdout_runs"]} 次，plans 与 HYOID 保持原值。交付检查时，修正后实验目录占 {resource["corrected_output_gib"]:.2f} GiB，数据盘剩余 {resource["free_gib"]:.1f} GiB。完整验收信息见[verification.json](/home/heinrich/MorionGen/config/evalkit_k3_20261116/verification.json)。','']
    target=ROOT/'develop_logs/notes/20261116_evalkit_k3_alignment.md';target.write_text('\n'.join(lines));print(target)

if __name__=='__main__':main()
