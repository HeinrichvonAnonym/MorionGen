#!/usr/bin/env python3
"""Render the frozen K development/holdout evidence as a project stage report."""
import argparse,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def f(x,d=2):return '—' if x is None else f'{x:.{d}f}'
def pct(x):return '—' if x is None else f'{100*x:.1f}%'

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',*['| '+' | '.join(map(str,r))+' |' for r in rows]])


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--evidence',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--figures',type=Path,required=True)
    a=ap.parse_args();folder=a.evidence
    freeze=json.loads((folder/'freeze.json').read_text());dev=json.loads((folder/'development_summary.json').read_text());final=json.loads((folder/'final_summary.json').read_text());reg=json.loads((folder/'regression_summary.json').read_text())
    datasets=[]
    for phase,summary in [('开发',dev),('最终留出',final)]:
        for name,r in summary['datasets'].items():
            datasets.append([phase,{'andy':'Andy','han':'Han','collaborate':'CoLLaboratE' if phase=='开发' else '电视场景'}[name],r['trials'],r['statuses'].get('computed',0),r['recordings'],r['groups'],r['profile_window_statuses'].get('computed',0)])
    refs=[]
    for phase,summary in [('开发',dev),('留出',final)]:
        for name,r in summary['datasets'].items():
            v=r['reference'];refs.append([phase+'/'+name,'头胸模板一致性' if name!='collaborate' else '胸段位置/旋转一致性',
                f(v['record_macro_median_deg']),f(v['record_macro_p95_deg']),pct(v['record_macro_within_5_deg']),pct(v['record_macro_within_10_deg']),pct(v['record_macro_within_20_deg']),pct(v['frame_coverage'])])
    profiles=[]
    for m in ('motionlcm','omnicontrol','ardy','chois','closd','dart','kimodo','primal'):
        row=[m]
        for d in ('andy','collaborate'):
            v=final['datasets'][d]['methods'][m]
            row.extend([f(v['end_to_end']['record_macro_p95_deg'])+' / '+pct(v['end_to_end']['frame_coverage']),f(v['thorax_end_to_end']['record_macro_p95_deg'])])
        profiles.append(row)
    fits=[]
    for r in reg:
        e=r['geometric_fit_errors'];fits.append([r['method'],'合格' if r['calibration']['qualified'] else '诊断拟合',pct(1-r['ik']['unmeasurable_share']),
                                               f(e['trunk_incline']['p95_deg']),f(max(e['left_elbow_flexion']['p95_deg'],e['right_elbow_flexion']['p95_deg'])),r['neck']['status']])
    gaps=[]
    for phase,summary in [('开发',dev),('留出',final)]:
        for d,r in summary['registered_gap_experiment'].items():
            cases=r['regions']['head_and_thorax']
            gaps.append([phase+'/'+d,*[f(cases[str(s)]['record_macro_p95_deg']) for s in (.1,.5,1.,2.)],
                         cases['0.1']['recordings'],pct(cases['0.1']['record_macro_within_10_deg'])])
    test=json.loads((folder/'tests.json').read_text())
    totaltests=sum(r.get('tests',0) for r in test['commands'])
    text=f'''# Evalkit K：对齐、可测性与冻结验证

本轮把 K 从分散的读取与拟合逻辑整理为独立模块，完成八种格式接入、按人开发验证，以及冻结后对 Andy 75 条试次和电视场景 532 条唯一轨迹的验证。实现与配置见 [冻结记录]({folder.resolve()}/freeze.json)，逐分支结果见 [最终汇总]({folder.resolve()}/final_summary.json)。两套最终留出已使用；后续若据其结果调参，应登记为新版本。

核心判断是：**已有原生旋转的六种格式可以检验信息保留与坐标转换；位置型两种格式支持胸段和骨方向指标，完整颈部姿态保留“不可可靠评价”。全身拟合的适用性按部位和标定条件另报。** 这使 K 能给后续外力、工效和肌肉模块提供明确的有效范围。

## 验证的是哪一层

“段姿态”指头部、胸廓等身体段的三维朝向；旋转测地误差是两个朝向之间的最小三维转角。逆运动学（IK）则用关节位置拟合肌骨模型的关节角，两者分别校验。

- Andy/Han 从原始光学标记构建四点段坐标，再与经独立标定窗建立的冗余模板重建比较。两种方法共享测量标记，结果描述**重建一致性**；绝对解剖角度还依赖标记放置及中立标定。
- CoLLaboratE 从原生 BVH 层级、旋转通道和非根平移恢复全身运动，用于表示一致性与场景迁移检查。源骨架和固定目标模板的体型差异单列为重定向损失。
- 八种格式分别记录参考到目标骨架、编码读回、最终 K 相对参考三个阶段。Andy/Han 的此项实验只把实测头胸放在模板身体上，身体其余部位保持先验属性；CoLLaboratE 提供完整身体表示对照。
- 5°/10°/20° 是工程精度档。下表的“中位数均值”和“P95 均值”均先按原记录计算再汇总；Han 的段先归并原记录。每个档位比例也采用原记录等权，另存人数、帧覆盖和有符号偏差。

{table(['阶段','来源','登记段/试次','成功计算','原记录','人员组','具有格式评分窗'],datasets)}

每条记录的参考检查使用全程。八格式导出取首个至少 1 秒的完整连续窗，最多 10 秒；PRIMAL 前两帧仅作历史。短记录、缺完整窗以及数值精度不足均保留在总记录分母。电视场景属于未见任务域；跨子集身份尚未统一，人员组数量按包内编号解释。

## K 的具体调整

`utils/motiogen/evalkit/k/` 集中管理运动、几何、标定、拟合、参考和适配。`fit.py` 保留兼容入口；工效读取、原生颈部旁路和运动学诊断使用统一 K 数据。配置在 [evalkit_k_v1.json]({ROOT}/config/evalkit_k_v1.json)，模型中立几何与来源哈希在 `config/evalkit_k_calibration/`。

内部统一为米、右手 Y-up；头胸段采用解剖右侧、上方、后方三轴。中立姿态、前屈方向、PRIMAL 原生 Y-up、CLoSD 刚体四元数、DART 局部矩阵与旧格式缺字段都有独立处理。位置型输入中的额外旋转字段会被忽略；CHOIS 的身体腕节点与额外手点、CLoSD 的刚体名字分别映射掩码。

每个指标携带有效掩码、来源和不可评价原因。`model_convention` 表示模型中立几何的一致性检查通过，外部精度仍由验证范围限定。`supported` 用于明确给定段标定的对照；它描述计算条件。`k.npz` 保存位置、段旋转和逐指标掩码；拟合坐标、原始时钟与拟合时钟对应关系分别保存。短缺测预测写入额外数组，观测掩码保持原状。

全身拟合把“片段最直立帧”改为有条件的标定：躯干倾斜 ≤20°、大腿倾斜 ≤25°，合格资料至少 0.5 秒。这些阈值是本轮工程约定。资料不足时保留诊断拟合，标定有效性为 false；拟合不可用也会保留 K 中可用的原生旋转。缓存键覆盖输入内容、配置、模型和实现版本。

开发中发现 {len(dev['datasets']['collaborate']['position_precision_invalid_trials'])} 条 SlikWeaving 记录的位置数值精度不足。其中 `SWPCS01G01R34`、`SWPCS01G01R98` 的根平移达到约 10²⁶ cm，浮点 FK 丢失了局部高度差。新增数值精度检查：世界坐标的浮点间隔超过 0.1 mm 时，位置指标标记为未知，原生旋转独立保留。全部记录编号见开发汇总；原始资料和此前试跑均保留。

## 开发折、参考覆盖与留出

Andy 8 折、LAB 10 折、Han 7 折均按已登记人员分组执行。模型中立几何来自已有资产；本轮选择的是参考残差门与格式规则。光学残差门在训练折中按“人员等权的 10° 内比例至少 95%，再优先覆盖”选择：最终采用 {json.dumps(freeze['selected_reference_gates'],ensure_ascii=False)}（单位米）。LAB 检查固定层级和 collar 映射的跨人稳定性。每折计数与候选阈值见 [开发汇总]({folder.resolve()}/development_summary.json)。

{table(['阶段/来源','比较量','中位数均值°','P95均值°','≤5°','≤10°','≤20°','帧覆盖'],refs)}

Andy 的比较覆盖要求头胸四点参考同时可计算，并排除模板标定帧。它与此前“至少三点可见”的候选覆盖使用不同条件；自然缺测处的模板重建可以增加可用输出，但在缺少独立参考时只报告恢复覆盖。

以下是冻结后的**格式对照窗**误差。六种旋转格式的低误差支持坐标与序列化路径；其实际生成质量和外部解剖精度分别检验。D1/D2 的头部终端旋转可以在关节中心不变时改变，因此保留颈头连线几何量，完整颈部旋转留空。

{table(['方法','Andy颈P95均值° / 窗内有效帧','Andy胸P95均值°','电视颈P95均值° / 窗内有效帧','电视胸P95均值°'],profiles)}

CLoSD 的原生四元数路径已接入；受控导出的关节 DOF 参数仍标为 unavailable，接触力也留待独立动力学实验。指、眼、下颌与额外手点保留模板掩码，不参与独立精度结论。

## 缺测恢复与全身拟合

遮蔽按登记 seed，在头、胸、头胸同时缺测三种条件下执行。预测接口只接收遮蔽后的两个段旋转与时间，原始标记、位置、局部旋转和派生特征留在评分侧。下表列头胸同时缺测后颈部旋转的 P95 均值；0.5/1/2 秒为压力测试，运行时恢复上限保持 0.1 秒。每项只在具有完整端点参照的窗口中评分，原记录数明确列出。

{table(['阶段/来源','0.1s °','0.5s °','1s °','2s °','0.1s可比原记录','0.1s≤10°'],gaps)}

八份真实基线输出完成全身拟合、外力和工效接口回归；另有一个 0.2 秒窗口、11 帧肌肉求解回归。下表比较源骨架与拟合模型关节中心的几何量，反映模型和骨架定义差异及拟合损失；肌肉求解的数值成功仅作为接口证据。

{table(['方法','全身标定','IK有效评分帧','躯干倾斜P95°','两侧肘角最大P95°','颈部状态'],fits)}

DART 样例保留诊断拟合，同时原生头胸分支独立工作；OmniControl 样例存在较明显的手臂拟合损失。全身金标批量 IK 的独立关节角精度尚需更完整的人体标定资料，本轮以完整 BVH 的几何对照、八份真实输出及开发叠加图界定当前范围。工效分档作为解读附录，未用于选择 K 的通过门。

![原始参考、目标骨架与拟合]({a.figures.resolve()}/gold_overlays.png)

![真实输出的全身拟合叠加]({a.figures.resolve()}/body_overlays.png)

![光学参考与模板重建叠加]({a.figures.resolve()}/optical_overlays.png)

图中目标骨架采用固定体型，关节位置差异与角度误差分开解释；异常世界坐标记录显示资料限制。典型及高误差 Andy 记录按开发记录 P95 的中位数和最大值选取。

## 复现与后续使用

已完成 {totaltests} 项自动检查（K、工效规则与 BVH 通道/FK），八份真实输出回归、25 折开发汇总，以及冻结后的 607 条最终记录验证。命令和测试结果见 [tests.json]({folder.resolve()}/tests.json)。完整逐记录结果与中间模型放在数据盘；仓库提交配置、汇总、源代码和本报告。

```bash
/home/heinrich/.local/share/opensim-api/bin/python python/scripts/evalkit_k.py \\
  --source /absolute/path/motion.npz --method dart --out /data/path/k_result --fit
# 需要离线短缺测预测时，加 --predict-short-gaps。
```

验证入口为 `evalkit_k_validate.py`、`evalkit_k_gap_validate.py`，最终运行必须指定本轮 `freeze.json`。`evalkit_k_summarize.py` 重建统计，`evalkit_k_regression.py` 重建接口回归。冻结文件记录了源码、上游 HumanML 编解码器、模型、数据清单和环境指纹。

下一阶段可在本接口上补充有中立姿态与体型标定的全身参考，处理手臂拟合高误差及额外身体接触。位置型头部预测若另行开发，继续作为预测分支，按独立验证确定精度范围；本轮已使用的 Andy/电视留出结果作为历史验证保留。
'''
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(text)
    print(a.out)


if __name__=='__main__':main()
