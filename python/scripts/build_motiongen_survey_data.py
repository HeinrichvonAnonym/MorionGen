"""Assemble reviewed evidence, source diagrams and lossless report text archive."""
import argparse
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
NOTES=ROOT/'develop_logs/notes'
RUNS=ROOT/'develop_logs/test_runs'
APP=NOTES/'motiongen_survey'
BASE=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c95839/MorionGen/baselines')


class VisibleText(HTMLParser):
    def __init__(self):
        super().__init__(); self.skip=0; self.parts=[]
    def handle_starttag(self,tag,attrs):
        if tag in ['script','style']: self.skip+=1
        if tag in ['p','div','section','h1','h2','h3','tr','li','details']: self.parts.append('\n')
    def handle_endtag(self,tag):
        if tag in ['script','style']: self.skip-=1
        if tag in ['td','th']: self.parts.append(' | ')
    def handle_data(self,data):
        if not self.skip: self.parts.append(data)


def query(rows,files,definition,ids):
    return {'rows':rows,'source':{'label':'本地实验与版本化源码','files':files,
        'metricDefinitions':[{'label':'范围与口径','definition':definition,'componentIds':ids}],
        'evidenceFlow':[{'title':'可复算来源','detail':'读取所列本地文件；生成脚本 python/scripts/build_motiongen_survey_data.py。实验统计来自 summarize_feedback_replication.py；不重跑推理、不补造缺失值。'}]}}


def build(complete=False):
    methods=json.loads((ROOT/'config/motiongen_survey_methods.json').read_text())
    for row in methods:
        if row.get('repo'):
            row['url']='https://github.com/'+row['repo']
            row['links']=[f"{row['url']}/blob/{row['commit']}/{p}" for p in row['files']]
            dirname={'CHOIS':'chois_release','PRIMAL':'PRIMAL'}.get(row['name'],row['name'])
            if row['name'] in ['ARDY','Kimodo']: dirname=row['name'].lower()
            for file in row['files']:
                assert (BASE/dirname/file).is_file(), (dirname,file)
    archive=[]
    for path in sorted(NOTES.iterdir()):
        if not path.is_file() or path.suffix not in ['.md','.html'] or path.name.startswith('motiongen_survey'): continue
        raw=path.read_bytes(); text=raw.decode()
        if path.suffix=='.html':
            parser=VisibleText(); parser.feed(text)
            text='\n'.join(line.strip() for line in ''.join(parser.parts).splitlines() if line.strip())
        archive.append({'file':str(path.relative_to(ROOT)),'name':path.name,'sha256':hashlib.sha256(raw).hexdigest(),
                        'bytes':len(raw),'text':text,'format':path.suffix,
                        'archive_scope':'Markdown 全文；HTML 静态可见正文。动态数值另由结构化实验查询整合，原始 HTML 保留。'})
    assert len(archive)==6, [r['name'] for r in archive]
    baseline=[
        {'method':'MotionLCM','experiments':'T2M 3 seed；reach 3；20 秒 rolling 1；稳态计时 20 次','A':'reach 0/3 ≤5 cm；1.086–1.204 m','B':'50 次重规划；拼接前49 mm / 后3.7 mm','E':'仅运动学代理；拼接改善不等于生成一致','H':'稳态 RTF .0073；rolling .124'},
        {'method':'OmniControl','experiments':'T2M 1；reach 3 seed','A':'reach 1.7–3.1 mm','B':'没有在线目标切换实验','E':'jerk 103.7–212.9 m/s³','H':'reach RTF 8.95–9.34'},
        {'method':'CHOIS','experiments':'120 帧 carry 单窗口及重复测量','A':'接触 F1 .662；手物距离 .198 m','B':'不是执行中物理接触闭环测试','E':'jerk 264.9 m/s³','H':'RTF 1.06'},
        {'method':'ARDY','experiments':'3 seed × 20 秒 × 50 次重规划','A':'历史/未来约束接口；不冒充 reach 命中率','B':'历史误差 .456–.598 mm；边界18.7–20.2 mm','E':'KIT jerk 百分位约36–69，仅作尺度参考','H':'RTF .280–.326'},
        {'method':'Kimodo','experiments':'初始文本/reach/rotation；BONES 1,856 raw + 416 correction 对；KIT830参考','A':'raw 5cm：末端95.44%，全身96.72%，root67.77%','B':'离线约束，不证明实时反馈','E':'raw−GT jerk −19.05；correction 滑步代价','H':'raw RTF P50/P95 1.165/1.435；初次完整5.10h'}]
    old=json.loads((RUNS/'20261024_online_feedback_summary.json').read_text())
    native=json.loads((RUNS/'20261024_primal_feedback_summary.json').read_text())
    v4path=RUNS/'20261024_feedback_replication_v4'/('summary.json' if complete else 'partial_summary.json')
    v4=json.loads(v4path.read_text()) if v4path.exists() else {'rows':[],'groups':[],'effects':[],'scenario_effects':[],'sensitivity':[],'clips':[],'prefix_checks':[],'clock_checks':[],'completion':{},'n_runs':0}
    ids=['replication-table','replication-chart','replication-summary','replication-interpretation','scenario-chart','sensitivity-table','motion-player','confidence-methods']
    queries={
        'methods':query(methods,['config/motiongen_survey_methods.json'],'层级/接口图来自 pinned 源码或列出的官方项目页；代码机制与本地性能严格区分。',['method-diagrams','method-matrix']),
        'legacy':query(baseline,[r['file'] for r in archive[:3]],'不同任务/数据集/人体配置，不做合并排名；RTF=计算时间/动作时长。',['baseline-table','legacy-findings']),
        'archive':query(archive,[r['file'] for r in archive],'保留六份历史报告的来源、哈希和正文；最新综合解释覆盖早期小样本外推，不覆盖原文件。',['report-archive','engineering']),
        'replication':query(v4['rows'],[str(v4path.relative_to(ROOT))],'每行一条15秒episode；5秒切目标，后10秒平均XY根节点距离为主指标。12场景×3seed×2clock×2goal开关×2method=288；两个clock不当独立样本。',ids),
        'effects':query(v4['effects'],[str(v4path.relative_to(ROOT))],'反馈开−关配对差，先每场景平均3seed，再12场景bootstrap20,000次；另以初态×方向×seed全因子水平交叉重采样，保留共享依赖。95%百分位区间仅描述固定设计；E指标为完整15秒，H为各episode请求P95的均值，不是合并全部请求的P95。',['replication-summary','replication-table']),
        'scenario_effects':query(v4['scenario_effects'],[str(v4path.relative_to(ROOT))],'每个场景3个seed的配对差均值，未合并clock。',['scenario-chart']),
        'sensitivity':query(v4['sensitivity'],[str(v4path.relative_to(ROOT))],'同一批轨迹，半径.2/.3/.5m、保持0/.5/1s，观察5/10s；没有额外推理，不重复累计样本。',['sensitivity-table','replication-interpretation']),
        'clips':query(v4['clips'],[str(v4path.relative_to(ROOT))],'固定seed10实时轨迹，30fps降采样到10fps仅用于播放，所有统计仍使用450帧原数据。',['motion-player']),
        'native':query([{'condition':k,**v} for k,v in native['means'].items()],['develop_logs/test_runs/20261024_primal_feedback_summary.json'],'PRIMAL 原生2×2组件消融，每格5seed；同一adapted初态。',['native-ablation']),
        'previous_online':query(old['groups'],['develop_logs/test_runs/20261024_online_feedback_summary.json'],'v2/v3共120episode；v2目标前轨迹分叉是调度诊断，不作反馈单因素因果效应。',['previous-online'])}
    snapshot={'surface':'report','title':'MotionGen Survey','status':'本地实测与源码调查','buildStatus':'complete' if complete else 'creating',
              'generatedAt':datetime.now(timezone.utc).isoformat(),'filters':[], 'queries':queries,
              'report':{'asOf':'2026-10-24','subtitle':'Human motion · A/B/E/H · 源码结构、配对实验与证据整合'},
              'replicationAudit':{k:v4[k] for k in ['n_runs','prefix_checks','clock_checks','completion']}}
    destination=APP/'src/data.json' if APP.exists() else RUNS/'motiongen_survey_reviewed.json'
    if destination.exists():
        previous=json.loads(destination.read_text())
        if 'id' in previous: snapshot['id']=previous['id']
    destination.write_text(json.dumps(snapshot,ensure_ascii=False,separators=(',',':')))
    print(json.dumps({'snapshot':str(destination),'reports':len(archive),'methods':len(methods),'episodes':v4['n_runs']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--complete',action='store_true'); build(p.parse_args().complete)
