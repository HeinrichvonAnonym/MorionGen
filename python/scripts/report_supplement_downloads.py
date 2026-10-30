"""Account for new assets and reused dependencies, including unused downloads."""
import hashlib,json,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import ROOT,load_config,data_root
from motiogen.runtime import checksum,json_write
data=data_root(load_config());assets=data/'outputs/assets';rows=[]
for folder in ['motiongpt-official','motiongpt-flan-t5-base','uhc-official']:
    for p in sorted((assets/folder).iterdir()):
        if not p.is_file() or p.name.endswith('.json') or p.name=='README.md':continue
        if folder=='motiongpt-official':
            source='https://drive.google.com/uc?id=1AYsmEG8I3fAAoraT4vau0GnesWBWyeT8' if p.name=='t2m.zip' else 'https://huggingface.co/OpenMotionLab/MotionGPT-base'
            purpose='官方评价器与 VQ 归一化资产；文件扩展名为 zip，实际格式是 tar.gz' if p.name=='t2m.zip' else '原生 T5＋运动 VQ 生成权重'
        elif folder=='motiongpt-flan-t5-base':source='https://huggingface.co/google/flan-t5-base';purpose='官方语言模型初始化及 tokenizer；随后严格加载 MotionGPT checkpoint'
        else:
            source='https://drive.google.com/uc?id='+('1CA3lKhIEOwzhX7IMdu_B_QnjrXfPXYnM' if p.name=='uhc_implicit.p' else '12rxODUmyOyOg11rQUY7GTmT-NDH7R2uN')
            purpose='UHC implicit-RFC 预训练策略' if p.name=='uhc_implicit.p' else '下载检查后确定是单条姿态资产，未用于正式跟踪；实际参考使用仓库自带测试集'
        rows.append({'path':str(p),'bytes':p.stat().st_size,'sha256':checksum(p),'source':source,'purpose':purpose})
engine=data/'staged_dependencies/mujoco210-linux-x86_64.tar.gz'
rows.append({'path':str(engine),'bytes':engine.stat().st_size,'sha256':checksum(engine),
    'source':'https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz','purpose':'UHC 原生 mujoco-py 运行引擎，独立环境使用'})
for p in (data/'staged_dependencies/uhc-osmesa').glob('*.deb'):
    rows.append({'path':str(p),'bytes':p.stat().st_size,'sha256':checksum(p),'source':'Ubuntu jammy-updates 官方 APT 源；包版本见文件名','purpose':'仅解压到数据盘提供 OSMesa 编译头与库，未修改系统安装'})
for row in rows:
    path=Path(row['path']);timer=path.with_name(path.name+'.download.json')
    receipt=json.loads(timer.read_text()) if timer.exists() else {}
    row['download_elapsed_seconds']=receipt.get('elapsed_seconds')
    row['download_timer_receipt']=str(timer) if timer.exists() else None
out=ROOT/'develop_logs/test_runs/20261028_supplement/download_inventory.json'
json_write(out,{'files':rows,'free_bytes':{'system':shutil.disk_usage(ROOT).free,'data':shutil.disk_usage(data).free},
    'reuse':['CLoSD 官方三性别 SMPL','MotionLCM 原生文本生成权重和 GloVe/T2M evaluator','已有 ARDY/DART/CLoSD 权重','UHC 仓库自带 10 条参考动作'],
    'reference_recovery':'2026-10-29 在既有 CLoSD 官方 t2m_test.npy 找到 4648 条处理后测试记录；本轮八条源文本均有匹配动作。该文件为旧资产复用，不是本轮新增下载。详见 reference_recovery/audit.json。完整原始 AMASS 不是本轮共同评价的前置条件。',
    'authorization_blocker':None})
text=['# MotionGen 本轮下载清单（2026-10-28）','',
      '本轮确实新增下载了 MotionGPT、UHC 预训练权重及运行／评价依赖，没有下载训练全量数据，也没有启动重训。大资产均在数据盘。',
      '',f'当前系统盘剩余 {shutil.disk_usage(ROOT).free/2**30:.1f} GiB，数据盘剩余 {shutil.disk_usage(data).free/2**40:.2f} TiB；本轮资产不占用系统盘大文件空间。','',
      '| 资产 | 字节数 | 下载耗时 | 用途 |','|---|---:|---:|---|']
for r in rows:
    duration=f'{r["download_elapsed_seconds"]:.1f} 秒' if r['download_elapsed_seconds'] is not None else '没有逐文件计时记录'
    text.append(f'| [{Path(r["path"]).name}]({r["path"]}) | {r["bytes"]:,} | {duration} | {r["purpose"]} |')
text+=['','耗时来自下载器保存的单次传输记录，不含环境安装和后续解压；缺少计时的资产不以文件修改时间补写，不把并发传输耗时相加当作自然时间。']
text+=['','## 来源与校验','']
for r in rows:text += [f'- `{Path(r["path"]).name}`：{r["source"]}',f'  - SHA-256：`{r["sha256"]}`']
text+=['','## 复用与缺口','',
       '- SMPL 人体模型、MotionLCM 文本权重、已有方法权重和 GloVe/T2M 评价器均复用；没有重复下载整套人体模型。',
       '- UHC 测试集随官方仓库克隆取得，正式跟踪使用其中八条独立源动作。单独下载的 `sample.pkl` 没有充当八条参考数据。',
       '- MotionGPT、UHC、SMPLSim、PHC_MJX 源码的提交号记录在本轮 `source_audit.json`。PHC_MJX 未取得可运行策略；PhysDiff 官方入口未给出可执行下载。',
       '- 2026-10-29 更正：4T 盘既有 CLoSD 官方 `t2m_test.npy` 包含 4,648 条处理后 HumanML3D 测试记录；本轮八条源文本全部找到了匹配动作，五条与独立原文件数值完全相同。此前“只有六条参考动作”遗漏了官方缓存。该文件为旧资产复用，不是本轮新增下载，也不代表完整原始 AMASS。',
       '- 已下载资产没有出现新的登录要求。Python 包安装及环境克隆属于准备成本，不计作推理 GPU·h。','',
       f'机器可读明细：[download_inventory.json]({out})。']
target=ROOT/'develop_logs/notes/20261028_supplement_downloads.md';target.write_text('\n'.join(text)+'\n');print(target)
