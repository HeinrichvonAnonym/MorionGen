"""Shared pinned T2M evaluator; pilot semantic checks, no unsupported FID."""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import ROOT,data_root,load_config
from motiogen.runtime import checksum,json_write
from motiogen.motion import load_motion
from motiogen.metrics import evaluate
from motiogen.posture_geometry import observable_angles
parser=argparse.ArgumentParser()
parser.add_argument('--reference-audit',type=Path)
parser.add_argument('--output',type=Path)
args=parser.parse_args()
data=data_root(load_config());assets=data/'outputs/assets';sys.path.insert(0,str(data/'baselines/MotionGPT'))
import torch
from mGPT.archs.tm2t_evaluator import TextEncoderBiGRUCo,MovementConvEncoder,MotionEncoderBiGRUCo
from mGPT.data.humanml.utils.word_vectorizer import WordVectorizer
torch.set_num_threads(4)
checkpoint=assets/'motionlcm/t2m/t2m/text_mot_match/model/finest.tar'
cp=torch.load(checkpoint,map_location='cpu')
textnet=TextEncoderBiGRUCo(300,15,512,512);move=MovementConvEncoder(259,512,512);motionnet=MotionEncoderBiGRUCo(512,1024,512)
for model,key in [(textnet,'text_encoder'),(move,'movement_encoder'),(motionnet,'motion_encoder')]:model.load_state_dict(cp[key],strict=True);model.eval()
meta=assets/'motiongpt-official/t2m/t2m/Comp_v6_KLD01/meta';mean=np.load(meta/'mean.npy');std=np.load(meta/'std.npy')
vectorizer=WordVectorizer(str(assets/'motionlcm/glove'),'our_vab')
rows=[];embeddings={};texts={}
sources={'MotionGPT':'motiongpt_queue','MotionLCM':'motionlcm_text_queue'}
for method,queue in sources.items():
    state=json.loads((ROOT/f'develop_logs/test_runs/20261028_supplement/{queue}/state.json').read_text())
    run=Path(next(iter(state['jobs'].values()))['run_dir'])
    for row in json.loads((run/'generation.json').read_text())['rows']:
        record=dict(row,method=method)
        if row['status']!='success':rows.append(record);continue
        folder=Path(row['output']);feature=np.load(folder/'native.npz')['features']
        tokens=row['tokens'].split()[:20];tokens=['sos/OTHER']+tokens+['eos/OTHER'];length=len(tokens);tokens+=['unk/OTHER']*(22-length)
        words,pos=zip(*(vectorizer[token] for token in tokens))
        with torch.no_grad():
            text=textnet(torch.tensor(np.asarray(words)[None],dtype=torch.float32),torch.tensor(np.asarray(pos)[None],dtype=torch.float32),torch.tensor([length]))
            normalized=torch.tensor(((feature-mean)/std)[None],dtype=torch.float32)
            em=motionnet(move(normalized[...,:-4]),torch.tensor([len(feature)//4]))
        key=f'{method}:{row["source_id"]}:{row["seed"]}';embeddings[key]=em[0].numpy();texts[row['source_id']]=text[0].numpy()
        record['matching_distance']=float(torch.linalg.vector_norm(text-em))
        motion=load_motion(folder/'motion.npz');record['kinematics']=evaluate(motion,{'name':'native_text_only'})['metrics']
        angles=observable_angles(motion['joints'],motion['joint_names'].tolist());record['angle_ranges_deg']={k:[float(v.min()),float(v.max())] for k,v in angles.items()}
        rows.append(record)
ids=sorted(texts);all_text=np.stack([texts[k] for k in ids])
for row in rows:
    if row['status']!='success':continue
    distances=np.linalg.norm(all_text-embeddings[f'{row["method"]}:{row["source_id"]}:{row["seed"]}'],axis=1)
    row['retrieval_rank_among_eight_texts']=int(np.flatnonzero(np.argsort(distances)==ids.index(row['source_id']))[0])+1
references=[]
distribution=[]
if args.reference_audit:
    audit=json.loads(args.reference_audit.read_text())
    ref_embeddings={}
    for row in audit['selected_references']:
        if not row['reference_path'] or not row['exact_caption_match'] or not row['tokens_match']:
            raise ValueError('Reference identity mismatch')
        path=Path(row['reference_path'])
        if checksum(path)!=row['reference_sha256']:raise ValueError('Reference checksum changed')
        feature=np.load(path)
        with torch.no_grad():
            normalized=torch.tensor(((feature-mean)/std)[None],dtype=torch.float32)
            embedding=motionnet(move(normalized[...,:-4]),torch.tensor([len(feature)//4]))[0].numpy()
        ref_embeddings[row['key']]=embedding
        references.append({'source_id':row['key'],'frames':len(feature),'path':str(path),
            'sha256':row['reference_sha256'],'matching_distance':float(np.linalg.norm(texts[row['key']]-embedding))})
    for row in rows:
        if row['status']!='success':continue
        row['reference_embedding_distance']=float(np.linalg.norm(embeddings[f'{row["method"]}:{row["source_id"]}:{row["seed"]}']-ref_embeddings[row['source_id']]))
        row['reference_frames']=next(r['frames'] for r in references if r['source_id']==row['source_id'])
        recovered=next(r for r in references if r['source_id']==row['source_id'])
        row['generation_request_reference_path']=row.get('reference_path')
        row['reference_path']=recovered['path']
        row['reference_sha256']=recovered['sha256']
        row['reference_recovered_after_generation']=True
    # Low-rank covariance identity avoids an unstable 512x512 singular sqrtm.
    # tr(sqrt(Cx Cy)) equals the nuclear norm of centered sample cross-product.
    reference=np.stack([ref_embeddings[k] for k in ids]).astype(np.float64)
    for method in sources:
        for seed in sorted({r['seed'] for r in rows if r['method']==method}):
            generated=np.stack([embeddings[f'{method}:{k}:{seed}'] for k in ids]).astype(np.float64)
            xc=(generated-generated.mean(0))/np.sqrt(len(generated)-1)
            yc=(reference-reference.mean(0))/np.sqrt(len(reference)-1)
            value=np.sum((generated.mean(0)-reference.mean(0))**2)+np.sum(xc**2)+np.sum(yc**2)-2*np.linalg.svd(xc@yc.T,compute_uv=False).sum()
            distribution.append({'method':method,'seed':seed,'source_count':len(ids),
                'fid_pipeline_diagnostic':float(max(0,value)),
                'scope':'Eight source references only; singular sample covariance, throughput/evaluator check, not a population quality estimate.'})
out=args.output or ROOT/'develop_logs/test_runs/20261028_supplement/generation_evaluation.json'
json_write(out,{'rows':rows,'references':references,'distribution_diagnostics':distribution,'evaluator_sha256':checksum(checkpoint),'normalization':{str(p):checksum(p) for p in [meta/'mean.npy',meta/'std.npy']},
    'scope':'Shared evaluator semantic distance and eight-text retrieval diagnostics, not official 32-way R-precision. Eight source texts x two seeds; cluster by source. Reference and generated lengths retained without stretching or truncation. No pure backbone causal claim.'})
print(json.dumps({'rows':len(rows),'output':str(out)}))
