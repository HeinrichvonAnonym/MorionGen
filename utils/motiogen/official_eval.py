"""Use pinned upstream metrics on provenance-checked official embeddings, never on three demos."""
import importlib.util
import json
import zipfile
from pathlib import Path

import numpy as np

from .adapters import MissingResources,ResourceGate,Unsupported
from .config import ROOT,read_json
from .runtime import budgets,checksum,json_write


def preflight(cfg,name,protocol_file):
    spec=read_json(ROOT/'config/evaluation.json')[name]
    if not spec['metrics_source']:
        raise Unsupported(spec['reason'])
    file=Path(protocol_file)
    if not file.is_file():
        raise MissingResources('Official embedding protocol manifest is absent')
    protocol=read_json(file)
    required=('evaluator_id','evaluator_checkpoint','evaluator_sha256','normalization_file','normalization_sha256',
              'embedding_archive','embedding_sha256','reference_dataset','reference_split','preprocess_source','skeleton','fps')
    if any(not protocol.get(key) for key in required) or protocol['evaluator_id']!=spec['evaluator_id']:
        raise MissingResources('Official evaluator/preprocessing provenance is incomplete or mismatched')
    for field,digest in (('evaluator_checkpoint','evaluator_sha256'),('normalization_file','normalization_sha256'),('embedding_archive','embedding_sha256')):
        if not Path(protocol[field]).is_file() or checksum(protocol[field])!=protocol[digest]:
            raise MissingResources(f'Official evaluation artifact absent/stale: {field}')
    archive=Path(protocol['embedding_archive'])
    with zipfile.ZipFile(archive) as source:
        payload=sum(x.file_size for x in source.infolist())
    # Embeddings only, max 10k x 1024 floats; no text/motion model allocation.
    if payload>512*1024**2 or 4*payload+2*1024**3>budgets(cfg)['ram_bytes']:
        raise ResourceGate('Embedding evaluator working-memory bound exceeds current budget')
    return spec,protocol


def evaluate_official(cfg,name,protocol_file,run_dir,seed):
    spec,protocol=preflight(cfg,name,protocol_file)
    with np.load(protocol['embedding_archive'],allow_pickle=False) as data:
        gen=data['generated_motion']; text=data['paired_text']; reference=data['reference_motion']
    if any(x.ndim!=2 or x.shape[1]!=spec['embedding_dim'] or not np.isfinite(x).all() for x in (gen,text,reference)):
        raise ValueError('Invalid/non-finite official embedding shape')
    n=len(gen); batch=spec['retrieval_batch']
    if text.shape!=gen.shape or n<320 or n%batch or len(reference)<320 or max(n,len(reference))>10000:
        raise MissingResources('This distribution protocol requires >=320 samples, paired text, complete 32-item retrieval batches and >=320 real references. Tiny demos are insufficient.')
    path=ROOT/cfg['baselines'][name]['repo']/spec['metrics_source']
    loader=importlib.util.spec_from_file_location('motiogen_official_metrics',path)
    upstream=importlib.util.module_from_spec(loader); loader.loader.exec_module(upstream)
    np.random.seed(seed)
    order=np.random.permutation(n); gen=gen[order]; text=text[order]
    if name=='motionlcm':
        import torch
        mu,cov=upstream.calculate_activation_statistics_np(gen); ref_mu,ref_cov=upstream.calculate_activation_statistics_np(reference)
        fid=upstream.calculate_frechet_distance_np(mu,cov,ref_mu,ref_cov)
        diversity=upstream.calculate_diversity_np(gen,spec['diversity_pairs'])
        ranks=[]
        for i in range(0,n,batch):
            distances=upstream.euclidean_distance_matrix(torch.from_numpy(text[i:i+batch]),torch.from_numpy(gen[i:i+batch])).nan_to_num()
            ranks.append(upstream.calculate_top_k(torch.argsort(distances,dim=1),3).numpy())
    else:
        mu,cov=upstream.calculate_activation_statistics(gen); ref_mu,ref_cov=upstream.calculate_activation_statistics(reference)
        fid=upstream.calculate_frechet_distance(mu,cov,ref_mu,ref_cov)
        diversity=upstream.calculate_diversity(gen,spec['diversity_pairs'])
        ranks=[upstream.calculate_R_precision(text[i:i+batch],gen[i:i+batch],3) for i in range(0,n,batch)]
    metrics={'FID':float(fid),'diversity':float(diversity),'R_precision_top_k':np.concatenate(ranks).mean(0).tolist()}
    if not all(np.isfinite(np.asarray(value)).all() for value in metrics.values()):
        raise ValueError('Official metric returned a non-finite result')
    result={'status':'success','metrics':metrics,'n_generated':n,'n_reference':len(reference),'retrieval_batch':batch,
            'diversity_pairs':spec['diversity_pairs'],'seed':seed,'protocol':protocol,'official_metric_source':str(path),
            'official_metric_source_sha256':checksum(path),'scope':'One evaluation replication. Confidence intervals require repeated official-protocol replications; cannot compare across incompatible evaluators.'}
    json_write(Path(run_dir)/'official_metrics.json',result)
    return result
