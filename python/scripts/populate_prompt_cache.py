#!/usr/bin/env python
"""Create exact ARDY prompt-cache entries for registered prompt sets.

Loads the unmodified LLM2Vec encoder once, encodes every distinct prompt with
the same batch-1 call a cache miss uses, then writes one exact cache entry per
prompt set through prompt_cache.install. Run in the motiogen-ardy environment.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'utils'))
from motiogen.config import asset_root, load_config, repo_root
from motiogen.runtime import gpu_lock, json_write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', action='append', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    sets = set()
    for path in args.plan:
        for job in json.loads(path.read_text())['jobs']:
            if job['method'] == 'ardy' and job['config'].get('_registered_prompt_cache'):
                scene = job['scene']
                sets.add(tuple(sorted({scene['prompt'], *[u['prompt'] for u in scene.get('updates', [])]})))
    cfg = load_config()
    cfg['gpu_index'] = int(os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
    repo = repo_root(cfg, 'ardy'); os.chdir(repo); sys.path.insert(0, str(repo))
    os.environ['TEXT_ENCODER_MODE'] = 'local'
    os.environ['TEXT_ENCODERS_DIR'] = str(asset_root(cfg)/'text/llm2vec')
    os.environ['CHECKPOINT_DIR'] = str(asset_root(cfg)/'ardy')
    from motiogen import local_llm
    from motiogen.prompt_cache import install
    original = local_llm.precompute_encoder
    every = sorted({p for s in sets for p in s})
    memo = {}

    def encode_all(config, name, prompts, cache_path=None):
        if not memo:
            memo.update(original(config, name, every).encoded)
        return local_llm.FrozenTextEncoder({p: memo[p] for p in prompts}, 'cuda')
    local_llm.precompute_encoder = encode_all
    args.out.mkdir(parents=True, exist_ok=True)
    install(cfg, 'ardy', args.out)
    rows = []
    with gpu_lock(cfg):
        for prompts in sorted(sets):
            local_llm.precompute_encoder(cfg, 'ardy', list(prompts))
            rows.append({'prompts': list(prompts), **json.loads((args.out/'prompt_cache.json').read_text())})
    json_write(args.out/'populate_receipt.json', {'sets': rows, 'distinct_prompts': every,
               'encoder_loads': int(bool(memo)), 'scope': 'Exact batch-1 embeddings; identical to cache-miss encoding'})
    print(json.dumps({'sets': len(rows), 'new_entries': sum(not r['hit'] for r in rows), 'prompts': len(every)}))


if __name__ == '__main__':
    main()
