#!/usr/bin/env python
"""Time one training step (forward, backward, AdamW update) of the released Kimodo denoiser.

Inputs are random tensors of the training shapes (motion clips up to 10 s at 30 fps, precomputed
50 x 4096 text features), so the text encoder is never loaded. The result anchors training-cost
estimates; it says nothing about how many steps a fine-tune needs.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

CKPT = Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c95839/MorionGen/outputs/assets/kimodo/Kimodo-SOMA-RP-v1.1')
OUT = Path(__file__).resolve().parents[2]/'develop_logs/test_runs/20261102_kimodo_step_bench'


def denoiser():
    conf = OmegaConf.load(CKPT/'config.yaml')
    conf = OmegaConf.merge(conf, {'checkpoint_dir': str(CKPT)})
    return instantiate(OmegaConf.to_container(conf, resolve=True)['denoiser'])


def step_time(model, optimizer, batch, frames, dtype, repeats):
    device = next(model.parameters()).device
    dim = model.motion_rep.motion_rep_dim
    x = torch.randn(batch, frames, dim, device=device)
    pad = torch.ones(batch, frames, dtype=torch.bool, device=device)
    text = torch.randn(batch, 50, 4096, device=device)
    text_pad = torch.ones(batch, 50, dtype=torch.bool, device=device)
    heading = torch.zeros(batch, device=device)
    mask = (torch.rand(batch, frames, dim, device=device) < .05).float()

    def once():
        t = torch.randint(0, 1000, (batch,), device=device)
        with torch.autocast('cuda', dtype=dtype, enabled=dtype != torch.float32):
            loss = (model(x, pad, text, text_pad, t, heading, mask, x).float()-x).pow(2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    for _ in range(3):
        once()
    torch.cuda.synchronize(); start = time.perf_counter()
    for _ in range(repeats):
        once()
    torch.cuda.synchronize()
    return (time.perf_counter()-start)/repeats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batches', default='4,8,16,32')
    parser.add_argument('--frames', type=int, default=300)
    parser.add_argument('--repeats', type=int, default=10)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    model = denoiser().cuda().train()
    params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    rows = []
    for dtype in (torch.float32, torch.bfloat16):
        for batch in map(int, args.batches.split(',')):
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            try:
                seconds = step_time(model, optimizer, batch, args.frames, dtype, args.repeats)
                row = {'dtype': str(dtype).split('.')[-1], 'batch': batch, 'frames': args.frames, 'step_s': seconds,
                       'samples_per_s': batch/seconds, 'peak_gb': torch.cuda.max_memory_allocated()/1e9}
            except torch.cuda.OutOfMemoryError:
                row = {'dtype': str(dtype).split('.')[-1], 'batch': batch, 'frames': args.frames, 'oom': True}
                optimizer.zero_grad(set_to_none=True)
            rows.append(row); print(json.dumps(row), flush=True)
    result = {'gpu': torch.cuda.get_device_name(), 'parameters': params, 'rows': rows}
    (OUT/f"bench_{torch.cuda.get_device_name().replace(' ', '_')}.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({'parameters': params}))


if __name__ == '__main__':
    main()
