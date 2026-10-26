"""Extra inference-only CHOIS sampler timings, preserving primary output/RNG."""
import time
from .runtime import json_write

def install(adapter,repetitions=3):
    import torch
    if adapter.name!='chois':raise ValueError('Dedicated sampler timing currently supports CHOIS')
    original=adapter.model.sample
    def sample(*a,**kw):
        cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state()
        torch.cuda.synchronize();start=time.perf_counter()
        primary=original(*a,**kw)
        torch.cuda.synchronize();cold=time.perf_counter()-start
        rows=[]
        # The primary result and subsequent global RNG state remain unchanged.
        with torch.random.fork_rng(devices=[0]):
            for i in range(repetitions+1):
                torch.set_rng_state(cpu);torch.cuda.set_rng_state(gpu)
                torch.cuda.synchronize();start=time.perf_counter()
                extra=original(*a,**kw)
                torch.cuda.synchronize();elapsed=time.perf_counter()-start
                del extra
                rows.append({'seconds':elapsed,'warmup':i==0})
        json_write(adapter.path/'native_sampler_timing.json',{'cold_sampler_seconds':cold,'repetitions':rows,
            'scope':'Same native inputs/noise and fixed weights, excluding data loading, mesh reconstruction, metrics and video; one warmup excluded from steady sample set'})
        return primary
    adapter.model.sample=sample
