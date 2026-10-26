"""Count executed native branches without changing generated tensors."""
import ast
import inspect
import sys
import textwrap
import time
from collections import Counter


class NativeBranchCounter:
    def __init__(self, model):
        self.model = model
        self.counts = Counter()
        function = model.generate_perpetual_navigation.__func__
        source, first = inspect.getsourcelines(function)
        tree = ast.parse(textwrap.dedent(''.join(source)))
        self.sites = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and isinstance(node.test, ast.Name):
                label = {'snap_to_ground': 'ground_projection',
                         'reproj_kpts': 'keypoint_reprojection'}.get(node.test.id)
                if label:
                    for statement in node.body:
                        for call in ast.walk(statement):
                            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == '_fwd_smplx_seq':
                                for line in range(call.lineno, call.end_lineno + 1):
                                    self.sites[first + line - 1] = label
        if set(self.sites.values()) != {'ground_projection', 'keypoint_reprojection'}:
            raise RuntimeError('Native PRIMAL branch locations changed')
        self.module = inspect.getmodule(function)
        self.original_fk = model._fwd_smplx_seq
        self.original_snap = model.snap_init_cond_to_ground
        self.original_inertia = self.module.inertialize

    def __enter__(self):
        def fk(*args, **kwargs):
            caller = sys._getframe(1)
            if caller.f_code.co_name == 'generate_perpetual_navigation':
                label = self.sites.get(caller.f_lineno)
                if label:
                    self.counts[label] += 1
            return self.original_fk(*args, **kwargs)
        def snap(*args, **kwargs):
            self.counts['initial_ground_snap'] += 1
            return self.original_snap(*args, **kwargs)
        def inertia(*args, **kwargs):
            self.counts['inertialization'] += 1
            return self.original_inertia(*args, **kwargs)
        self.model._fwd_smplx_seq = fk
        self.model.snap_init_cond_to_ground = snap
        self.module.inertialize = inertia
        return self

    def __exit__(self, *_):
        self.model._fwd_smplx_seq = self.original_fk
        self.model.snap_init_cond_to_ground = self.original_snap
        self.module.inertialize = self.original_inertia

    def validate(self, ground, reprojection, inertia):
        expected = {'initial_ground_snap': ground, 'ground_projection': ground,
                    'keypoint_reprojection': reprojection, 'inertialization': inertia}
        for label, enabled in expected.items():
            if bool(self.counts[label]) != enabled:
                raise RuntimeError(f'PRIMAL branch {label}: enabled={enabled}, calls={self.counts[label]}')
        return dict(self.counts)


class NativePrimitiveClock:
    """Pace native denoising primitives, without a controller or new inputs."""
    def __init__(self,model,steps,frames_per_primitive,clock):
        self.scheduler=model.noise_scheduler
        self.original=self.scheduler.step
        self.steps=steps;self.frames=frames_per_primitive;self.clock=clock
        self.calls=0;self.trace=[];self.started=None;self.wait_seconds=0.

    def __enter__(self):
        import torch
        def step(*args,**kwargs):
            if self.calls%self.steps==0:
                torch.cuda.synchronize()
                if self.started is None:self.started=time.perf_counter()
                frame=(self.calls//self.steps)*self.frames
                scheduled=frame/30.
                waited=0.
                if self.clock=='realtime':
                    wait_start=time.perf_counter()
                    time.sleep(max(0.,self.started+scheduled-wait_start))
                    waited=time.perf_counter()-wait_start
                    self.wait_seconds+=waited
                wall=time.perf_counter()-self.started
                self.trace.append({'frame':frame,'request_wall_seconds':wall,
                                   'paced_wait_seconds':waited,
                                   'schedule_lag_seconds':max(0.,wall-scheduled) if self.clock=='realtime' else None})
            result=self.original(*args,**kwargs);self.calls+=1
            if self.calls%self.steps==0:
                torch.cuda.synchronize()
                self.trace[-1]['denoising_completion_wall_seconds']=time.perf_counter()-self.started
            return result
        self.scheduler.step=step
        return self

    def __exit__(self,*_):
        self.scheduler.step=self.original
        if self.calls%self.steps:raise RuntimeError('Partial native primitive denoising call count')
