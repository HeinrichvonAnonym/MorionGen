"""Run official DART rollout in an isolated runtime directory with local assets."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(json.loads((ROOT / 'config/baselines.json').read_text())['data_root'])


def link(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        if target.resolve() != source.resolve():
            raise RuntimeError(f'Conflicting runtime link: {target}')
    elif not target.exists():
        target.symlink_to(source, target_is_directory=source.is_dir())


def main():
    import numpy as np
    # Upstream HumanML quaternion utility uses the removed alias np.float.
    if 'float' not in np.__dict__:
        np.float = float
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--text', default='stand still*10,walk forward*10')
    parser.add_argument('--online', action='store_true')
    parser.add_argument('--native', action='store_true', help='Use official environment step and state transition')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--suite', action='store_true', help='Run the preregistered v4 multi-scenario replication')
    parser.add_argument('--frames', type=int, default=300)
    parser.add_argument('--clock', choices=['paused', 'realtime'], default='paused')
    parser.add_argument('--goal-feedback', type=int, choices=[0, 1], default=1)
    parser.add_argument('--goal-start',type=float,nargs=2,default=[1.,0.])
    parser.add_argument('--goal-next',type=float,nargs=2,default=[-1.,.5])
    parser.add_argument('--handoff', choices=['observed', 'committed'], default='committed')
    args = parser.parse_args()
    repo = DATA / 'baselines/DART'
    assets = DATA / 'outputs/assets/dart-official'
    runtime = DATA / 'outputs/runtime/dart'
    runtime.mkdir(parents=True, exist_ok=True)
    for item in repo.iterdir():
        if item.name not in ['.git', 'data', 'mvae', 'mld_denoiser', 'policy_train']:
            link(item, runtime / item.name)
    for item in (assets / 'data').iterdir():
        link(item, runtime / 'data' / item.name)
    for name in ['mvae', 'mld_denoiser', 'policy_train']:
        link(assets / name, runtime / name)
    for gender in ['male', 'female', 'neutral']:
        raw = DATA / f'outputs/assets/smplx-locked-head/{gender}/model.npz'
        compatible = DATA / f'outputs/assets/smplx-locked-head-compatible/SMPLX_{gender.upper()}.npz'
        if not compatible.exists():
            with np.load(raw, allow_pickle=True) as original, np.load(
                f'/home/heinrich/TactileLab/data/GRAB/smplx/models/smplx/SMPLX_{gender.upper()}.npz',
                allow_pickle=True) as auxiliary:
                for key in ['f', 'kintree_table']:
                    if not np.array_equal(original[key], auxiliary[key]):
                        raise RuntimeError(f'Cannot transfer metadata across different {key}')
                payload = {key: original[key] for key in original.files}
                for key in ['hands_componentsl', 'hands_componentsr', 'hands_meanl', 'hands_meanr',
                            'lmk_faces_idx', 'lmk_bary_coords', 'dynamic_lmk_faces_idx', 'dynamic_lmk_bary_coords']:
                    if key not in payload:
                        payload[key] = auxiliary[key]
                compatible.parent.mkdir(parents=True, exist_ok=True)
                np.savez(compatible, **payload)
                with np.load(compatible, allow_pickle=True) as result:
                    for key in ['v_template', 'shapedirs', 'posedirs', 'weights', 'J_regressor']:
                        if not np.array_equal(result[key], original[key]):
                            raise RuntimeError(f'Core geometry changed: {key}')
        target = runtime / f'data/smplx_lockedhead_20230207/models_lockedhead/smplx/SMPLX_{gender.upper()}.npz'
        if target.is_symlink() and target.resolve() == raw.resolve():
            target.unlink()  # Replace only the runtime link created by this script.
        link(compatible, target)
    os.chdir(runtime)
    sys.path.insert(0, str(DATA / 'staged_dependencies/pytorch3d'))
    sys.path.insert(0, str(repo))
    import numpy as np
    import torch
    import clip
    original_load = clip.load
    def local_clip(name, *positional, **kwargs):
        kwargs['download_root'] = str(DATA / 'outputs/assets/omnicontrol')
        return original_load(name, *positional, **kwargs)
    clip.load = local_clip
    if args.native:
        sys.path.insert(0, str(ROOT / 'utils'))
        from motiogen.dart_native import run
        run(args, DATA, ROOT)
        return
    from mld.rollout_mld import load_mld, RolloutArgs, SinglePrimitiveDataset, create_gaussian_diffusion, rollout
    if args.online or args.suite:
        sys.path.insert(0, str(ROOT / 'utils'))
        from motiogen.dart_feedback import run
        run(args, DATA, ROOT)
        return
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda')
    checkpoint = 'mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000.pt'
    da, dm, va, vm = load_mld(checkpoint, device)
    diffusion = create_gaussian_diffusion(da.diffusion_args)
    dataset = SinglePrimitiveDataset(cfg_path=va.data_args.cfg_path,
        dataset_path=va.data_args.data_dir, body_type=va.data_args.body_type,
        sequence_path='./data/stand.pkl', batch_size=1, device=device,
        enforce_gender='male', enforce_zero_beta=1)
    out = ROOT / f'develop_logs/test_runs/20261024_dart_smoke_seed{args.seed}'
    out.mkdir(parents=True, exist_ok=True)
    options = RolloutArgs(seed=args.seed, device=device, batch_size=1)
    options.save_dir = out
    torch.cuda.synchronize()
    start = time.perf_counter()
    rollout(args.text, da, dm, va, vm, diffusion, dataset, options)
    torch.cuda.synchronize()
    receipt = {'seed': args.seed, 'text': args.text, 'seconds': time.perf_counter() - start,
               'gpu_peak_bytes': torch.cuda.max_memory_allocated(),
               'scope': 'official scheduled text composition, not reactive goal-change',
               'body_source': '/home/heinrich/Downloads/smplx_locked_head.tar.bz2'}
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2))
    print(receipt)


if __name__ == '__main__':
    main()
