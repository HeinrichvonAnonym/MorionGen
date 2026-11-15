#!/usr/bin/env python3
"""Run the pinned GVHMR demo inference while omitting its optional renders."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--static-cam", action="store_true")
    args = parser.parse_args()

    os.chdir(args.repo)
    sys.path.insert(0, str(args.repo))
    # Reuse the official argument/config builder so preprocessing, checkpoint
    # selection and output layout remain identical to tools/demo/demo.py.
    official_argv = [
        "tools/demo/demo.py", "--video", str(args.video),
        "--output_root", str(args.output_root),
    ]
    if args.static_cam:
        official_argv.append("--static_cam")
    sys.argv = official_argv

    import hydra
    import torch
    from tools.demo.demo import load_data_dict, parse_args_to_cfg, run_preprocess
    from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
    from hmr4d.utils.net_utils import detach_to_cpu
    from hmr4d.utils.pylogger import Log

    cfg = parse_args_to_cfg()
    run_preprocess(cfg)
    data = load_data_dict(cfg)
    result_path = Path(cfg.paths.hmr4d_results)
    if not result_path.exists():
        model: DemoPL = hydra.utils.instantiate(cfg.model, _recursive_=False)
        model.load_pretrained_model(cfg.ckpt_path)
        model = model.eval().cuda()
        with torch.no_grad():
            prediction = detach_to_cpu(model.predict(data, static_cam=cfg.static_cam))
        result_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(prediction, result_path)
    Log.info(f"[MotionRetarget] Body result: {result_path}; optional demo rendering skipped")
    print(result_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
