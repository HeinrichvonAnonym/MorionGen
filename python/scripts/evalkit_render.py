#!/usr/bin/env python
"""Acceptance video of one synthetic clip: evalkit (left) against the truth (right), with the three dimensions over
time: residual force (truth label shaded, tool judgment ticks), REBA and RULA score ranges with their action
thresholds, and muscle effort. Meshes, muscles, GRF and residual arrows as render_biomech_animation. OpenSim python.

  evalkit_render.py --clip <synth clip dir> --tool <evalkit out dir> --truth <truth_eval dir> --out <mp4> --title TEXT
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'utils/motiogen'), str(ROOT/'python/scripts')]
import render_biomech_animation as rba  # noqa: E402
from render_biomech_animation import osim, plt, table, vtp, vector  # noqa: E402

LABEL_COLOUR = {1: '#f4c7c3', -1: '#e6e6e6'}
CLASS_NAME = {0: '不可测', 1: '确定风险', 2: '不确定', 3: '确定可接受'}


class KitPanel(rba.Panel):
    def __init__(self, label, title, model_path, coordinates, grf_mot, so_dir, times, com, residual, markers, floor):
        """floor: the panel's floor height; everything is drawn shifted so that it lies on the grid (y = 0)."""
        self.label, self.title, self.metrics = label, title, {}
        self.model = osim.Model(str(model_path)); self.state = self.model.initSystem()
        self.weight = self.model.getTotalMass(self.state)*rba.GRAVITY
        names, data = table(coordinates)
        cs = self.model.getCoordinateSet()
        self.time, self.coordinates = data[:, 0], []
        for i, name in enumerate(names[1:], 1):
            if cs.contains(name):
                c = cs.get(name)
                v = np.deg2rad(data[:, i]) if c.getMotionType() == 1 else data[:, i]-(floor if name == 'pelvis_ty' else 0.)
                self.coordinates.append((c, v))
        tx, tz = data[:, names.index('pelvis_tx')], data[:, names.index('pelvis_tz')]
        kernel = np.ones(101)/101
        self.centre = np.stack([np.convolve(np.pad(v, 50, mode='edge'), kernel, 'valid') for v in (tx, tz)], 1)
        self.path = np.stack([tx, np.zeros_like(tx), tz], 1)
        self.meshes = []
        for component in self.model.getComponentsList():
            if component.getConcreteClassName() == 'Mesh':
                mesh = osim.Mesh.safeDownCast(component)
                points, triangles = vtp(mesh.get_mesh_file(), vector(mesh.get_scale_factors()))
                self.meshes.append((mesh.getFrame(), points, triangles))
        self.muscles = [self.model.getMuscles().get(i) for i in range(self.model.getMuscles().getSize())]
        act = Path(so_dir)/'so_so_activation.sto' if so_dir else None
        if act and act.exists():
            n, d = table(act); self.activation_time = d[:, 0]
            self.activation = np.stack([d[:, n.index(m.getName())] for m in self.muscles], 1)
        else:
            self.activation_time = self.time[[0, -1]]; self.activation = np.zeros((2, len(self.muscles)))
        self.grf = table(grf_mot)[1].copy(); self.grf[:, [5, 14]] -= floor
        self.dynamics_time, self.residual = times, residual
        self.com = com-np.array([0., floor, 0.])
        self.marker_time, self.markers = markers[0], markers[1]-np.array([0., floor, 0.])


def truth_dynamics(model_path, coordinates, tr, mass, floor):
    import biomech
    t, labels, data = biomech.read_storage(coordinates)
    kin = biomech.body_kinematics(osim.Model(str(model_path)), labels, data)
    F, M, C = biomech.required_wrench(kin, float(np.median(np.diff(t))))
    points = {s: biomech.sole_points(kin, s)[0] for s in ('r', 'l')}
    masks = {s: {k: tr[f'contact_{s}_{k}'] for k in ('heel', 'toe')} for s in ('r', 'l')}
    _, rF, _ = biomech.estimate_grf(F, M, C, points, masks, floor, mass*9.80665)
    return t, C, rF


def joints_yup(path):
    import biomech_sources
    j, _, fps = biomech_sources.load(path)
    return np.arange(len(j))/fps, j


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--clip', type=Path, required=True); a.add_argument('--tool', type=Path, required=True)
    a.add_argument('--truth', type=Path, required=True); a.add_argument('--out', type=Path, required=True)
    a.add_argument('--title', required=True); a.add_argument('--fps', type=int, default=25)
    args = a.parse_args()
    osim.Logger.setLevelString('error')
    from evalkit.spec import SPEC
    meta = json.loads((args.clip/'input/meta.json').read_text())
    tr = dict(np.load(args.clip/'truth/truth.npz'))
    tool = dict(np.load(args.tool/'frames.npz', allow_pickle=True))
    te = dict(np.load(args.truth/'truth_eval.npz', allow_pickle=True))
    tw = args.tool/'work'
    tool_title = f"evalkit（输入地面 {meta['floor_height_m']:+.3f} m，含 ±5 cm 平移）"
    left = KitPanel('evalkit', tool_title, tw/'model.osim', tw/'coordinates.mot', tw/'grf.mot', tw/'so',
                    tool['times'], tool['com'], tool['residual_force'], rba.trc(tw/'markers.trc'), meta['floor_height_m'])
    t_truth, com, rF = truth_dynamics(args.truth/'model.osim', args.clip/'truth/coordinates.mot', tr, meta['mass_kg'], meta['truth_floor_m'])
    mt, mj = joints_yup(args.clip/'truth/clean_joints.npz')
    mj = mj.copy(); mj[..., 1] -= meta['floor_shift_m']
    right = KitPanel('真值', '真值（模型内构造：真接触、真坐标）', args.truth/'model.osim', args.clip/'truth/coordinates.mot',
                     args.clip/'truth/grf.mot', args.truth/'so' if (args.truth/'so').exists() else None, t_truth, com, rF, (mt, mj), meta['truth_floor_m'])
    start = max(tool['times'][tool['scored']][0], t_truth[0]); end = min(tool['times'][tool['scored']][-1], t_truth[-1])
    times = np.arange(start, end, 1/args.fps)
    fig = plt.figure(figsize=(16, 9), dpi=100)
    fig.suptitle(args.title, fontsize=13, y=.99)
    axes = [fig.add_axes([.01+i*.495, .43, .48, .52]) for i in range(2)]
    strips = [fig.add_axes([.06, .05+k*.085, .9, .07]) for k in range(4)][::-1]
    tau = SPEC['residual']['tau_bw']; depth = SPEC['residual']['penetration_depth_m']
    fig.text(.5, .405, '左：评估工具（生成式稀疏关节 → 缩放 + IK → 高度接触 → GRF 拟合）；右：真值。绿箭头 GRF，品红箭头残余力（1 BW = 0.6 m），'
             '肌肉颜色为静态优化激活（未做 SO 的片段为蓝色）', ha='center', fontsize=9.5, color='#444')

    def static_strips():
        ax = strips[0]; ax.cla()
        for lab, colour in LABEL_COLOUR.items():
            m = tr['e_label'] == lab
            for a0, a1 in runs(m):
                ax.axvspan(tr['times'][a0], tr['times'][min(a1, len(m)-1)], color=colour, lw=0)
        ax.plot(tool['times'], tool['residual_bw'], color='#d81b8c', lw=1.1, label='工具残余力')
        ax.plot(tr['times'], tr['residual_bw'], color='#3a6fb0', lw=1.1, label='真值残余力')
        if 'residual_positive' in tool:
            p = tool['residual_positive'] | tool['unmeasurable']
            ax.scatter(tool['times'][p], np.full(p.sum(), 1.05), s=4, c='#d81b8c', marker='|')
        if tau:
            ax.axhline(tau['contact'], color='#999', lw=.7, ls='--')
        ax.set_ylim(0, 1.15); ax.set_xlim(start, end); ax.set_ylabel('残余力 BW', fontsize=9)
        ax.legend(loc='upper left', fontsize=8, frameon=False, ncol=2)
        ax.text(.99, .85, '底色：真值阳性（红）/ 模糊（灰）；顶部竖线：工具判阳', transform=ax.transAxes, ha='right', va='top', fontsize=8)
        for k, (m, risk) in enumerate((('reba', 4), ('rula', 5)), 1):
            ax = strips[k]; ax.cla()
            for src, colour, lab in ((tool, '#d81b8c', '工具'), (te, '#3a6fb0', '真值')):
                b = src[f'ergo_{m}_bounds']; tt = src['ergo_times']
                ax.fill_between(tt, b[:, 0], b[:, 1], color=colour, alpha=.25, step='mid', label=f'{lab}分数区间')
            ax.axhline(risk-.5, color='k', lw=.8, ls='--')
            ax.set_xlim(start, end); ax.set_ylim(0, 13 if m == 'reba' else 8); ax.set_ylabel(m.upper(), fontsize=9)
            ax.legend(loc='upper left', fontsize=8, frameon=False, ncol=2)
        ax = strips[3]; ax.cla()
        for src, colour, lab in ((tool, '#d81b8c', '工具'), (te, '#3a6fb0', '真值')):
            if 'muscle_effort' in src:
                ax.plot(src['muscle_times'], src['muscle_effort'], color=colour, lw=1.1, label=f'{lab}全身用力 Σa²')
        ax.set_xlim(start, end); ax.set_ylim(0, None); ax.set_ylabel('Σa²', fontsize=9); ax.set_xlabel('时间 (s)')
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc='upper left', fontsize=8, frameon=False, ncol=2)
        else:
            ax.text(.5, .5, '本片段未做静态优化', transform=ax.transAxes, ha='center', fontsize=9, color='#777')

    static_strips()
    cursors = [s.axvline(start, color='k', lw=.8) for s in strips]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    w, h = fig.canvas.get_width_height()
    enc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}',
                            '-r', str(args.fps), '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '22',
                            '-pix_fmt', 'yuv420p', '-threads', '2', '-movflags', '+faststart', str(args.out)], stdin=subprocess.PIPE)
    still = start+.5*(end-start)
    for k, t in enumerate(times):
        for ax, panel in zip(axes, (left, right)):
            rba.draw(ax, panel, panel.pose(t))
        for c in cursors:
            c.set_xdata([t, t])
        fig.canvas.draw()
        enc.stdin.write(np.ascontiguousarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3]).tobytes())
        if abs(t-still) < .5/args.fps:
            fig.savefig(args.out.with_suffix('.png'), dpi=100)
    enc.stdin.close()
    if enc.wait():
        raise SystemExit(f'ffmpeg failed for {args.out}')
    print(f'done {args.out}', flush=True)


def runs(mask):
    edges = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


if __name__ == '__main__':
    main()
