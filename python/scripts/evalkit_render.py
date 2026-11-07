#!/usr/bin/env python
"""Acceptance video of one synthetic clip: evalkit (left) against the truth (right), with the three dimensions over
time: residual force (truth label shaded, tool judgment ticks), REBA and RULA score ranges with their action
thresholds, and muscle effort. Meshes, muscles, GRF and residual arrows as render_biomech_animation. OpenSim python.

  evalkit_render.py --clip <synth clip dir> --tool <evalkit out dir> --truth <truth_eval dir> --out <mp4> --title TEXT
  evalkit_render.py --smoke-method M --rows <evalkit_batch out dir> --out <mp4>     five generated rows of one method
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
    for side, d in ((tool, args.tool), (te, args.truth)):
        if 'muscle_effort' not in side and (d/'video_muscle.npz').exists():
            side.update(np.load(d/'video_muscle.npz'))
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
    tau = SPEC['residual']['tau_bw']
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
        for a0, a1 in runs(tr['unmeasurable_label'] == 1):
            ax.axvspan(tr['times'][a0], tr['times'][min(a1, len(tr['times'])-1)], ymin=0, ymax=.07, color='#7b3fa0', lw=0)
        if 'residual_positive' in tool:
            p = tool['residual_positive']
            ax.scatter(tool['times'][p], np.full(p.sum(), 1.07), s=4, c='#d81b8c', marker='|')
        u = tool['unmeasurable']
        ax.scatter(tool['times'][u], np.full(u.sum(), .97), s=4, c='#7b3fa0', marker='|')
        if tau:
            ax.axhline(tau['contact'], color='#999', lw=.7, ls='--')
        ax.set_ylim(0, 1.15); ax.set_xlim(start, end); ax.set_ylabel('残余力 BW', fontsize=9)
        ax.legend(loc='upper left', fontsize=8, frameon=False, ncol=2)
        ax.text(.99, .85, '底色：真值阳性（红）/ 模糊（灰）；底部紫条：真值不可测；顶部竖线：工具残余力判阳（品红）/ 工具不可测（紫）',
                transform=ax.transAxes, ha='right', va='top', fontsize=8)
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


CLASS_COLOUR = {0: '#9e9e9e', 1: '#d84b3a', 2: '#f2c14e', 3: '#5aa469'}


def smoke(rows_dir, method, out, fps, seconds):
    """One generated method: five evalkit rows side by side, each with residual force (tool positives ticked),
    REBA / RULA class bands and muscle effort under its model view (no truth exists for generated motion)."""
    osim.Logger.setLevelString('error')
    from evalkit.spec import SPEC
    rows = []
    for d in sorted(Path(rows_dir).iterdir()):
        if (d/'evalkit.json').exists() and json.loads((d/'evalkit.json').read_text())['row']['method'] == method:
            rows.append(d)
    rows = rows[:5]
    fig = plt.figure(figsize=(20, 10), dpi=96)
    width = 1/len(rows)
    tau = SPEC['residual']['tau_bw']
    panels, windows, cursors, axes = [], [], [], []
    for i, d in enumerate(rows):
        rec, f = json.loads((d/'evalkit.json').read_text()), dict(np.load(d/'frames.npz', allow_pickle=True))
        w, floor = d/'work', rec['receipt']['floor_m']
        sc = f['times'][f['scored']]
        start = sc[0]; end = min(sc[-1], start+seconds); windows.append((start, end))
        reba = rec['ergo']['reba']
        title = (f"{rec['row']['id']}\n{(rec['receipt']['prompt'] or '（无文本）')[:48]}\n"
                 f"残余力判阳 {rec['residual']['positive_share']:.0%}  不可测 {rec['ik']['unmeasurable_share']:.0%}\n"
                 f"REBA 风险/不定/可接受 {reba['certain_risk_share']:.0%}/{reba['indeterminate_share']:.0%}/{reba['certain_acceptable_share']:.0%}")
        panels.append(KitPanel(method, method, w/'model.osim', w/'coordinates.mot', w/'grf.mot', w/'so', f['times'], f['com'],
                               f['residual_force'], rba.trc(w/'markers.trc'), floor))
        axes.append(fig.add_axes([i*width+.005, .40, width-.01, .48]))
        fig.text(i*width+width/2, .89, title, ha='center', va='bottom', fontsize=8)
        strips = [fig.add_axes([i*width+.03, .27-k*.11, width-.045, .09]) for k in range(3)]
        ax = strips[0]
        ax.plot(f['times'], f['residual_bw'], color='#d81b8c', lw=1)
        p = f['residual_positive'] | f['unmeasurable']
        ax.scatter(f['times'][p], np.full(p.sum(), 1.05), s=3, c='#d81b8c', marker='|')
        ax.axhline(tau['contact'], color='#999', lw=.7, ls='--'); ax.set_ylim(0, 1.15); ax.set_xlim(start, end)
        ax.set_ylabel('残余力 BW', fontsize=8)
        ax = strips[1]
        for k, m in enumerate(('reba', 'rula')):
            c = f[f'ergo_{m}_class']; t = f['ergo_times']
            ax.scatter(t, np.full(len(t), 1-k), c=[CLASS_COLOUR[int(x)] for x in c], s=14, marker='s', lw=0)
        ax.set_yticks([1, 0], ['REBA', 'RULA'], fontsize=8); ax.set_ylim(-.6, 1.6); ax.set_xlim(start, end)
        ax = strips[2]
        ax.plot(f['muscle_times'], f['muscle_effort'], color='#3a6fb0', lw=1)
        ax.set_xlim(start, end); ax.set_ylim(0, None); ax.set_ylabel('Σa²', fontsize=8); ax.set_xlabel('时间 (s)', fontsize=8)
        for s in strips: s.tick_params(labelsize=7)
        cursors += [(s.axvline(start, color='k', lw=.8), i) for s in strips]
    fig.suptitle(f'{method}：E4 C0 每方法 5 条（评估工具；生成数据无真值）', fontsize=13, y=.995)
    fig.text(.5, .005, '类别色：红 确定风险 / 黄 不确定 / 绿 确定可接受 / 灰 不可测；残余力顶部竖线为工具判阳或不可测帧；'
             '绿箭头 GRF，品红箭头残余力；肌肉颜色为静态优化激活', ha='center', fontsize=9, color='#444')
    out.parent.mkdir(parents=True, exist_ok=True)
    W, H = fig.canvas.get_width_height()
    enc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
                            '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
                            '-pix_fmt', 'yuv420p', '-threads', '2', '-movflags', '+faststart', str(out)], stdin=subprocess.PIPE)
    longest = max(e-s for s, e in windows)
    for k, u in enumerate(np.arange(0, longest, 1/fps)):
        for ax, panel, (s, e) in zip(axes, panels, windows):
            rba.draw(ax, panel, panel.pose(min(s+u, e)))
        for c, i in cursors:
            t = min(windows[i][0]+u, windows[i][1]); c.set_xdata([t, t])
        fig.canvas.draw()
        enc.stdin.write(np.ascontiguousarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3]).tobytes())
        if k == int(.5*longest*fps):
            fig.savefig(out.with_suffix('.png'), dpi=96)
    enc.stdin.close()
    if enc.wait():
        raise SystemExit(f'ffmpeg failed for {out}')
    print(f'done {out}', flush=True)


def runs(mask):
    edges = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


if __name__ == '__main__':
    if '--smoke-method' in sys.argv:
        s = argparse.ArgumentParser()
        s.add_argument('--smoke-method', required=True); s.add_argument('--rows', type=Path, required=True)
        s.add_argument('--out', type=Path, required=True); s.add_argument('--fps', type=int, default=12)
        s.add_argument('--seconds', type=float, default=8.)
        x = s.parse_args()
        smoke(x.rows, x.smoke_method, x.out, x.fps, x.seconds)
    else:
        main()
