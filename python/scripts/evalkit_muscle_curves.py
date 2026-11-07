#!/usr/bin/env python
"""Muscle-load curve overview of an evalkit batch: one panel per row (methods as rows of the grid), lower-limb muscle
effort (sum of squared activations of the 80 leg muscles; trunk and arms are torque-driven and not in it) over time,
the residual force of the same evaluation on a second axis and the frames without any foot contact shaded (there the
pelvis residual actuators carry the body weight and the leg effort is not a load estimate).

  evalkit_muscle_curves.py --manifest M.json --rows DIR [--compare-rows DIR2 --compare-suffix __selffloor --compare-label L] --out F.png
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load(d):
    d = Path(d)
    if not (d/'frames.npz').exists():
        return None
    f = dict(np.load(d/'frames.npz', allow_pickle=True))
    if 'muscle_effort' not in f:
        return None
    s = f['scored']; t0 = f['times'][s][0]
    contact = np.any([f[f'contact_{a}_{b}'] for a in 'rl' for b in ('heel', 'toe')], 0)
    return {'t': f['muscle_times']-t0, 'effort': f['muscle_effort'], 'rt': f['times'][s]-t0, 'residual': f['residual_bw'][s],
            'no_contact': ~contact[s], 'mean': float(np.mean(f['muscle_effort']))}


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--manifest', required=True); a.add_argument('--rows', required=True); a.add_argument('--out', required=True)
    a.add_argument('--label', default='main arm'); a.add_argument('--compare-rows'); a.add_argument('--compare-suffix', default='')
    a.add_argument('--compare-label', default='compare'); a.add_argument('--title', default='')
    args = a.parse_args()
    rows = json.loads(Path(args.manifest).read_text())['rows']
    methods = sorted({r['method'] for r in rows})
    per = {m: [r for r in rows if r['method'] == m] for m in methods}
    ncol = max(len(v) for v in per.values())
    fig, axes = plt.subplots(len(methods), ncol, figsize=(4.2*ncol, 2.5*len(methods)), squeeze=False)
    summary = {}
    for i, m in enumerate(methods):
        for j in range(ncol):
            ax = axes[i, j]
            if j >= len(per[m]):
                ax.axis('off'); continue
            r = per[m][j]; x = load(Path(args.rows)/r['id'])
            y = load(Path(args.compare_rows)/f"{r['id']}{args.compare_suffix}") if args.compare_rows else None
            if x is None:
                ax.text(.5, .5, 'no muscle output', ha='center', transform=ax.transAxes); continue
            for k in np.flatnonzero(np.diff(np.r_[0, x['no_contact'].astype(int), 0]) == 1):
                end = k+np.argmax(~np.r_[x['no_contact'][k:], False])
                ax.axvspan(x['rt'][k], x['rt'][min(end, len(x['rt'])-1)], color='tab:red', alpha=.12, lw=0)
            ax.plot(x['t'], x['effort'], color='tab:blue', lw=1.4, label=args.label)
            if y is not None:
                ax.plot(y['t'], y['effort'], color='tab:orange', lw=1.2, ls='--', label=args.compare_label)
            ax2 = ax.twinx(); ax2.plot(x['rt'], x['residual'], color='0.45', lw=.7); ax2.set_ylim(0, max(1., float(x['residual'].max())*1.05))
            ax2.tick_params(labelsize=6, colors='0.4')
            if j == ncol-1:
                ax2.set_ylabel('residual (BW)', fontsize=7, color='0.4')
            name = r['id'][len(m)+1:] if r['id'].startswith(m+'_') else r['id']
            ax.set_title(f"{name[:40]}\nmean {x['mean']:.2f}" + (f" / {y['mean']:.2f}" if y else '') +
                         f"  no contact {x['no_contact'].mean():.0%}", fontsize=7)
            ax.tick_params(labelsize=6)
            if j == 0:
                ax.set_ylabel(f'{m}\neffort (sum a²)', fontsize=8)
            if i == len(methods)-1:
                ax.set_xlabel('time (s)', fontsize=7)
            summary.setdefault(m, []).append({'id': r['id'], 'mean': x['mean'], 'compare_mean': y['mean'] if y else None,
                                              'no_contact_share': float(x['no_contact'].mean())})
    handles = [plt.Line2D([], [], color='tab:blue', lw=1.4, label=f'leg muscle effort, {args.label}'),
               plt.Line2D([], [], color='0.45', lw=.7, label=f'residual force (BW), {args.label}'),
               matplotlib.patches.Patch(color='tab:red', alpha=.12, label=f'no foot contact, {args.label}')]
    if args.compare_rows:
        handles.insert(1, plt.Line2D([], [], color='tab:orange', lw=1.2, ls='--', label=f'leg muscle effort, {args.compare_label}'))
    fig.suptitle(args.title, y=.998, fontsize=12)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .985), ncol=len(handles), fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .965))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    Path(args.out).with_suffix('.json').write_text(json.dumps(summary, indent=1))
    print(args.out)


if __name__ == '__main__':
    main()
