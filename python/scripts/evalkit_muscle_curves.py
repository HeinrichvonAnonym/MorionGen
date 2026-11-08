#!/usr/bin/env python
"""Muscle-load curve overview of an evalkit batch: one panel per row (methods as rows of the grid), lower-limb muscle
effort (sum of squared activations of the 80 leg muscles; trunk and arms are torque-driven and not in it) over time,
the residual force of the same evaluation on a second axis and the frames without any foot contact shaded (there the
pelvis residual actuators carry the body weight and the leg effort is not a load estimate).

  evalkit_muscle_curves.py --manifest M.json --rows DIR [--compare-rows DIR2 --compare-suffix __selffloor --compare-label L] --out F.png
--quantity upper: lumbar and shoulder net joint moment / body mass instead (config/evalkit_upper_body_20261107.json;
inverse-dynamics moments of the torque-driven Rajagopal trunk and arms, no trunk or shoulder muscles: not a load).
--quantity lumbar: L5/S1 compression along the spine tangent / body weight of the upper-body muscle dimension (FBLS,
config/evalkit_upper_muscle_20261108.json; DIR/<id>/upper/lumbar/frames.npz), its type-2 (unreachable) frames shaded grey
and the forced frames (config/evalkit_v12_20261108.json forced: unreachable only by the external residual) shaded gold.
v1.2 view: --manifest manifest_selffloor.json --rows rows_selffloor --compare-rows rows --compare-strip __selffloor
(self floor primary, method floor dashed).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'utils')); sys.path.insert(0, str(ROOT/'utils/motiogen'))


def load(d, quantity='leg'):
    d = Path(d)
    if not (d/'frames.npz').exists():
        return None
    f = dict(np.load(d/'frames.npz', allow_pickle=True))
    if 'muscle_effort' not in f:
        return None
    s = f['scored']; t0 = f['times'][s][0]
    contact = np.any([f[f'contact_{a}_{b}'] for a in 'rl' for b in ('heel', 'toe')], 0)
    out = {'t': f['muscle_times']-t0, 'effort': f['muscle_effort'], 'rt': f['times'][s]-t0, 'residual': f['residual_bw'][s],
           'no_contact': ~contact[s], 'mean': float(np.mean(f['muscle_effort']))}
    if quantity == 'upper':
        for prefix in ('net_moment', 'muscle'):
            if f'{prefix}_lumbar_nm_kg' in f:
                out['effort'], out['shoulder'] = f[f'{prefix}_lumbar_nm_kg'], f[f'{prefix}_shoulder_nm_kg']
                break
        else:
            from motiogen.evalkit import muscle
            up = muscle.upper_body(d/'work/so', d/'work/model_so.osim')
            out['effort'], out['shoulder'] = up['lumbar'], up['shoulder']
        out['mean'] = float(np.mean(out['effort']))
    if quantity == 'lumbar':
        if not (d/'upper/lumbar/frames.npz').exists():
            return None
        u = np.load(d/'upper/lumbar/frames.npz'); rec = json.loads((d/'upper/lumbar/lumbar.json').read_text())
        bw = json.loads((d/'evalkit.json').read_text())['receipt']['mass_kg']*9.80665
        out['t'], out['effort'], out['unreachable'] = u['times']-t0, u['L5_S1_compression_n']/bw, ~u['reachable']
        out['forced_only'] = u['forced'] & ~u['reachable'] if 'forced' in u else np.zeros_like(out['unreachable'])
        out['status'] = rec['status']; out['type2'] = rec['metrics']['type2_fraction']
        out['forced_p95'] = rec.get('forced_metrics', {}).get('forced_L5_S1_compression_p95_bw')
        out['mean'] = rec['metrics'].get('L5_S1_compression_p95_bw', float('nan'))
    return out


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--manifest', required=True); a.add_argument('--rows', required=True); a.add_argument('--out', required=True)
    a.add_argument('--label', default='main arm'); a.add_argument('--compare-rows'); a.add_argument('--compare-suffix', default='')
    a.add_argument('--compare-strip', default='', help='suffix removed from the row id before --compare-suffix is added')
    a.add_argument('--compare-label', default='compare'); a.add_argument('--title', default='')
    a.add_argument('--quantity', choices=('leg', 'upper', 'lumbar'), default='leg')
    args = a.parse_args()
    upper, lumbar = args.quantity == 'upper', args.quantity == 'lumbar'
    what = {'upper': 'lumbar net moment (N m/kg)', 'lumbar': 'L5/S1 compression (BW, FBLS)', 'leg': 'leg muscle effort'}[args.quantity]
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
            r = per[m][j]; x = load(Path(args.rows)/r['id'], args.quantity)
            y = load(Path(args.compare_rows)/f"{r['id'].removesuffix(args.compare_strip)}{args.compare_suffix}", args.quantity) if args.compare_rows else None
            if x is None:
                ax.text(.5, .5, 'no muscle output', ha='center', transform=ax.transAxes); continue
            for k in np.flatnonzero(np.diff(np.r_[0, x['no_contact'].astype(int), 0]) == 1):
                end = k+np.argmax(~np.r_[x['no_contact'][k:], False])
                ax.axvspan(x['rt'][k], x['rt'][min(end, len(x['rt'])-1)], color='tab:red', alpha=.12, lw=0)
            if lumbar:
                for k in np.flatnonzero(np.diff(np.r_[0, x['unreachable'].astype(int), 0]) == 1):
                    end = k+np.argmax(~np.r_[x['unreachable'][k:], False])
                    ax.axvspan(x['t'][k], x['t'][min(end, len(x['t'])-1)], color='0.5', alpha=.3, lw=0)
                for k in np.flatnonzero(np.diff(np.r_[0, x['forced_only'].astype(int), 0]) == 1):
                    end = k+np.argmax(~np.r_[x['forced_only'][k:], False])
                    ax.axvspan(x['t'][k], x['t'][min(end, len(x['t'])-1)], color='gold', alpha=.45, lw=0)
            ax.plot(x['t'], x['effort'], color='tab:blue', lw=1.4, label=args.label)
            if upper:
                ax.plot(x['t'], x['shoulder'], color='tab:green', lw=1., ls=':')
            if y is not None:
                ax.plot(y['t'], y['effort'], color='tab:orange', lw=1.2, ls='--', label=args.compare_label)
            ax2 = ax.twinx(); ax2.plot(x['rt'], x['residual'], color='0.45', lw=.7); ax2.set_ylim(0, max(1., float(x['residual'].max())*1.05))
            ax2.tick_params(labelsize=6, colors='0.4')
            if j == ncol-1:
                ax2.set_ylabel('residual (BW)', fontsize=7, color='0.4')
            name = r['id'][len(m)+1:] if r['id'].startswith(m+'_') else r['id']
            fp = '' if not lumbar or x['forced_p95'] is None else f"  forced {x['forced_p95']:.2f}"
            stat = (f"p95 {x['mean']:.2f}" + (f" / {y['mean']:.2f}" if y else '') + f" BW{fp}  type2 {x['type2']:.0%}  {x['status']}" if lumbar else
                    f"mean {x['mean']:.2f}" + (f" / {y['mean']:.2f}" if y else ''))
            ax.set_title(f"{name[:40]}\n{stat}  no contact {x['no_contact'].mean():.0%}", fontsize=7)
            ax.tick_params(labelsize=6)
            if j == 0:
                ax.set_ylabel(f'{m}\n' + {'upper': 'net moment (N m/kg)', 'lumbar': 'L5/S1 (BW)', 'leg': 'effort (sum a²)'}[args.quantity], fontsize=8)
            if i == len(methods)-1:
                ax.set_xlabel('time (s)', fontsize=7)
            summary.setdefault(m, []).append({'id': r['id'], 'mean': x['mean'], 'compare_mean': y['mean'] if y else None,
                                              'no_contact_share': float(x['no_contact'].mean())})
    handles = [plt.Line2D([], [], color='tab:blue', lw=1.4, label=f'{what}, {args.label}'),
               plt.Line2D([], [], color='0.45', lw=.7, label=f'residual force (BW), {args.label}'),
               matplotlib.patches.Patch(color='tab:red', alpha=.12, label=f'no foot contact, {args.label}')]
    if upper:
        handles.insert(1, plt.Line2D([], [], color='tab:green', lw=1., ls=':', label=f'shoulder net moment (N m/kg, max side), {args.label}'))
    if lumbar:
        handles.append(matplotlib.patches.Patch(color='0.5', alpha=.3, label='type-2 frames (unreachable: residual, lumbar reserve, range limit)'))
        handles.append(matplotlib.patches.Patch(color='gold', alpha=.45, label='forced frames (unreachable by the external residual only)'))
    if args.compare_rows:
        handles.insert(1, plt.Line2D([], [], color='tab:orange', lw=1.2, ls='--', label=f'{what}, {args.compare_label}'))
    fig.suptitle(args.title, y=.998, fontsize=12)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .985), ncol=len(handles), fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .965))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    Path(args.out).with_suffix('.json').write_text(json.dumps(summary, indent=1))
    print(args.out)


if __name__ == '__main__':
    main()
