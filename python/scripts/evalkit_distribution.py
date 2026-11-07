"""Distribution evaluators: ground-truth self-checks, C0 smoke and figures (config/evalkit_v1_20261107.json `distribution`).

  selfcheck --evaluator E    split half, round trip, perturbation monotone, text match, FID-vs-N -> <out>/<E>/selfcheck.json
  smoke --evaluator E [--manifest M]   manifest rows of that evaluator -> <out>/<E>/smoke.json (ground truth from --reference)
  figure --evaluator E       PCA projection, FID vs perturbation, FID vs N, retrieval -> <out>/<E>/page.png
--reference (default: the registered self-check directory) holds <E>/selfcheck.json and embeddings.npz.
Evaluators: t2m, omomo and flowmdm_babel (motiogen-motionlcm; flowmdm_babel in its own process, its upstream package name
collides with OmniControl's), tmr_soma (motiogen-kimodo; re-executes itself there with the Kimodo environment).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'utils'))
from motiogen.evalkit import distribution as D  # noqa: E402
from motiogen.evalkit import sources  # noqa: E402

OUT = ROOT/'develop_logs/test_runs/20261107_evalkit/distribution'
MANIFEST = ROOT/'develop_logs/test_runs/20261107_evalkit/smoke8x5/manifest.json'
NOISE_M = (.01, .02, .05, .10)
SHUFFLE = (.05, .2, .5)
N_LIST = (8, 16, 32, 64, 128, 256, 512, 1024)
REPEATS = 20


def perturb(joints, kind, level, r):
    if kind == 'noise':
        return joints + r.normal(0, level, joints.shape)
    j = joints.copy()
    pick = np.sort(r.choice(len(j), max(2, int(round(level*len(j)))), replace=False))
    j[pick] = j[r.permutation(pick)]
    return j


def levels():
    return [('noise', s, f'noise_{s}') for s in NOISE_M] + [('shuffle', p, f'shuffle_{p}') for p in SHUFFLE]


def motion_file(source):
    s = Path(source)
    return s if s.suffix == '.npz' else s/'motion.npz'


class T2MAdapter:
    name = 't2m'

    def __init__(self):
        self.ev = ev = D.T2M('motionlcm')
        self.names, self.motions, self.tokens = ev.ground_truth()
        ev.offsets(self.motions[0])
        sys.path.insert(0, str(D.BASELINES/'OmniControl'))
        from data_loaders.humanml.utils.metrics import calculate_activation_statistics, calculate_frechet_distance
        self._stats, self._frechet = calculate_activation_statistics, calculate_frechet_distance

    def fid(self, a, b):
        try:
            return float(self._frechet(*self._stats(a), *self._stats(b)))
        except ValueError:  # official sqrtm check: imaginary part of a singular small-N covariance product
            return float('nan')

    def info(self):
        return {'checkpoint': str(self.ev.checkpoint.relative_to(D.DATA)), 'epoch': self.ev.epoch, 'published_real': D.PUBLISHED_REAL}

    def ground_truth(self):
        return self.ev.embed_motions(self.motions), self.ev.embed_texts(self.tokens)

    def text_metrics(self, text, motion):
        return D.text_metrics(text, motion)

    def real(self, gt, text):
        return {**D.text_metrics(text, gt), 'diversity': D.diversity(gt)}

    def perturbed(self, index, r):
        ev = self.ev
        joints = [ev.joints(self.motions[i]) for i in index]
        sets = {'roundtrip': [ev.features(j) for j in joints]}
        for kind, level, label in levels(): sets[label] = [ev.features(perturb(j, kind, level, r)) for j in joints]
        key = 'evalkit_dist_t2m_crop_a'
        emb = {'original': ev.embed_motions([self.motions[i] for i in index], key=key), **{k: ev.embed_motions(v, key=key) for k, v in sets.items()}}
        z = [np.abs((f-self.motions[i][:len(f)])/ev.std).mean() for f, i in zip(sets['roundtrip'], index)]
        return emb, {'feature_abs_z_mean': float(np.mean(z))}

    def smoke(self, rows):
        ev = self.ev
        prompts = [sources.prompt(r['source']) for r in rows]
        unique = sorted({p for p in prompts if p})
        try:
            tagged, tokenizer = dict(zip(unique, D.tag(unique))), 'spacy en_core_web_sm'
        except Exception as e:  # model not installed: smoke-only fallback, recorded per row
            tagged = dict(zip(unique, D.tag_from_ground_truth(unique)))
            tokenizer = f'fallback: most frequent tag in the official test tokens ({type(e).__name__})'
        feats, record = [], []
        for row, prompt in zip(rows, prompts):
            z = np.load(motion_file(row['source']), allow_pickle=True)
            f = ev.features(ev.joints_from_motion(z['joints'], z['joint_names'], float(z['fps'])))
            feats.append(f)
            record.append({'id': row['id'], 'method': row['method'], 'prompt': prompt, 'tokens': tagged.get(prompt), 'tokenizer': tokenizer,
                           'frames_20fps': len(f), 'finite': bool(np.isfinite(f).all())})
        emb = ev.embed_motions(feats, key='evalkit_dist_t2m_crop_smoke')
        with_text = [i for i, p in enumerate(prompts) if p and tagged.get(p)]
        for i, t in zip(with_text, ev.embed_texts([tagged[prompts[i]] for i in with_text])):
            record[i]['matching_distance'] = float(np.linalg.norm(t-emb[i]))
        return emb, record


class TMRAdapter:
    name = 'tmr_soma'

    def __init__(self):
        self.ev = D.TMRSoma()
        cases = [c for c in self.ev.cases() if json.loads((c/'meta.json').read_text()).get('text')]
        _, first = np.unique(self.ev.cached(cases, 'gt_motion_embedding.npy').round(5), axis=0, return_index=True)
        self.cases = [cases[i] for i in sorted(first)]

    def fid(self, a, b):
        return self.ev.fid(a, b)

    def info(self):
        return {'model': D.TMRSoma.MODEL, 'ground_truth_cases': str(D.KIMODO_CASES.relative_to(D.DATA))}

    def ground_truth(self):
        return self.ev.cached(self.cases, 'gt_motion_embedding.npy'), self.ev.cached(self.cases, 'text_embedding.npy')

    def text_metrics(self, text, motion):
        return self.ev.retrieval(text, motion)

    def real(self, gt, text):
        return self.ev.retrieval(text, gt)

    def perturbed(self, index, r):
        ev = self.ev
        joints = [ev.posed(self.cases[i]/'gt_motion.npz') for i in index]
        emb = {'original': ev.embed(joints), 'roundtrip_native_reencode': ev.embed(joints)}
        for kind, level, label in levels(): emb[label] = ev.embed([perturb(j, kind, level, r) for j in joints])
        cached = ev.cached([self.cases[i] for i in index], 'gt_motion_embedding.npy')
        return emb, {'reencode_vs_cached_max_abs': float(np.abs(emb['original']-cached).max())}

    def smoke(self, rows):
        ev = self.ev
        joints = [ev.posed(motion_file(r['source'])) for r in rows]
        emb = ev.embed(joints)
        cached = ev.cached([Path(r['source']) for r in rows], 'motion_embedding.npy')
        text = ev.cached([Path(r['source']) for r in rows], 'text_embedding.npy')
        record = [{'id': r['id'], 'method': r['method'], 'prompt': sources.prompt(r['source']), 'frames': len(j),
                   'finite': bool(np.isfinite(e).all()), 'reencode_vs_cached_max_abs': float(np.abs(e-c).max()),
                   't2m_similarity': float(e@t)} for r, j, e, c, t in zip(rows, joints, emb, cached, text)]
        return emb, record


def chois_native(source):
    """CHOIS run -> its native evaluation file (raw/chois_results/res_npz_files/<setting>/<sequence>.npz)."""
    files = sorted((Path(source)/'raw/chois_results/res_npz_files').glob('*/*.npz'))
    if len(files) != 1:
        raise ValueError(f'expected one native CHOIS evaluation file, found {len(files)}')
    return files[0]


class OMOMOAdapter:
    name = 'omomo'

    def __init__(self):
        self.ev = D.OMOMO()
        self.files, self.jpos, self.tokens = self.ev.ground_truth()

    def fid(self, a, b):
        return self.ev.fid(a, b)

    def info(self):
        rel = lambda p: str(p.relative_to(D.DATA))
        return {'checkpoint': rel(D.CHOIS_CHECKPOINT), 'epoch': self.ev.epoch, 'ground_truth_files': rel(D.CHOIS_RELATED/'res_npz_files/gt'),
                'official_chois_reference': self.official()}

    def official(self):
        """Released CHOIS results (res_npz_files/chois, same test windows) through this evaluator, for scale only."""
        files = sorted((D.CHOIS_RELATED/'res_npz_files/chois').glob('*.npz'))
        loaded = [self.ev.load(f) for f in files]
        emb = self.ev.embed_motions([j for j, _ in loaded]); text = self.ev.embed_texts([self.ev.tokens(s) for _, s in loaded])
        gt = self.ev.embed_motions(self.jpos)
        return {'samples': len(files), 'fid_vs_ground_truth': self.ev.fid(emb, gt), **self.ev.text_metrics(text, emb),
                'diversity': D.diversity(emb)}

    def ground_truth(self):
        return self.ev.embed_motions(self.jpos), self.ev.embed_texts(self.tokens)

    def text_metrics(self, text, motion):
        return self.ev.text_metrics(text, motion)

    def real(self, gt, text):
        return {**self.ev.text_metrics(text, gt), 'diversity': D.diversity(gt)}

    def perturbed(self, index, r):
        ev = self.ev
        joints = [self.jpos[i] for i in index]
        emb = {'original': ev.embed_motions(joints), 'roundtrip_native_reencode': ev.embed_motions([ev.load(self.files[i])[0] for i in index])}
        for kind, level, label in levels(): emb[label] = ev.embed_motions([perturb(j, kind, level, r) for j in joints])
        return emb, {'input': 'native global_jpos files (no conversion); round trip = reload and re-encode'}

    def smoke(self, rows):
        ev, record, jpos, text = self.ev, [], [], []
        for r in rows:
            f = chois_native(r['source']); j, seq = ev.load(f)
            z = np.load(motion_file(r['source']), allow_pickle=True)
            jpos.append(j); text.append(ev.tokens(seq))
            record.append({'id': r['id'], 'method': r['method'], 'sequence': seq, 'native_file': f.parent.name, 'frames': len(j),
                           'finite': bool(np.isfinite(j).all()), 'prompt': sources.prompt(r['source']),
                           'native_vs_motion_npz_shape': [list(j.shape), list(z['joints'].shape)]})
        emb = ev.embed_motions(jpos)
        for rec, t, e in zip(record, ev.embed_texts(text), emb):
            rec['matching_distance'] = float(np.linalg.norm(t-e))
        return emb, record


class FlowMDMAdapter:
    """Protocol: config/evalkit_dist_flowmdm_babel_20261107.json (scoring reference = ground-truth joints through the IK)."""
    name = 'flowmdm_babel'
    WINDOW = 200

    def __init__(self):
        self.ev = ev = D.FlowMDMBabel()
        self.gt, self.ref = ev.ground_truth(), ev.ik_reference()
        self.tokens = D.tag_from_ground_truth(self.gt['text'], unknown='OTHER')
        self.coverage = float(np.mean([not t.endswith('/OTHER') for s in self.tokens for t in s]))

    def fid(self, a, b):
        try:
            return D.fid(a, b)
        except ValueError:  # official sqrtm check on a singular small-N covariance product
            return float('nan')

    def info(self):
        res = np.concatenate(self.ref['residual_m'])
        return {'checkpoint': str(D.FLOWMDM_CHECKPOINT.relative_to(D.DATA)), 'epoch': self.ev.epoch, 'segments': len(self.gt['key']),
                'protocol': 'config/evalkit_dist_flowmdm_babel_20261107.json', 'token_coverage_fallback': self.coverage,
                'reference': 'ground-truth joints through the IK', 'reference_ik_residual_mm': {'mean': float(res.mean()*1e3),
                'p99': float(np.percentile(res, 99)*1e3)}, 'native': getattr(self, 'native', None)}

    def ground_truth(self):
        self.text = self.ev.embed_texts(self.tokens)
        return self.ev.embed_motions(self.ref['features']), self.text

    def text_metrics(self, text, motion):
        return D.text_metrics(text, motion, key='evalkit_dist_flowmdm_babel_r_precision')

    def real(self, gt, text):
        return {**self.text_metrics(text, gt), 'diversity': D.diversity(gt, key='evalkit_dist_flowmdm_babel_diversity')}

    def perturbed(self, index, r):
        ev = self.ev
        joints = [self.gt['joints'][i] for i in index]
        sets = {'roundtrip': ev.features_from_joints([self.ref['joints'][i] for i in index])[0]}
        for kind, level, label in levels():
            sets[label] = ev.features_from_joints([perturb(j, kind, level, r).astype(np.float32) for j in joints])[0]
            print(label, flush=True)
        emb = {'original': ev.embed_motions([self.ref['features'][i] for i in index]), **{k: ev.embed_motions(v) for k, v in sets.items()}}
        native = ev.embed_motions(self.gt['features'])
        n = len(native); half = D.rng('evalkit_dist_flowmdm_babel_split_0').permutation(n)
        a, b = half[:n//2], half[n//2:]
        ik = ev.embed_motions(self.ref['features'])
        self.native = {'fid_native_a_vs_ik_a': self.fid(native[a], ik[a]), 'fid_native_a_vs_ik_b': self.fid(native[a], ik[b]),
                       'fid_native_a_vs_native_b': self.fid(native[a], native[b]), 'fid_ik_a_vs_ik_b': self.fid(ik[a], ik[b]),
                       'text_match_native': self.text_metrics(self.text, native), 'diversity_native': D.diversity(native, key='evalkit_dist_flowmdm_babel_diversity'),
                       'scope': 'descriptive: information the joint-only path loses relative to the official rotations'}
        z = [np.abs(f-self.ref['features'][i]).mean() for f, i in zip(sets['roundtrip'], index)]
        return emb, {'feature_abs_z_mean': float(np.mean(z))}

    def smoke(self, rows):
        ev, windows, record, owner = self.ev, [], [], []
        for k, row in enumerate(rows):
            z = np.load(motion_file(row['source']), allow_pickle=True)
            names = [str(n) for n in z['joint_names']]
            j = np.asarray(z['joints'], np.float32)[:, [names.index(n) for n in D.HUMANML_NAMES]]
            feats, fits = ev.features_from_joints([j], fps=float(z['fps']))
            f, fit = feats[0], fits[0]
            starts = [s for s in range(0, len(f), self.WINDOW) if len(f)-s >= ev.MIN_FRAMES]
            for s in starts: windows.append(f[s:s+self.WINDOW]); owner.append(k)
            record.append({'id': row['id'], 'method': row['method'], 'prompt': sources.prompt(row['source']), 'frames_30fps': len(f),
                           'windows': len(starts), 'ik_residual_mm_mean': float(fit['residual_m'].mean()*1e3),
                           'ik_residual_mm_p99': float(np.percentile(fit['residual_m'], 99)*1e3),
                           'bone_lengths_m': ev.bone_lengths(j).round(4).tolist(), 'finite': bool(np.isfinite(f).all())})
        emb = ev.embed_motions(windows)
        for k, rec in enumerate(record):
            rec['window_embedding_norm'] = [float(np.linalg.norm(e)) for e, o in zip(emb, owner) if o == k]
        self.windows = {'owner': np.array(owner)}
        return emb, record


ADAPTERS = {'t2m': T2MAdapter, 'tmr_soma': TMRAdapter, 'omomo': OMOMOAdapter, 'flowmdm_babel': FlowMDMAdapter}


def selfcheck(adapter, out):
    name = adapter.name
    gt, text = adapter.ground_truth()
    n = len(gt)
    halves = [D.rng(f'evalkit_dist_{name}_split_{s}').permutation(n) for s in range(5)]
    split = [adapter.fid(gt[p[:n//2]], gt[p[n//2:]]) for p in halves]
    full = not np.isfinite(split).any()
    a, b = (np.arange(n), np.arange(n)) if full else (halves[0][:n//2], halves[0][n//2:])
    matched = adapter.text_metrics(text, gt)
    shuffled = adapter.text_metrics(text[D.rng(f'evalkit_dist_{name}_text_shuffle').permutation(n)], gt)
    emb, extra = adapter.perturbed(a, D.rng(f'evalkit_dist_{name}_perturb'))
    ref = gt[b]
    curve = {k: adapter.fid(v, ref) for k, v in emb.items()}
    noise = [curve['original']] + [curve[f'noise_{s}'] for s in NOISE_M]
    shuffle = [curve['original']] + [curve[f'shuffle_{p}'] for p in SHUFFLE]
    roundtrip = next(k for k in emb if k.startswith('roundtrip'))
    drift = adapter.fid(emb[roundtrip], emb['original'])
    rn = D.rng(f'evalkit_dist_{name}_n_bias')
    sizes = [N for N in N_LIST if N < len(a)] + ([len(a)] if full else [])
    bias = {lab: {str(N): [adapter.fid(x[rn.choice(len(x), N, replace=False)], ref) for _ in range(REPEATS)] for N in sizes}
            for lab, x in (('ground_truth', emb['original']), ('noise_0.02', emb['noise_0.02']), ('noise_0.05', emb['noise_0.05']))}
    defined = lambda N: all(np.isfinite(bias[k][str(N)]).all() for k in bias)
    min_n = next((N for N in sizes if defined(N) and np.percentile(bias['ground_truth'][str(N)], 95) < np.percentile(bias['noise_0.02'][str(N)], 5)), None)
    nb = {'n': sizes, 'repeats': REPEATS, 'fid': bias, 'min_n_rule': 'p95 ground truth < p5 noise 0.02 m (约定，未经推演)',
          'undefined_n': [N for N in sizes if not defined(N)], 'min_n': min_n}
    if full:
        nb['reference'] = 'full test set (draws are subsets of it, so the ground-truth curve is biased low)'
        reencode = float(np.abs(emb[roundtrip]-emb['original']).max())
        rt = {'kind': roundtrip, 'fid_to_original': drift, 'embedding_max_abs_diff': reencode, **extra, 'pass': reencode == 0.,
              'rule': 'full-set mode: identical embeddings on re-encode (no conversion between the native file and the evaluator)'}
    else:
        rt = {'kind': roundtrip, 'fid_to_reference_half': curve[roundtrip], 'fid_original_half_vs_half': curve['original'],
              'fid_to_original': drift, **extra, 'pass': bool(drift < min(split)),
              'rule': 'FID(round trip, original) below the smallest split-half FID (约定，未经推演; set after the t2m trial)'}
    result = {
        'evaluator': name, **adapter.info(), 'ground_truth': n, 'real': adapter.real(gt, text), 'mode': 'full_set' if full else 'split_half',
        'checks': {
            'split_half_fid': {'values': split, 'max': max(split), 'defined': not full},
            'round_trip': rt,
            'perturbation_monotone': {'fid': curve, 'noise_monotone': bool(np.all(np.diff(noise) > 0)),
                                      'shuffle_monotone': bool(np.all(np.diff(shuffle) > 0)),
                                      'pass': bool(np.all(np.diff(noise) > 0) and np.all(np.diff(shuffle) > 0)),
                                      'undefined_levels': [k for k, v in curve.items() if not np.isfinite(v)],
                                      'monotone_on_defined_levels': bool(all(np.all(np.diff([v for v in c if np.isfinite(v)]) > 0)
                                                                             for c in (noise, shuffle)))},
            'text_match': {'matched': matched, 'shuffled': shuffled, 'chance_batch32': [1/32, 2/32, 3/32],
                           'pass': bool(matched['top3'] > 3*shuffled['top3'])},
            'n_bias': nb},
        'scope': 'ground truth only; ' + ('full-set mode: perturbed sets and the reference are the whole test set' if full
                                          else 'perturbed sets come from half A, the reference is half B')}
    (out/name).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/name/'embeddings.npz', ground_truth=gt, text=text, half_a=a, half_b=b,
                        **{f'perturbed_{k}': v for k, v in emb.items()})
    (out/name/'selfcheck.json').write_text(json.dumps(result, indent=1, ensure_ascii=False, default=float))
    print(json.dumps({k: result['checks'][k].get('pass') for k in result['checks']}
                     | {'min_n': min_n, 'split_max': max(split), 'noise': noise, 'shuffle': shuffle}, default=float))


def smoke(adapter, manifest, out, reference):
    name = adapter.name
    rows = [r for r in json.loads(Path(manifest).read_text())['rows'] if D.REGISTRY[r['method']] == name]
    gt = np.load(reference/name/'embeddings.npz')['ground_truth']
    (out/name).mkdir(parents=True, exist_ok=True)
    emb, record = adapter.smoke(rows)
    owner = getattr(adapter, 'windows', {}).get('owner', np.arange(len(rows)))
    methods = {}
    for m in sorted({r['method'] for r in rows}):
        i = [k for k, r in enumerate(rows) if r['method'] == m]
        e = emb[np.isin(owner, i)]
        methods[m] = {'rows': len(i), 'embeddings': len(e), 'native': D.NATIVE.get(m, True), 'fid_n5_vs_ground_truth': adapter.fid(e, gt),
                      'diversity': float(np.linalg.norm(e[:, None]-e[None], axis=-1)[np.triu_indices(len(e), 1)].mean()),
                      'all_finite': bool(np.isfinite(e).all() and all(record[k]['finite'] for k in i))}
    result = {'evaluator': name, 'rows': record, 'methods': methods,
              'scope': 'smoke: pipeline runs and values are finite; N=5 FID and single-prompt text scores are not quality estimates',
              'diversity_note': 'mean pairwise distance of the smoke embeddings (all pairs; the official 300-pair draw needs more samples)'}
    np.savez_compressed(out/name/'smoke_embeddings.npz', embeddings=emb, ids=np.array([r['id'] for r in rows]), owner=owner)
    (out/name/'smoke.json').write_text(json.dumps(result, indent=1, ensure_ascii=False, default=float))
    print(json.dumps(methods, indent=1, default=float))


def figure(name, out, reference):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d, check = out/name, json.loads((reference/name/'selfcheck.json').read_text())
    emb = np.load(reference/name/'embeddings.npz')
    gt = emb['ground_truth']
    mu = gt.mean(0)
    basis = np.linalg.svd(gt-mu, full_matrices=False)[2][:2].T
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    ax[0, 0].scatter(*((gt-mu)@basis).T, s=3, c='0.75', label=f'ground truth ({len(gt)})')
    for k, c in (('perturbed_noise_0.05', 'tab:orange'), ('perturbed_noise_0.1', 'tab:red')):
        ax[0, 0].scatter(*((emb[k]-mu)@basis).mean(0), marker='X', s=120, c=c, edgecolor='k', label=k.replace('perturbed_', 'GT + ')+' (centroid)')
    if (d/'smoke.json').exists():
        smoke_, se = json.loads((d/'smoke.json').read_text()), np.load(d/'smoke_embeddings.npz')
        owner = se['owner'] if 'owner' in se.files else np.arange(len(smoke_['rows']))
        for m, color in zip(sorted(smoke_['methods']), plt.cm.tab10.colors):
            i = [k for k, r in enumerate(smoke_['rows']) if r['method'] == m]
            native, e = smoke_['methods'][m]['native'], se['embeddings'][np.isin(owner, i)]
            ax[0, 0].scatter(*((e-mu)@basis).T, s=50, color=color, marker='o' if native else '^', edgecolor='k',
                             label=f"{m}{'' if native else ' (non-native)'} C0 x{len(i)}" + (f' ({len(e)} windows)' if len(e) != len(i) else ''))
    ax[0, 0].set_title('Evaluator embedding, PCA fit on ground truth'); ax[0, 0].legend(fontsize=8)
    c = check['checks']['perturbation_monotone']['fid']
    shown = lambda keys: np.clip([c[k] for k in keys], 1e-6, None)
    ax[0, 1].plot([0, *NOISE_M], shown(['original']+[f'noise_{s}' for s in NOISE_M]), 'o-', label='joint noise sigma (m)')
    ax[0, 1].plot([0, *SHUFFLE], shown(['original']+[f'shuffle_{p}' for p in SHUFFLE]), 's-', label='frame shuffle fraction')
    rt = check['checks']['round_trip']
    rt_fid = rt.get('fid_to_reference_half', rt['fid_to_original'])
    ax[0, 1].axhline(max(rt_fid, 1e-6), ls=':', c='k', label=f"{rt['kind']} {rt_fid:.3g}")
    ax[0, 1].set_yscale('log'); ax[0, 1].set_xlabel('perturbation level'); ax[0, 1].set_ylabel('FID vs reference half')
    ax[0, 1].set_title('FID rises with perturbation (level 0 = unperturbed half A)'); ax[0, 1].legend(fontsize=8)
    b = check['checks']['n_bias']
    for lab, col in (('ground_truth', 'k'), ('noise_0.02', 'tab:orange'), ('noise_0.05', 'tab:red')):
        v = np.clip(np.array([b['fid'][lab][str(N)] for N in b['n']]), 1e-6, None)
        ax[1, 0].plot(b['n'], np.nanmedian(v, 1), 'o-', c=col, label=lab)
        ax[1, 0].fill_between(b['n'], np.nanpercentile(v, 5, 1), np.nanpercentile(v, 95, 1), color=col, alpha=.2)
    if b['min_n']: ax[1, 0].axvline(b['min_n'], ls='--', c='g', label=f"min N = {b['min_n']} (convention)")
    if b['undefined_n']: ax[1, 0].plot([], [], ' ', label=f"official FID undefined at N = {b['undefined_n']}")
    ax[1, 0].set_xscale('log'); ax[1, 0].set_yscale('log'); ax[1, 0].set_xlabel('N samples')
    ax[1, 0].set_ylabel('FID vs ' + ('full test set' if 'reference' in b else 'reference half'))
    ax[1, 0].set_title(f"FID bias vs sample size (5-95% band, {b['repeats']} draws)"); ax[1, 0].legend(fontsize=8)
    t = check['checks']['text_match']
    bars = [('matched', [t['matched'][f'top{i}'] for i in (1, 2, 3)]), ('shuffled text', [t['shuffled'][f'top{i}'] for i in (1, 2, 3)])]
    if 'published_real' in check: bars.append(('published real', [check['published_real'][f'top{i}'] for i in (1, 2, 3)]))
    for k, (lab, vals) in enumerate(bars): ax[1, 1].bar(np.arange(3)+k*.27, vals, .27, label=lab)
    ax[1, 1].set_xticks(np.arange(3)+.27, ['top-1', 'top-2', 'top-3'])
    ax[1, 1].set_title(f"Text-motion retrieval ({t['matched'].get('protocol', 'batches of 32')})"); ax[1, 1].legend(fontsize=8)
    fig.suptitle(f'{name}: ground-truth self-checks + C0 smoke (scores from different evaluators are not comparable)')
    fig.tight_layout(); fig.savefig(d/'page.png', dpi=110)
    print(d/'page.png')


def reexec_kimodo():
    from motiogen.config import environment_root, load_config, repo_root
    from motiogen.stratified import _environment
    cfg = load_config()
    # ~/.local carries huggingface-hub 2.x, which the Kimodo transformers pin rejects.
    env = _environment(cfg) | {'EVALKIT_KIMODO': '1', 'PYTHONNOUSERSITE': '1'}
    os.chdir(repo_root(cfg, 'kimodo'))
    python = str(environment_root(cfg, 'kimodo')/'bin/python')
    os.execve(python, [python, __file__, *sys.argv[1:]], env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['selfcheck', 'smoke', 'figure'])
    ap.add_argument('--evaluator', default='t2m', choices=sorted(ADAPTERS))
    ap.add_argument('--manifest', type=Path, default=MANIFEST)
    ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('--reference', type=Path, default=OUT, help='directory holding <evaluator>/selfcheck.json and embeddings.npz')
    a = ap.parse_args()
    a.out, a.reference = a.out.resolve(), a.reference.resolve()
    if a.step == 'figure': return figure(a.evaluator, a.out, a.reference)
    if a.step == 'smoke' and not any(D.REGISTRY[r['method']] == a.evaluator for r in json.loads(a.manifest.read_text())['rows']):
        return print(json.dumps({'evaluator': a.evaluator, 'rows': 0, 'skipped': 'no manifest row maps to this evaluator'}))
    if a.evaluator == 'tmr_soma' and not os.environ.get('EVALKIT_KIMODO'): return reexec_kimodo()
    adapter = ADAPTERS[a.evaluator]()
    selfcheck(adapter, a.out) if a.step == 'selfcheck' else smoke(adapter, a.manifest.resolve(), a.out, a.reference)


if __name__ == '__main__':
    main()
