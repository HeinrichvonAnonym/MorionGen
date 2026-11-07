"""Distribution evaluators: method registry, native embeddings, official metrics, ground-truth self-checks.

Scores from different evaluators are not comparable. Protocol: config/evalkit_v1_20261107.json `distribution`.
Runs in a torch environment (motiogen-motionlcm); upstream code is imported from the pinned baseline trees.
"""
import sys
import zlib
from pathlib import Path

import numpy as np

from motiogen.config import data_root, load_config
from motiogen.motion import HUMANML_NAMES, native_positions

DATA = data_root(load_config())
ASSETS, BASELINES = DATA/'outputs/assets', DATA/'baselines'
REGISTRY = {'motionlcm': 't2m', 'omnicontrol': 't2m', 'closd': 't2m', 'ardy': 't2m', 'primal': 't2m',
            'kimodo': 'tmr_soma', 'chois': 'omomo', 'dart': 'flowmdm_babel'}
NATIVE = {'ardy': False, 'primal': False}
T2M_CHECKPOINTS = {'closd': ASSETS/'closd-official/evaluators/t2m/text_mot_match/model/finest.tar',
                   'motionlcm': ASSETS/'motionlcm/t2m/t2m/text_mot_match/model/finest.tar'}
T2M_GT = ASSETS/'closd-official/data/humanml3d/t2m_test.npy'
T2M_NORM = ASSETS/'motiongpt-official/t2m/t2m/Comp_v6_KLD01/meta'
PUBLISHED_REAL = {'top1': .511, 'top2': .703, 'top3': .797, 'matching': 2.974, 'diversity': 9.503}
# cskel27 (ARDY) has no SMPL spine3 or collars; nearest joints stand in (non-native mapping).
SMPL22_ALIASES = {'spine3': 'Spine3', 'left_collar': 'LeftShoulder', 'right_collar': 'RightShoulder'}
SPACY_PYTHON = DATA/'envs/motiogen-spacy/bin/python'
TAGGER = '''import json, sys, spacy
nlp = spacy.load("en_core_web_sm")
out = []
for s in json.load(sys.stdin):
    out.append(["%s/%s" % (t.lemma_ if t.pos_ in ("NOUN", "VERB") and t.text != "left" else t.text, t.pos_)
                for t in nlp(s.replace("-", "").lower()) if t.text.isalpha()])
print(json.dumps(out))'''


def rng(key):
    return np.random.default_rng(zlib.crc32(key.encode()))


def tag(texts):
    """HumanML3D caption tokens (text_process.py rule) via spaCy en_core_web_sm in its own environment."""
    import json, subprocess
    run = subprocess.run([str(SPACY_PYTHON), '-c', TAGGER], input=json.dumps(list(texts)), capture_output=True, text=True, check=True)
    return json.loads(run.stdout)


def tag_from_ground_truth(texts, unknown=None):
    """Fallback without spaCy: each word with its most frequent tag in the official test-caption tokens (lemmas of
    nouns / verbs by the same lookup); a word never seen there makes the caption None, or gets word/<unknown> if given."""
    import collections
    data = np.load(T2M_GT, allow_pickle=True).item()
    seen = collections.defaultdict(collections.Counter)
    for v in data['data_dict'].values():
        for t in v['text']:
            words = [w for w in t['caption'].replace('-', '').lower().split()]
            for w, tok in zip([w.strip('.,!?;:') for w in words if w.strip('.,!?;:').isalpha()], t['tokens']):
                seen[w][tok] += 1
    out = []
    for s in texts:
        words = [w.strip('.,!?;:') for w in s.replace('-', '').lower().split()]
        toks = [seen[w].most_common(1)[0][0] if seen[w] else (f'{w}/{unknown}' if unknown else None) for w in words if w.isalpha()]
        out.append(None if None in toks else toks)
    return out


def fid(a, b):
    from data_loaders.humanml.utils.metrics import calculate_activation_statistics, calculate_frechet_distance
    return float(calculate_frechet_distance(*calculate_activation_statistics(a), *calculate_activation_statistics(b)))


def text_metrics(text, motion, batch=32, repeats=20, key='evalkit_dist_r_precision'):
    """Official HumanML3D R-precision (top-1..3) and matching score: shuffled batches of 32 (drop last), mean of repeats."""
    from data_loaders.humanml.utils.metrics import calculate_R_precision, euclidean_distance_matrix
    r, n = rng(key), len(text)//batch*batch
    top, match = np.zeros(3), 0.
    for _ in range(repeats):
        order = r.permutation(len(text))[:n]
        for i in range(0, n, batch):
            t, m = text[order[i:i+batch]], motion[order[i:i+batch]]
            top += calculate_R_precision(t, m, top_k=3, sum_all=True)
            match += np.trace(euclidean_distance_matrix(t, m))
    n_all = n*repeats
    return {'top1': top[0]/n_all, 'top2': top[1]/n_all, 'top3': top[2]/n_all, 'matching': match/n_all, 'samples': n}


def diversity(embeddings, times=300, key='diversity'):
    r = rng(key)
    i, j = r.choice(len(embeddings), times, replace=False), r.choice(len(embeddings), times, replace=False)
    return float(np.linalg.norm(embeddings[i]-embeddings[j], axis=1).mean())


class T2M:
    """HumanML3D text-motion-matching evaluator (Guo et al. 2022) with the official test-time protocol."""
    UNIT, MAX_FRAMES, MAX_TOKENS, FPS = 4, 196, 20, 20.

    def __init__(self, checkpoint='motionlcm', device=None):
        sys.path.insert(0, str(BASELINES/'OmniControl'))
        import torch
        from data_loaders.humanml.networks.modules import MotionEncoderBiGRUCo, MovementConvEncoder, TextEncoderBiGRUCo
        from data_loaders.humanml.utils.word_vectorizer import POS_enumerator, WordVectorizer
        import data_loaders.humanml.scripts.motion_process as process
        self.torch, self.process = torch, process
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint = T2M_CHECKPOINTS[checkpoint]
        state = torch.load(self.checkpoint, map_location='cpu')
        self.text_net = TextEncoderBiGRUCo(300, len(POS_enumerator), 512, 512, self.device)
        self.movement_net = MovementConvEncoder(259, 512, 512)
        self.motion_net = MotionEncoderBiGRUCo(512, 1024, 512, self.device)
        for net, key in ((self.text_net, 'text_encoder'), (self.movement_net, 'movement_encoder'), (self.motion_net, 'motion_encoder')):
            net.load_state_dict(state[key]); net.to(self.device).eval()
        self.epoch = int(state['epoch'])
        self.mean, self.std = np.load(T2M_NORM/'mean.npy'), np.load(T2M_NORM/'std.npy')
        self.words = WordVectorizer(str(ASSETS/'motionlcm/glove'), 'our_vab')
        self._offsets = None

    def ground_truth(self, key='evalkit_dist_t2m'):
        """Test motions (raw 263-d), one caption per motion drawn with the seed."""
        data = np.load(T2M_GT, allow_pickle=True).item()
        r = rng(key)
        names = list(data['name_list'])
        motions = [data['data_dict'][n]['motion'] for n in names]
        tokens = [data['data_dict'][n]['text'][r.integers(len(data['data_dict'][n]['text']))]['tokens'] for n in names]
        return names, motions, tokens

    def crop(self, features, r):
        n = len(features)//self.UNIT*self.UNIT
        if r.random() < 1/3: n -= self.UNIT
        n = max(min(n, self.MAX_FRAMES), self.UNIT)
        start = r.integers(0, len(features)-n+1)
        return features[start:start+n]

    def embed_motions(self, motions, key='evalkit_dist_t2m_crop', batch=256):
        torch, r, out = self.torch, rng(key), []
        for i in range(0, len(motions), batch):
            chunk = [self.crop(m, r) for m in motions[i:i+batch]]
            lens = np.array([len(m) for m in chunk])
            x = np.zeros((len(chunk), self.MAX_FRAMES, 263), np.float32)
            for k, m in enumerate(chunk): x[k, :len(m)] = (m-self.mean)/self.std
            order = np.argsort(-lens, kind='stable')
            with torch.no_grad():
                moved = self.movement_net(torch.from_numpy(x[order]).to(self.device)[..., :-4])
                em = self.motion_net(moved, torch.from_numpy(lens[order]//self.UNIT)).cpu().numpy()
            out.append(em[np.argsort(order)])
        return np.concatenate(out)

    def embed_texts(self, token_lists, batch=256):
        torch, out = self.torch, []
        for i in range(0, len(token_lists), batch):
            words, pos, lens = [], [], []
            for tokens in token_lists[i:i+batch]:
                t = ['sos/OTHER'] + list(tokens)[:self.MAX_TOKENS] + ['eos/OTHER']
                lens.append(len(t)); t += ['unk/OTHER']*(self.MAX_TOKENS+2-len(t))
                w, p = zip(*(self.words[x] for x in t)); words.append(w); pos.append(p)
            lens = np.array(lens); order = np.argsort(-lens, kind='stable')
            with torch.no_grad():
                em = self.text_net(torch.tensor(np.array(words)[order], dtype=torch.float32, device=self.device),
                                   torch.tensor(np.array(pos)[order], dtype=torch.float32, device=self.device),
                                   torch.from_numpy(lens[order])).cpu().numpy()
            out.append(em[np.argsort(order)])
        return np.concatenate(out)

    def joints(self, features):
        """Raw 263-d features -> HumanML3D Y-up joint positions at 20 fps."""
        torch = self.torch
        return self.process.recover_from_ric(torch.from_numpy(np.asarray(features, np.float32)), 22).numpy()

    def offsets(self, reference_features):
        if self._offsets is None:
            from data_loaders.humanml.common.skeleton import Skeleton
            from data_loaders.humanml.utils.paramUtil import t2m_kinematic_chain, t2m_raw_offsets
            p = self.process
            p.n_raw_offsets, p.kinematic_chain = self.torch.from_numpy(t2m_raw_offsets), t2m_kinematic_chain
            p.l_idx1, p.l_idx2, p.fid_r, p.fid_l, p.face_joint_indx = 5, 8, [8, 11], [7, 10], [2, 1, 17, 16]
            skeleton = Skeleton(p.n_raw_offsets, p.kinematic_chain, 'cpu')
            p.tgt_offsets = self._offsets = skeleton.get_offsets_joints(self.torch.from_numpy(self.joints(reference_features)[0]))
        return self._offsets

    def features(self, joints_yup):
        """HumanML3D Y-up 20 fps joints (T, 22, 3) -> raw 263-d features (T-1 frames), upstream process_file."""
        if self._offsets is None: raise RuntimeError('call offsets() with a ground-truth motion first')
        return self.process.process_file(np.asarray(joints_yup, np.float64), 0.002)[0]

    @classmethod
    def joints_from_motion(cls, joints, names, fps):
        """Canonical Z-up joints -> HumanML3D Y-up SMPL-22 at 20 fps (linear resampling)."""
        names = [str(n) for n in names]
        index = [names.index(n if n in names else SMPL22_ALIASES[n]) for n in HUMANML_NAMES]
        x = native_positions(np.asarray(joints)[:, index], 'Y-up')
        t = np.arange(len(x))/float(fps)
        u = np.arange(0, t[-1]+1e-9, 1/cls.FPS)
        return np.stack([np.stack([np.interp(u, t, x[:, j, k]) for k in range(3)], -1) for j in range(22)], 1)


CHOIS_EVAL = BASELINES/'chois_release/t2m_eval'
CHOIS_RELATED = ASSETS/'chois/evaluators/chois_eval_related'
CHOIS_CHECKPOINT = ASSETS/'chois/evaluators/checkpoints/omomo/text_motion_features/model/finest.tar'


class OMOMO:
    """CHOIS text-motion evaluator trained on OMOMO (t2m_eval/final_evaluations.py): 24 global joint positions
    (CHOIS res_npz global_jpos) normalised by t2m_mean_std_jpos, captions of the sequence annotation."""
    UNIT, MAX_TOKENS = 4, 30

    def __init__(self, device=None):
        sys.path.insert(0, str(CHOIS_EVAL))
        import joblib
        import torch
        from networks.modules import MotionEncoderBiGRUCo, MovementConvEncoder, TextEncoderBiGRUCo
        from utils import metrics
        from utils.word_vectorizer import POS_enumerator, WordVectorizer
        self.torch, self.metrics = torch, metrics
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        state = torch.load(CHOIS_CHECKPOINT, map_location='cpu', weights_only=False)
        self.text_net = TextEncoderBiGRUCo(300, len(POS_enumerator), 512, 512, self.device)
        self.movement_net = MovementConvEncoder(72, 512, 512)
        self.motion_net = MotionEncoderBiGRUCo(512, 1024, 512, self.device)
        for net, key in ((self.text_net, 'text_encoder'), (self.movement_net, 'movement_encoder'), (self.motion_net, 'motion_encoder')):
            net.load_state_dict(state[key]); net.to(self.device).eval()
        self.epoch = int(state['epoch'])
        norm = joblib.load(CHOIS_RELATED/'t2m_mean_std_jpos.p')
        self.mean, self.std = np.asarray(norm['jpos_mean'], np.float32), np.asarray(norm['jpos_std'], np.float32)
        self.words = WordVectorizer(str(CHOIS_RELATED/'glove_840B'), 'our_vab')

    @staticmethod
    def load(path):
        z = np.load(path)
        return np.asarray(z['global_jpos'], np.float32), str(z['seq_name'])

    @staticmethod
    def tokens(seq_name):
        """Caption tokens of the sequence (last line of its annotation file, as the official loader)."""
        lines = (CHOIS_RELATED/'omomo_text_anno_txt_data'/f'{seq_name}.txt').read_text().strip().splitlines()
        return lines[-1].split('#')[1].split(' ')

    def ground_truth(self):
        files = sorted((CHOIS_RELATED/'res_npz_files/gt').glob('*.npz'))
        return files, [self.load(f)[0] for f in files], [self.tokens(self.load(f)[1]) for f in files]

    def embed_motions(self, jpos, batch=256):
        torch, out = self.torch, []
        for i in range(0, len(jpos), batch):
            chunk = jpos[i:i+batch]; lens = np.array([len(j) for j in chunk])
            x = np.zeros((len(chunk), lens.max(), 72), np.float32)
            for k, j in enumerate(chunk): x[k, :len(j)] = (np.reshape(j, (len(j), 72))-self.mean)/self.std
            order = np.argsort(-lens, kind='stable')
            with torch.no_grad():
                moved = self.movement_net(torch.from_numpy(x[order]).to(self.device))
                em = self.motion_net(moved, torch.from_numpy(lens[order]//self.UNIT)).cpu().numpy()
            out.append(em[np.argsort(order)])
        return np.concatenate(out)

    def embed_texts(self, token_lists, batch=256):
        torch, out = self.torch, []
        for i in range(0, len(token_lists), batch):
            words, pos, lens = [], [], []
            for tokens in token_lists[i:i+batch]:
                t = ['sos/OTHER'] + list(tokens)[:self.MAX_TOKENS] + ['eos/OTHER']
                lens.append(len(t)); t += ['unk/OTHER']*(self.MAX_TOKENS+2-len(t))
                w, p = zip(*(self.words[x] for x in t)); words.append(w); pos.append(p)
            lens = np.array(lens); order = np.argsort(-lens, kind='stable')
            with torch.no_grad():
                em = self.text_net(torch.tensor(np.array(words)[order], dtype=torch.float32, device=self.device),
                                   torch.tensor(np.array(pos)[order], dtype=torch.float32, device=self.device),
                                   torch.from_numpy(lens[order])).cpu().numpy()
            out.append(em[np.argsort(order)])
        return np.concatenate(out)

    def fid(self, a, b):
        m = self.metrics
        try:
            return float(m.calculate_frechet_distance(*m.calculate_activation_statistics(a), *m.calculate_activation_statistics(b)))
        except ValueError:  # official sqrtm check on a singular small-N covariance product
            return float('nan')

    def text_metrics(self, text, motion, batch=32, repeats=20, key='evalkit_dist_omomo_r_precision'):
        """Official matching score and R-precision (batches of 32, drop last) on seeded shuffled batches, mean of repeats."""
        m, r, n = self.metrics, rng(key), len(text)//batch*batch
        top, match = np.zeros(3), 0.
        for _ in range(repeats):
            order = r.permutation(len(text))[:n]
            for i in range(0, n, batch):
                d = m.euclidean_distance_matrix(text[order[i:i+batch]], motion[order[i:i+batch]])
                match += np.trace(d); top += m.calculate_top_k(np.argsort(d, axis=1), top_k=3).sum(0)
        n_all = n*repeats
        return {'top1': top[0]/n_all, 'top2': top[1]/n_all, 'top3': top[2]/n_all, 'matching': match/n_all, 'samples': n}


FLOWMDM = BASELINES/'DART/FlowMDM'
FLOWMDM_ASSETS = ASSETS/'dart-official/flowmdm'
FLOWMDM_CHECKPOINT = FLOWMDM_ASSETS/'text_mot_match_babel_random_motion_bs64/model/finest.tar'
FLOWMDM_CACHE = DATA/'outputs/evalkit_20261107/distribution/flowmdm_babel_val_segments.pkl'
SMPLH_MALE = ASSETS/'chois/processed_data/smpl_all_models/smplh_amass/male/model.npz'
SMPL22_PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)


class FlowMDMBabel:
    """FlowMDM BABEL text-motion evaluator (DART's vendored FlowMDM): 135-d Globalvelandy features (root height, canonical
    root xy velocity, 22 SMPL rotations in 6d, babel-amass normalisation) at 30 fps; ground truth = BABEL_SingleEval val."""
    UNIT, MIN_FRAMES, MAX_FRAMES, MAX_TOKENS, FPS = 4, 15, 200, 20, 30.

    def __init__(self, device=None):
        import os
        sys.path.insert(0, str(FLOWMDM))
        import torch
        from data_loaders.amass.transforms.rots2rfeats import Globalvelandy
        from data_loaders.humanml.networks.modules import MotionEncoderBiGRUCo, MovementConvEncoder, TextEncoderBiGRUCo
        from data_loaders.humanml.utils.word_vectorizer import POS_enumerator, WordVectorizer
        self.torch = torch
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        cwd = os.getcwd()
        os.chdir(FLOWMDM)  # upstream loads the normalisation statistics from a path relative to the repository
        try:
            self.transform = Globalvelandy(path='data_loaders/amass/deps', normalization=True, pose_rep='rot6d', canonicalize=True, offset=True)
        finally:
            os.chdir(cwd)
        state = torch.load(FLOWMDM_CHECKPOINT, map_location='cpu')
        self.text_net = TextEncoderBiGRUCo(300, len(POS_enumerator), 512, 512, self.device)
        self.movement_net = MovementConvEncoder(135, 512, 512)
        self.motion_net = MotionEncoderBiGRUCo(512, 1024, 512, self.device)
        for net, key in ((self.text_net, 'text_encoder'), (self.movement_net, 'movement_encoder'), (self.motion_net, 'motion_encoder')):
            net.load_state_dict(state[key]); net.to(self.device).eval()
        self.epoch = int(state['epoch'])
        self.words = WordVectorizer(str(ASSETS/'motionlcm/glove'), 'our_vab')
        body = np.load(SMPLH_MALE, allow_pickle=True)
        self.rest = (np.asarray(body['J_regressor'])[:22] @ np.asarray(body['v_template'])).astype(np.float32)
        offsets = self.rest[1:] - self.rest[list(SMPL22_PARENTS[1:])]
        self.directions = np.r_[np.zeros((1, 3)), offsets/np.linalg.norm(offsets, axis=1, keepdims=True)].astype(np.float32)

    def features(self, rots, trans):
        """Rotation matrices (T, 22, 3, 3) + root translation (T, 3), Z-up -> normalised 135-d features (upstream transform)."""
        from data_loaders.amass.transforms.smpl import RotTransDatastruct
        torch = self.torch
        data = RotTransDatastruct(rots=torch.as_tensor(np.asarray(rots), dtype=torch.float32), trans=torch.as_tensor(np.asarray(trans), dtype=torch.float32))
        return self.transform(data).numpy()

    def ground_truth(self):
        """BABEL val segments as BABEL_SingleEval (load_mode gt, sampler 15..200, no cropping): frame annotations only,
        transitions excluded, proc_label with fix_spell / standarize_text; features from the SMPL-H poses, joints for the round trip."""
        import pickle
        if FLOWMDM_CACHE.exists():
            return pickle.loads(FLOWMDM_CACHE.read_bytes())
        import json
        import joblib
        from data_loaders.amass.nlp_consts import fix_spell
        from data_loaders.amass.tools.smpl import smpl_data_to_matrix_and_trans
        torch = self.torch
        annots = json.loads((FLOWMDM_ASSETS/'babel-teach/val.json').read_text())
        out = {'key': [], 'text': [], 'features': [], 'joints': []}
        for sample in joblib.load(FLOWMDM_ASSETS/'babel-smplh-30fps-male/val.pth.tar'):
            labels, n = annots[sample['babel_id']], len(sample['poses'])
            if labels['frame_ann'] is None:
                continue
            for index, seg in enumerate(labels['frame_ann']['labels']):
                action = fix_spell(seg['proc_label'])
                start, end = int(seg['start_t']*sample['fps']), min(n, int(seg['end_t']*sample['fps']))
                if action == 'transition' or not self.MIN_FRAMES <= end-start <= self.MAX_FRAMES:
                    continue
                rt = smpl_data_to_matrix_and_trans({'poses': torch.from_numpy(sample['poses'][start:end]).float(),
                                                    'trans': torch.from_numpy(sample['trans'][start:end]).float()}, nohands=True)
                out['key'].append(f"{sample['babel_id']}-{index}")
                out['text'].append({'t-pose': 't-pose', 't pose': 't-pose', 'tpose': 't-pose', 'a-pose': 'a-pose', 'a pose': 'a-pose',
                                    'apose': 'a-pose'}.get(action, action))
                out['features'].append(self.transform(rt).numpy())
                out['joints'].append(np.asarray(sample['joint_positions'][start:end, :22], np.float32))
        FLOWMDM_CACHE.parent.mkdir(parents=True, exist_ok=True)
        FLOWMDM_CACHE.write_bytes(pickle.dumps(out, protocol=4))
        return out

    def ik_reference(self):
        """Ground-truth segment joints through the same IK as the generated rows (the scoring reference)."""
        import pickle
        path = FLOWMDM_CACHE.with_name('flowmdm_babel_val_ik.pkl')
        if path.exists():
            return pickle.loads(path.read_bytes())
        feats, fits = self.features_from_joints(self.ground_truth()['joints'])
        out = {'features': feats, 'joints': [f['joints'] for f in fits], 'residual_m': [f['residual_m'] for f in fits]}
        path.write_bytes(pickle.dumps(out, protocol=4))
        return out

    def embed_motions(self, features, batch=256):
        """Whole segments (no crop, as BABEL_SingleEval), zero-padded to 200 frames, lengths // 4 into the motion GRU."""
        torch, out = self.torch, []
        for i in range(0, len(features), batch):
            chunk = features[i:i+batch]
            lens = np.array([len(m) for m in chunk])
            x = np.zeros((len(chunk), self.MAX_FRAMES, 135), np.float32)
            for k, m in enumerate(chunk): x[k, :len(m)] = m
            order = np.argsort(-lens, kind='stable')
            with torch.no_grad():
                moved = self.movement_net(torch.from_numpy(x[order]).to(self.device))
                em = self.motion_net(moved, torch.from_numpy(lens[order]//self.UNIT)).cpu().numpy()
            out.append(em[np.argsort(order)])
        return np.concatenate(out)

    embed_texts = T2M.embed_texts

    def ik(self, joints, iterations=500, lr=.02, w_rest=1e-3, w_smooth=1e-2, chunk_frames=60000):
        """SMPL-22 joint positions (list of (T, 22, 3), Z-up, m) -> SMPL-H rotations and root translation.

        Skeleton: SMPL-H male (AMASS, the evaluator's training body) rest bone directions, bone lengths of each sequence
        (median over frames), so any SMPL-topology skeleton is reachable. Forward kinematics on joints only (joint
        locations do not depend on pose blend shapes); translation = pelvis - zero-beta rest root. Adam on 6d rotations
        minimises squared joint error (cm^2) + w_rest ||R - I||^2 + w_smooth ||R_t - R_t-1||^2 per joint (weights
        约定，未经推演). Twist and the rotations of joints without a child among the 22 (head, feet, wrists) are set by
        the rest and smoothness terms only.
        """
        torch = self.torch
        rest = torch.from_numpy(self.rest).to(self.device)
        directions = torch.from_numpy(self.directions).to(self.device)
        out, start = [], 0
        while start < len(joints):
            stop, frames = start, 0
            while stop < len(joints) and (frames == 0 or frames+len(joints[stop]) <= chunk_frames):
                frames += len(joints[stop]); stop += 1
            seqs = joints[start:stop]
            target = torch.from_numpy(np.concatenate(seqs).astype(np.float32)).to(self.device)
            same = torch.from_numpy(np.concatenate([np.r_[False, np.ones(len(s)-1, bool)] for s in seqs])).to(self.device)
            lengths = np.concatenate([np.repeat(self.bone_lengths(s)[None], len(s), 0) for s in seqs])
            offsets = directions[None]*torch.from_numpy(lengths.astype(np.float32)).to(self.device)[..., None]
            d6 = torch.tensor([1., 0, 0, 0, 1, 0], device=self.device).repeat(len(target), 22, 1)
            d6[:, 0] = self._root_init(target, rest)
            d6.requires_grad_(True)
            opt = torch.optim.Adam([d6], lr=lr)
            eye = torch.eye(3, device=self.device)
            for _ in range(iterations):
                R = self._rot6d(d6)
                err = ((self._fk(R, target[:, 0], offsets)-target)*100).pow(2).sum(-1).mean()
                reg = (R-eye).pow(2).sum((-1, -2)).mean()
                smooth = (R[1:]-R[:-1]).pow(2).sum((-1, -2))[same[1:]].mean() if same.any() else 0.
                loss = err + w_rest*reg + w_smooth*smooth
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                R = self._rot6d(d6)
                fitted = self._fk(R, target[:, 0], offsets)
                residual = (fitted-target).norm(dim=-1).cpu().numpy()
                trans = (target[:, 0]-rest[0]).cpu().numpy()
            R, fitted, cut = R.cpu().numpy(), fitted.cpu().numpy(), np.cumsum([len(s) for s in seqs])[:-1]
            out += [{'rots': r, 'trans': t, 'joints': p, 'residual_m': e}
                    for r, t, p, e in zip(np.split(R, cut), np.split(trans, cut), np.split(fitted, cut), np.split(residual, cut))]
            start = stop
        return out

    @staticmethod
    def bone_lengths(joints):
        j = np.asarray(joints)
        return np.r_[0., np.median(np.linalg.norm(j[:, 1:]-j[:, list(SMPL22_PARENTS[1:])], axis=-1), 0)]

    def _rot6d(self, d6):
        F = self.torch.nn.functional
        b1 = F.normalize(d6[..., :3], dim=-1)
        b2 = F.normalize(d6[..., 3:]-(b1*d6[..., 3:]).sum(-1, keepdim=True)*b1, dim=-1)
        return self.torch.stack((b1, b2, self.torch.cross(b1, b2, dim=-1)), -1)

    def _fk(self, R, root, offsets):
        G, P = [R[:, 0]], [root]
        for j in range(1, 22):
            p = SMPL22_PARENTS[j]
            P.append(P[p] + (G[p] @ offsets[:, j, :, None])[..., 0])
            G.append(G[p] @ R[:, j])
        return self.torch.stack(P, 1)

    def _root_init(self, target, rest):
        """Kabsch of the rest pelvis children (hips, spine1) onto the target, as a 6d rotation (columns 1-2)."""
        torch = self.torch
        a = (rest[[1, 2, 3]]-rest[0])[None]
        b = target[:, [1, 2, 3]]-target[:, [0]]
        u, _, vt = torch.linalg.svd(b.transpose(1, 2) @ a)
        s = torch.ones(len(b), 3, device=b.device); s[:, 2] = torch.sign(torch.det(u @ vt))
        R = u @ torch.diag_embed(s) @ vt
        return torch.cat((R[..., 0], R[..., 1]), -1)

    def features_from_joints(self, joints, fps=30., **ik):
        """Joint positions -> 30 fps (linear) -> IK -> upstream features; returns features and the IK records."""
        resampled = []
        for j in joints:
            j = np.asarray(j, np.float32)
            if abs(fps-self.FPS) > 1e-6:
                t = np.arange(len(j))/fps; u = np.arange(0, t[-1]+1e-9, 1/self.FPS)
                j = np.stack([np.stack([np.interp(u, t, j[:, k, c]) for c in range(3)], -1) for k in range(22)], 1).astype(np.float32)
            resampled.append(j)
        fits = self.ik(resampled, **ik)
        return [self.features(f['rots'], f['trans']) for f in fits], fits


KIMODO_CASES = DATA/'benchmarks/kimodo-stratified-20261008/generated_raw'


class TMRSoma:
    """Kimodo TMR-SOMA motion/text embeddings (unit vectors), upstream full-set retrieval and FID.

    Needs the Kimodo environment (stratified._environment) and its repository as working directory.
    """
    MODEL = 'TMR-SOMA-RP-v1'

    def __init__(self, device=None):
        import importlib
        import torch
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'python/scripts'))
        from kimodo_embed_instrumented import FrozenEncoderModule
        from kimodo.metrics import tmr
        from motiogen.kimodo_compat import load_tmr_with_frozen_raw_encoder
        from motiogen.local_llm import precompute_encoder
        self.torch, self.tmr = torch, tmr
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        frozen = FrozenEncoderModule(precompute_encoder(load_config(), 'kimodo', []))
        self.model = load_tmr_with_frozen_raw_encoder(importlib.import_module('kimodo.model.load_model'), self.MODEL, self.device, frozen)

    @staticmethod
    def cases():
        return sorted(p.parent for p in KIMODO_CASES.rglob('gt_motion.npz'))

    @staticmethod
    def cached(dirs, name):
        return np.stack([np.load(Path(d)/name).reshape(-1) for d in dirs])

    @staticmethod
    def posed(path):
        p = np.load(path)['posed_joints']
        return p[0] if p.ndim == 4 else p

    def embed(self, joints):
        torch, out = self.torch, []
        with torch.inference_mode():
            for j in joints:
                out.append(self.model.encode_motion(torch.from_numpy(np.asarray(j, np.float32)).to(self.device), unit_vector=True).cpu().numpy().reshape(-1))
        return np.stack(out)

    def fid(self, a, b):
        t = self.tmr
        return float(t.calculate_frechet_distance(*t.calculate_activation_statistics(a), *t.calculate_activation_statistics(b)))

    def retrieval(self, text, motion):
        """Upstream TMR_EmbeddingMetric text-to-motion retrieval over the whole set (texts within 0.99 similarity tie)."""
        t = self.tmr
        m = t.contrastive_metrics(scores=t.get_score_matrix_unit(text, motion), scores_t2t=t.get_score_matrix_unit(text, text),
                                  threshold=0.99, rounding=None)
        return {'top1': m['R01']/100, 'top2': m['R02']/100, 'top3': m['R03']/100, 'R05': m['R05']/100, 'R10': m['R10']/100,
                'MedR': m['MedR'], 'samples': len(text), 'protocol': 'whole set, upstream TMR retrieval'}
