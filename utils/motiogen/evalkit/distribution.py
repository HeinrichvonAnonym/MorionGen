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
