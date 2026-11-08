"""Write or check the section-14 freeze manifest config/freeze_20261109.json.

write: sha256 of the four frozen files and of every config they reference (followed transitively), the commit of the
generation and evaluation code (must be unmodified at that commit) and the HEAD of every upstream repository.
--check: recompute all of it against the manifest; exit 1 on any difference.
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'utils'))
from motiogen.config import data_root, load_config  # noqa: E402

OUT = ROOT/'config/freeze_20261109.json'
TAG = 'freeze-20261109'
FOUR = ['config/motion_output_contract_20261108.json', 'config/evalkit_batch_protocol_20261108.json',
        'config/analysis_plan_20261108.json', 'config/inference_registration_20261109.json']
USER_LIST = ['config/inference_tasks_20261109.json', 'config/evalkit_v12_20261108.json', 'config/evalkit_offset_reference_20261108.json',
             'config/evalkit_neck_supp_20261108.json', 'config/evalkit_rula_supp_20261108.json', 'config/evalkit_upper_muscle_20261108.json',
             'config/chois_object_mass_lookup_20261109.json']
CODE = {'generation': ['utils/motiogen', ':(exclude)utils/motiogen/evalkit', 'python/scripts/experiments.py',
                       'python/scripts/baseline_worker.py', 'python/scripts/closd_native.py', 'python/scripts/dart_feedback_smoke.py',
                       'python/scripts/primal_feedback_smoke.py', 'python/scripts/build_inference_smoke.py',
                       'config/baselines.json', 'config/scenarios'],
        'evaluation': ['utils/motiogen/evalkit', 'python/scripts/evalkit_batch.py', 'python/scripts/evalkit_distribution.py',
                       'scripts/run_evalkit_batch.sh']}
REF = re.compile(r'config/[A-Za-z0-9_./-]+\.(?:json|yaml|txt)')


def git(*args, cwd=ROOT):
    return subprocess.check_output(['git', *args], cwd=cwd, text=True).strip()


def sha(path):
    return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()


def referenced():
    """Every config reachable from the four files, the user's list and config/motion_world/*.json."""
    world = sorted(str(p.relative_to(ROOT)) for p in (ROOT/'config/motion_world').glob('*.json'))
    found, todo = {}, [(f, 'frozen file') for f in FOUR] + [(f, 'user list') for f in USER_LIST + world]
    while todo:
        path, by = todo.pop(0)
        if path in found:
            continue
        found[path] = by
        todo += [(m, path) for m in REF.findall((ROOT/path).read_text()) if (ROOT/m).is_file() and m not in found]
    manifest = str(OUT.relative_to(ROOT))
    return {p: {'sha256': sha(p), 'referenced_by': by} for p, by in sorted(found.items()) if p not in FOUR and p != manifest}


def upstream():
    root = data_root(load_config())/'baselines'
    out = {}
    for repo in sorted(p for p in root.iterdir() if (p/'.git').exists()):
        out[f'baselines/{repo.name}'] = {'head': git('rev-parse', 'HEAD', cwd=repo),
                                         'tracked_changes': len(git('status', '--porcelain', '--untracked-files=no', cwd=repo).splitlines())}
    return out


def code(commit):
    areas = {}
    for name, paths in CODE.items():
        dirty = git('status', '--porcelain', '--', *paths)
        changed = git('diff', '--name-only', commit, '--', *paths)
        if dirty or changed:
            raise SystemExit(f'{name} code differs from {commit}:\n{dirty}\n{changed}')
        areas[name] = {'paths': paths, 'last_commit': git('log', '-1', '--format=%H', commit, '--', *paths)}
    return areas


def build():
    commit = git('rev-parse', 'HEAD')
    return {
        'version': 'freeze-1', 'created': time.strftime('%Y-%m-%d %H:%M'), 'git_tag': TAG,
        'user_approval': '2026-11-09 04:08: item 4 approved, three pre-freeze changes, then the freeze (design section 14, item 5)',
        'rule': 'the four files and every config listed here are frozen; later changes only as timestamped revision records appended to '
                'the revisions_after_freeze list of the file concerned, and only before any batch result is seen; '
                'check with python3 python/scripts/build_freeze_manifest.py --check before generation and before evaluation',
        'frozen_files': {f: sha(f) for f in FOUR},
        'referenced_configs': referenced(),
        'code': {'commit': commit, 'areas': code(commit),
                 'not_yet_written': 'the batch plan builder (inference_tasks rows to queue plans; development rows only before the '
                                    'held-out decision): its commit is appended as a revision record of the inference registration, '
                                    'with a check that every job reproduces its frozen task row (method, condition, task, paraphrase, '
                                    'seed, text, target), before generation starts'},
        'upstream_repositories': upstream(),
        'execution_order': 'generation first; evaluation starts after the whole generation batch has finished '
                           '(inference registration execution_order, user 2026-11-09 04:08)',
    }


def check():
    m = json.loads(OUT.read_text())
    bad = [f'{p}: sha256 differs' for p, h in m['frozen_files'].items() if sha(p) != h]
    bad += [f'{p}: sha256 differs' for p, r in m['referenced_configs'].items() if sha(p) != r['sha256']]
    now = referenced()
    bad += [f'{p}: newly referenced, not in the manifest' for p in now if p not in m['referenced_configs']]
    for name, area in m['code']['areas'].items():
        changed = git('diff', '--name-only', m['code']['commit'], '--', *area['paths'])
        dirty = git('status', '--porcelain', '--', *area['paths'])
        if changed or dirty:
            bad.append(f'{name} code differs from {m["code"]["commit"][:12]}: {changed} {dirty}'.strip())
    for repo, r in upstream().items():
        if repo not in m['upstream_repositories'] or r != m['upstream_repositories'][repo]:
            bad.append(f'{repo}: {r} != {m["upstream_repositories"].get(repo)}')
    print(json.dumps({'manifest': str(OUT.relative_to(ROOT)), 'differences': bad}, indent=1))
    return 1 if bad else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    if parser.parse_args().check:
        raise SystemExit(check())
    OUT.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + '\n')
    print(OUT.relative_to(ROOT))
