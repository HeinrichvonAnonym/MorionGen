#!/usr/bin/env python3
"""Check the gold K registration without running any motion/model evaluation."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def check(folder, verify_local=False):
    def read(name):
        return json.loads((folder / name).read_text())

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    split, evidence, degradation = (read(n) for n in ('split.json', 'evidence.json', 'degradations.json'))
    rows = [json.loads(line) for line in (folder / 'trials.jsonl').read_text().splitlines()]
    require(digest(folder / 'trials.jsonl') == split['manifest']['sha256'], 'Manifest hash changed')
    require(split['protocol'] == degradation['protocol'], 'Protocol mismatch')
    ids = {r['id']: r for r in rows}
    require(len(ids) == len(rows), 'Duplicate row IDs')
    group_parts, recording_parts = defaultdict(set), defaultdict(set)
    for r in rows:
        require(r['status'] in ('included', 'zero_bytes', 'duplicate_alias'), f"Unknown status: {r['id']}")
        require(r['partition'] in ('development', 'final_holdout'), f"Unknown partition: {r['id']}")
        group_parts[r['group_id']].add(r['partition'])
        recording_parts[r['recording_id']].add(r['partition'])
        if r['dataset'] == 'andy':
            expected = 'final_holdout' if r['subject'] in split['andy']['final_holdout_subjects'] else 'development'
            require(r['partition'] == expected, 'Andy subject mismatch')
            require(r['group_id'] == 'andy/'+r['subject'], 'Andy subject grouping mismatch')
        elif r['dataset'] == 'collaborate':
            expected = 'final_holdout' if r['subset'] in split['collaborate']['final_holdout_subsets'] else 'development'
            require(r['partition'] == expected, 'CoLLaboratE scene mismatch')
        elif r['dataset'] == 'han':
            require(r['partition'] == 'development', 'Previously exposed Han subject in final holdout')
            require(r['group_id'] == 'han/'+r['subject'], 'Han split-file treated as a person')
        else:
            raise ValueError(f"Unknown dataset: {r['dataset']}")
    require(all(len(v) == 1 for v in group_parts.values()), 'Subject leakage')
    require(all(len(v) == 1 for v in recording_parts.values()), 'Recording leakage')
    for item in split['counts']:
        rr = [r for r in rows if r['dataset'] == item['dataset'] and r['partition'] == item['partition']]
        good = [r for r in rr if r['status'] == 'included']
        expected = dict(entries=len(rr), included_entries=len(good), recordings=len({r['recording_id'] for r in good}),
                        subject_groups=len({r['group_id'] for r in good}),
                        exclusions=dict(Counter(r['status'] for r in rr if r['status'] != 'included')))
        require(all(item[k] == v for k, v in expected.items()), f'Count mismatch: {item}')
    for group in evidence['duplicate_groups']:
        canonical = ids[group['canonical']]
        require(canonical['status'] == 'included', 'Duplicate canonical excluded')
        for alias in group['aliases']:
            r = ids[alias]
            require(r['status'] == 'duplicate_alias' and r['recording_id'] == canonical['id'], 'Duplicate alias lost')
            require(r['partition'] == canonical['partition'], 'Duplicate leaks across partition')
    dev_groups = {r['group_id'] for r in rows if r['partition'] == 'development'}
    for fold in split['development_folds']:
        train, valid = set(fold['training_groups']), set(fold['validation_groups'])
        require(train and valid and not train & valid, f"CV overlap: {fold['id']}")
        require(train | valid <= dev_groups, f"Holdout in development CV: {fold['id']}")
    require(set(split['andy']['forced_development_subjects']) <= set(split['andy']['development_subjects']), 'Historical Andy exposure in holdout')
    han_event = next(e for e in evidence['events'] if e['id'] == 'han_val_a')
    require(set(han_event['kin30_counts_by_subject']) == set(split['han']['development_subjects']), 'Historical Han subjects lost')
    require(sum(han_event['kin30_counts_by_subject'].values()) == len(han_event['kin30_result_files']), 'Historical count mismatch')
    profiles = degradation['profiles']
    methods = {'motionlcm', 'omnicontrol', 'ardy', 'chois', 'closd', 'dart', 'kimodo', 'primal'}
    require(len(profiles) == degradation['profile_count'] == 8, 'Expected eight profiles')
    require({p['method'] for p in profiles} == methods, 'Method coverage mismatch')
    require(len({p['id'] for p in profiles}) == 8, 'Duplicate profile IDs')
    for p in profiles:
        require(p['source_sample_method'] in evidence['baseline_samples'], 'Profile has no actual sample')
        require(bool(p['allowed_inputs']) and bool(p['transform']), 'Incomplete degradation definition')
        expected = 'withheld' if p['method'] in ('motionlcm', 'omnicontrol') else 'preserve_native'
        require(p['head_orientation_policy'] == expected, f"Wrong native capability: {p['method']}")

    checked = 0
    if verify_local:
        # Traverse only explicit evidence records. Never open held-out motion for evaluation.
        def walk(obj):
            nonlocal checked
            if isinstance(obj, dict):
                if {'path', 'bytes', 'sha256'} <= obj.keys():
                    p = Path(obj['path'])
                    p = p if p.is_absolute() else ROOT / p
                    require(p.is_file(), f'Missing evidence: {p}')
                    require(p.stat().st_size == obj['bytes'] and digest(p) == obj['sha256'], f'Evidence changed: {p}')
                    checked += 1
                for value in obj.values():
                    walk(value)
            elif isinstance(obj, list):
                for value in obj:
                    walk(value)
        walk(evidence)
        inventory = {Path(s['path']).stem: json.loads((ROOT / s['path']).read_text()) for s in evidence['inventory_sources']}
        require({r['id'].removeprefix('andy/') for r in rows if r['dataset']=='andy'} == {r['trial'] for r in inventory['andy_audit']['trials']}, 'Andy inventory incomplete')
        expected_collab = {'collaborate/'+a['archive'][:-4]+'/'+r['file'] for a in inventory['mingei_audit'] for r in a['trials']}
        require({r['id'] for r in rows if r['dataset']=='collaborate'} == expected_collab, 'CoLLaboratE inventory incomplete')
        expected_han = {(r['file'], r['trial']) for r in inventory['han_audit']['trials']}
        require({(r['source']['file'], r['source']['trial']) for r in rows if r['dataset']=='han' and r['status']=='included'} == expected_han, 'Han inventory incomplete')
        expected_empty = {r['file'] for r in inventory['han_audit']['files'] if r.get('status')=='zero_bytes'}
        require({r['source']['file'] for r in rows if r['dataset']=='han' and r['status']=='zero_bytes'} == expected_empty, 'Han empty files omitted')
    return dict(status='passed', rows=len(rows), included_entries=sum(r['status']=='included' for r in rows),
                final_unique_recordings=len({r['recording_id'] for r in rows if r['partition']=='final_holdout' and r['status']=='included'}),
                profiles=len(profiles), development_folds=len(split['development_folds']), evidence_files_verified=checked)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, default=ROOT/'config/gold_k_20261115')
    parser.add_argument('--verify-local-evidence', action='store_true')
    args = parser.parse_args()
    print(json.dumps(check(args.registration, args.verify_local_evidence), ensure_ascii=False, indent=2))
