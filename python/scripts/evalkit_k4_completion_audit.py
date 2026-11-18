"""Read-only audit of a completed frozen K4 regression; no fit or score changes."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    protocol = json.loads((a.run/'protocol.json').read_text())
    results = json.loads((a.run/'results_656.json').read_text())
    expected = {(r['window'], r['method']) for r in protocol['records']}
    actual = {(r['window'], r['method']) for r in results['rows']}
    assert len(expected) == len(actual) == len(results['rows']) == 656 and expected == actual
    assert len({w for w, _ in actual}) == 82
    assert set(Counter(m for _, m in actual).values()) == {82}
    sources = {(r['window'], r['method']): r for r in protocol['records']}
    failed = defaultdict(list)
    categories = Counter()
    clocks = Counter()
    audits = Counter()
    metric_counts = defaultdict(Counter)
    hashes = {}
    for row in results['rows']:
        assert row['status'] == 'completed'
        w, m = row['window'], row['method']
        summary_path = a.run/'tasks'/f'{w}_{m}'/'summary.json'
        summary = json.loads(summary_path.read_text())
        assert summary['passed'] == row['passed']
        assert summary['export_audit']['passed']
        audits['actual_export_passed'] += 1
        for view in ('native', 'export'):
            path = a.run/'scores'/view/f'{w}_{m}.json'
            score = json.loads(path.read_text())
            hashes[str(path)] = sha(path)
            for clock, data in (('D_clock', score), ('R_clock', score['original_clock'])):
                assert data['clock']['frames'] > 0
                for pair in ('D-R', 'F-D', 'F-R'):
                    for layer in ('full', 'continuous', 'source_jump_tolerance'):
                        values = data['comparisons'][pair][layer]
                        assert {'directions_deg', 'joint_angles_deg', 'positions_cm', 'temporal'} <= values.keys()
                        for nodes in values['positions_cm'].values():
                            for geometry in nodes.values():
                                assert {'world','pelvis_relative','common_length_world','common_length_pelvis_relative'} <= geometry.keys()
                        assert layer in data['head_neck_rotation_comparisons'][pair]
                acceptance = summary['acceptance'][view][clock]
                clocks[f'{view}/{clock}/{acceptance["status"]}'] += 1
                for item in acceptance['required'] + acceptance['full_window_coverage']:
                    metric_counts[(m, view, clock, item['metric'])][item['status']] += 1
                assert all(x['status'] in ('passed', 'unobservable') for x in acceptance['full_window_coverage'])
        if not row['passed']:
            failures = summary['failed_metrics']
            pose = any(x['metric'].startswith(('directions_deg/', 'positions_cm/', 'joint_angles_deg/', 'rotation/')) for x in failures)
            timing = any('rate' in x['metric'] or 'step' in x['metric'] for x in failures)
            category = 'pose_and_timing' if pose and timing else 'pose_only' if pose else 'timing_only'
            categories[category] += 1
            failed[w].append(dict(method=m, category=category, failures=failures))
    manifest = json.loads(Path(protocol['preservation_manifest']).read_text())
    workspace = Path(protocol['workspace_root'])
    changed = [n for n, digest in manifest['files'].items() if not (workspace/n).exists() or sha(workspace/n) != digest]
    assert not changed
    assert results['new_holdout_uses'] == 0
    out = dict(expected=656, completed=len(actual), passed=sum(r['passed'] for r in results['rows']),
               execution_failed=0, failed_windows=len(failed), failure_categories=dict(categories),
               acceptance_by_view_clock=dict(clocks), audit_counts=dict(audits),
               by_format=results['by_format'], by_participant_group=results['by_participant_group'],
               metric_counts=[dict(method=k[0],view=k[1],clock=k[2],metric=k[3],counts=dict(v)) for k,v in metric_counts.items()],
               failures=[dict(window=w, source=sources[(w, rs[0]['method'])], routes=rs) for w,rs in failed.items()],
               protection=dict(checked=len(manifest['files']),changed=changed,new_holdout_uses=0),
               score_sha256=hashes, results_sha256=sha(a.run/'results_656.json'),
               scope='Checks preserved score artifacts and denominators; does not rerun or modify frozen scoring; paired routes are not independent participants.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:
        json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps({k:out[k] for k in ('completed','passed','failed_windows','failure_categories','protection','acceptance_by_view_clock')}))


if __name__ == '__main__':
    main()
