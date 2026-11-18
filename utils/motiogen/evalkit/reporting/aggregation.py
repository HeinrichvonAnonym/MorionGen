"""Window-level acceptance aggregation; paired formats are never distinct motions."""
from collections import Counter
from statistics import mean,median


def summarize(records):
    """Records retain the fixed denominator and exact saved acceptance decisions.

    No pooled frame percentiles are reconstructed from per-window statistics.
    An unobservable metric is counted separately from passing an observed one.
    """
    result={}
    for dimension,key_for in (
        ('by_format',lambda r:r['method']),
        ('by_participant_format',lambda r:(r.get('group_id') or 'undeclared')+' | '+r['method']),
    ):
        groups={}
        for row in records:groups.setdefault(key_for(row),[]).append(row)
        result[dimension]={}
        for key,rows in groups.items():
            completed=[r for r in rows if r['status']=='completed']
            group=dict(expected_paths=len(rows),original_windows=len({r['window'] for r in rows}),
                       completed_paths=len(completed),passed_paths=sum(r.get('passed') is True for r in rows),
                       execution_failed_paths=sum(r['status']=='failed' for r in rows),
                       pending_paths=sum(r['status']=='pending' for r in rows),views={})
            for view in ('native','export'):
                group['views'][view]={}
                for clock in ('D_clock','R_clock'):
                    block={}
                    for category in ('required','temporal','full_window_coverage'):
                        metrics={}
                        for r in completed:
                            seen=set()
                            for item in r['acceptance'][view][clock][category]:
                                name=item['metric']
                                if name in seen:raise ValueError('Repeated metric within one window')
                                seen.add(name);metrics.setdefault(name,[]).append(item)
                        stats={}
                        for name,items in metrics.items():
                            values=[x['p95'] for x in items if x.get('p95') is not None]
                            states=Counter(x['status'] for x in items)
                            targets={x['target'] for x in items if x.get('target') is not None}
                            if len(targets)>1:raise ValueError('Inconsistent target across paired windows')
                            stats[name]=dict(status_counts=dict(states),reported_paths=len(items),
                                unreported_completed_paths=len(completed)-len(items),ungraded_paths=len(rows)-len(completed),
                                observed_frame_support=sum(x.get('expected_observed',0) for x in items),scored_frames=sum(x.get('scored',0) for x in items),
                                target=next(iter(targets),None),window_p95_count=len(values),
                                mean_window_p95=mean(values) if values else None,median_window_p95=median(values) if values else None,
                                worst_window_p95=max(values) if values else None)
                        block[category]=stats
                    group['views'][view][clock]=block
            result[dimension][key]=group
    result['scope']='Fixed path denominators; per-window P95 summaries are not pooled frame percentiles. Participant-format groups preserve registered identities. Eight formats of one original window are paired representations, not eight independent motions. D-R/F-D/F-R and geometry-specific distributions remain in per-record score evidence.'
    return result
