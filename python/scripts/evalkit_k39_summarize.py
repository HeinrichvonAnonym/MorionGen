"""Summarize fixed full-window K3.9 results without dropping failures."""
from evalkit_k39_prepare import *

def main():
 parser=argparse.ArgumentParser();parser.add_argument('results',type=Path);parser.add_argument('--expected',type=int,default=656);a=parser.parse_args()
 records=[read(p) for p in sorted(a.results.glob('*/*/result.json'))];good=[r for r in records if r['status']=='computed']
 summary={'expected_pairs':a.expected,'finished_pairs':len(records),'computed_pairs':len(good),'failed_pairs':len(records)-len(good),
  'pending_pairs':a.expected-len(records),'scope':'development only; window-balanced distributions of full-window statistics; angles degrees, positions metres',
  'methods':{},'reference_regressions':[{'row':r['row'],'regressions':r['reference_regressions']} for r in good if r['reference_regressions']]}
 for method in ['motionlcm','omnicontrol','kimodo','dart','closd','chois','ardy','primal']:
  rows=[r for r in good if r['row']['method']==method];result={'computed':len(rows),'participants':len({r['row']['group_id'] for r in rows}),'selected':{},'metrics':{}}
  for r in rows:result['selected'][r['selection']['selected']]=result['selected'].get(r['selection']['selected'],0)+1
  if rows:
   for space in ['raw','normalized']:
    result['metrics'][space]={}
    for kind in ['angles','positions','directions']:
     result['metrics'][space][kind]={}
     for metric in rows[0]['scores']['selected'][space][kind]:
      stages={}
      for stage in ['degradation','fit_deviation','end_to_end']:
       values=[r['scores']['selected'][space][kind][metric][stage] for r in rows]
       # Positions are reported by world/root-relative submetric in the scorer.
       if not values or 'p95_abs' not in values[0]:
        stages[stage]={}
        for frame in ['world','root_relative']:
         finite=[v[frame]['p95_abs'] for v in values if v[frame]['p95_abs'] is not None]
         stages[stage][frame]={'median_window_p95':float(np.median(finite)) if finite else None,'worst_window_p95':max(finite) if finite else None,'windows_at_target':sum(v<=.05 for v in finite),'scorable_windows':len(finite),'completed_windows':len(rows),'mean_scoring_coverage':float(np.mean([v[frame]['coverage'] for v in values]))}
        continue
       finite=[v['p95_abs'] for v in values if v['p95_abs'] is not None];limit=.05 if kind=='positions' else 5.
       stages[stage]={'median_window_p95':float(np.median(finite)) if finite else None,'worst_window_p95':max(finite) if finite else None,
        'windows_at_target':sum(v<=limit for v in finite),'scorable_windows':len(finite),'completed_windows':len(rows),
        'mean_scoring_coverage':float(np.mean([v['coverage'] for v in values]))}
      result['metrics'][space][kind][metric]=stages
  summary['methods'][method]=result
 atomic(a.results/'summary.json',summary)
 print({k:v for k,v in summary.items() if k not in ['methods','reference_regressions']});print('reference regressions',len(summary['reference_regressions']))
if __name__=='__main__':main()
