import importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
s=importlib.util.spec_from_file_location('distributions',ROOT/'utils/motiogen/evalkit/reporting/error_distribution.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_complement_requires_both_crossed_outcomes_and_keeps_dominated_out(tmp_path):
 rows=[]
 for w,a,b in [('001',True,False),('002',False,True),('003',False,True)]:
  cs={k:dict(status='evaluated',passed=v) for k,v in dict(a=a,b=b,always_bad=False,always_good=True).items()}
  cs['alias']=dict(status='evaluated',passed=b,equivalent_to='b')
  rows.append(dict(window=w,method='example',fully_evaluated=True,candidates=cs))
 p=tmp_path/'snapshot.json';p.write_text(json.dumps(dict(rows=rows)))
 g=m.complementary_methods(p)['example']
 assert g['members']==['a','b']
 assert len(g['pairs'])==1
 assert g['pairs'][0]['a_only']==['001']
 assert g['pairs'][0]['b_only']==['002','003']
 # Membership includes the method, not just its passing windows.
 selected=[(r['window'],n,c['passed']) for r in rows for n,c in r['candidates'].items() if n in g['members']]
 assert len(selected)==6 and sum(not x[2] for x in selected)==3

def test_incomplete_routes_cannot_create_complement(tmp_path):
 rows=[dict(window='001',method='x',fully_evaluated=True,candidates={'a':dict(status='evaluated',passed=True),'b':dict(status='evaluated',passed=False)}),dict(window='002',method='x',fully_evaluated=False,candidates={'a':dict(status='evaluated',passed=False),'b':dict(status='evaluated',passed=True)})]
 p=tmp_path/'snapshot.json';p.write_text(json.dumps(dict(rows=rows)))
 assert m.complementary_methods(p)['x']['members']==[]
