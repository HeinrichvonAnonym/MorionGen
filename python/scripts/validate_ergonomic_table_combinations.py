"""Exhaust lookup combinations against independently extracted source tables."""
from itertools import product
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'utils'))
from motiogen.ergonomics import _combine
from motiogen.runtime import checksum,json_write
BASE=ROOT/'develop_logs/test_runs/20261028_supplement'
rula_path=BASE/'rula_source_table_audit.json';reba_path=BASE/'reba_source_table_audit.json'
sources={'RULA':json.loads(rula_path.read_text())['extracted_tables'],
         'REBA':json.loads(reba_path.read_text())['extracted_tables']['REBA Assessment Worksheet']}
started=time.monotonic();counts={}
for method,t in sources.items():
    if method=='RULA':
        A=dict(zip(product(range(1,7),range(1,4),range(1,5),range(1,3)),[v for row in t['A'] for v in row]))
        B=dict(zip(product(range(1,7),range(1,7),range(1,3)),[v for row in t['B'] for v in row]))
        C=dict(zip(product(range(1,9),range(1,8)),[v for row in t['C'] for v in row]))
    else:
        A=dict(zip(product(range(1,6),range(1,4),range(1,5)),[v for row in t['A'] for v in row]))
        B=dict(zip(product(range(1,7),range(1,3),range(1,4)),[v for row in t['B'] for v in row]))
        C=dict(zip(product(range(1,13),range(1,13)),[v for row in t['C'] for v in row]))
    count=0
    for ak,av in A.items():
        for bk,bv in B.items():
            if method=='RULA':
                fixed=dict(zip(['upper_arm','lower_arm','wrist','wrist_twist'],ak));fixed.update(zip(['neck','trunk','legs'],bk))
                adjustments=product(range(2),range(4),range(2),range(4))
            else:
                fixed=dict(zip(['trunk','neck','legs'],ak));fixed.update(zip(['upper_arm','lower_arm','wrist'],bk))
                adjustments=product(range(4),range(4),range(4))
            for adjustment in adjustments:
                if method=='RULA':
                    ma,la,mb,lb=adjustment
                    expected=C[(min(8,av+ma+la),min(7,bv+mb+lb))]
                    values=dict(fixed,muscle_a=ma,load_a=la,muscle_b=mb,load_b=lb)
                else:
                    load,coupling,activity=adjustment
                    expected=C[(min(12,av+load),min(12,bv+coupling))]+activity
                    values=dict(fixed,load=load,coupling=coupling,activity=activity)
                actual,a,b=_combine(method,tuple(sorted((k,(v,)) for k,v in values.items())))
                if (actual,a,b)!=((expected,),(av,),(bv,)):
                    raise AssertionError((method,values,actual,expected,a,b))
                count+=1
    counts[method]=count
    print(method,count,flush=True)
assert counts=={'RULA':663552,'REBA':138240}
json_write(BASE/'ergonomic_lookup_validation.json',dict(counts=counts,elapsed_seconds=time.monotonic()-started,
    sources={str(p):checksum(p) for p in [rula_path,reba_path]},
    scope='Exhaustive discrete lookup/capping/adjustment validation against independently extracted selected sources. Does not establish anatomical-angle validity or inter-rater agreement. Source disagreements remain separately recorded.'))
