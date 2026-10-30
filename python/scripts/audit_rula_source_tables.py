"""Extract primary RULA A/B and Cornell C independently of engine JSON."""
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'utils'))
from motiogen.runtime import checksum,json_write
BASE=ROOT/'develop_logs/test_runs/20261028_supplement'
sources=BASE/'sources'
paper=sources/'rula_original.pdf';worksheet=sources/'rula_worksheet.pdf'
original=subprocess.check_output(['pdftotext','-layout',str(paper),'-'],text=True)
cornell=subprocess.check_output(['pdftotext','-layout',str(worksheet),'-'],text=True)
tables={}
a=original.split('Table 1 Table A',1)[1].split('Table 2 Table B',1)[0]
values=[]
for line in a.splitlines():
    tokens=line[:80].split()
    if len(tokens)==9 and all(re.fullmatch('[1-9]',t) for t in tokens):values.append(list(map(int,tokens)))
assert len(values)==18 and [v[0] for v in values]==[1,2,3]*6
tables['A']=[v[1:] for v in values]
b=original.split('Table 2 Table B',1)[1].split('Vol 24 No 2',1)[0]
values=[]
for line in b.splitlines():
    # PDF text extraction recognizes the printed first score 1 as '!'.
    # This one explicit glyph correction is recorded below, not generalized OCR.
    tokens=line.replace('!','1').split()
    if len(tokens)==13 and all(re.fullmatch('[1-9]',t) for t in tokens):values.append(list(map(int,tokens)))
assert len(values)==6 and [v[0] for v in values]==list(range(1,7))
tables['B']=[v[1:] for v in values]
c=cornell.split('Step 5: Look-up Posture Score in Table A',1)[1].split('Step 8: Find Row in Table C',1)[0]
values=[]
for line in c.splitlines():
    tokens=line[118:210].replace('+','').split()
    if len(tokens)==8 and all(re.fullmatch('[1-9]',t) for t in tokens):values.append(list(map(int,tokens)))
assert len(values)==8 and [v[0] for v in values]==list(range(1,9)),values
tables['C']=[v[1:] for v in values]
engine=json.loads((ROOT/'config/ergonomic_tables.json').read_text())['RULA']
checks=[]
for name,rows in tables.items():
    differences=[{'row':r,'column':c,'source':value,'engine':engine[name][r][c]} for r,row in enumerate(rows) for c,value in enumerate(row) if value!=engine[name][r][c]]
    checks.append(dict(table=name,cells=sum(map(len,rows)),differences=differences))
json_write(BASE/'rula_source_table_audit.json',dict(extracted_tables=tables,checks=checks,
    sources={str(p):checksum(p) for p in [paper,worksheet]},
    text_corrections=[{'table':'B','cell':'neck1/trunk1/legs1','extracted':'!','interpreted':1,'reason':'Printed digit one; one explicit PDF glyph normalization.'}],
    scope='A/B from original 1993 paper, C from Cornell worksheet. Cornell posture threshold/secondary A-table variants are not silently mixed into the selected original rule version. Independent source extraction does not establish human scoring validity.'))
assert all(not c['differences'] for c in checks),checks
print(json.dumps(checks))
