"""Independent spreadsheet extraction; distinguish source agreement from validity."""
import hashlib
import json
from pathlib import Path

import xlrd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'develop_logs/test_runs/20261028_supplement'
source = BASE/'sources/reba_tables.xls'
book = xlrd.open_workbook(source)
configured = json.loads((ROOT/'config/ergonomic_tables.json').read_text())['REBA']
locations = {
    'REBA Assessment Worksheet': {'A':(41,2,5,12), 'B':(49,2,6,6), 'C':(58,3,12,12)},
    'Report 2': {'A':(6,7,5,12), 'B':(16,7,6,6), 'C':(27,6,12,12)}}
extracted = {}
comparisons = []
for sheetname, tables in locations.items():
    sheet = book.sheet_by_name(sheetname)
    extracted[sheetname] = {}
    for name,(r0,c0,nr,nc) in tables.items():
        values = [[int(sheet.cell_value(r0+r,c0+c)) for c in range(nc)] for r in range(nr)]
        extracted[sheetname][name] = values
        differences = [{'row_zero_based':r,'column_zero_based':c,'spreadsheet':values[r][c],
                        'engine':configured[name][r][c]}
                       for r in range(nr) for c in range(nc) if values[r][c]!=configured[name][r][c]]
        comparisons.append(dict(sheet=sheetname, table=name, cells=nr*nc, differences=differences))
assert all(not r['differences'] for r in comparisons if r['sheet']=='REBA Assessment Worksheet')
print(json.dumps(comparisons))
result=dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            comparisons=comparisons,extracted_tables=extracted,
            scope='Independent machine extraction of 240 cells from the selected Cornell main worksheet. Internal workbook disagreement is retained; agreement with one source is not clinical validity or independent human scoring.')
(BASE/'reba_source_table_audit.json').write_text(json.dumps(result,indent=2))
