#!/usr/bin/env python
"""Append one timestamped record to the revisions_after_freeze list of a frozen file (freeze-20261109), leaving every
other byte of the file as it is.

  append_revision.py FILE RECORD_JSON
The record gets time (local clock, minutes) and before_any_batch_result: true; the caller asserts the latter by running
this before any batch result has been seen. Verified afterwards: the file parses and equals its previous content apart
from the one appended record.
"""
import json
import sys
import time
from pathlib import Path

KEY = '"revisions_after_freeze": ['


def main():
    path, record = Path(sys.argv[1]), json.loads(sys.argv[2])
    record = {'time': time.strftime('%Y-%m-%d %H:%M'), 'before_any_batch_result': True, **record}
    text = path.read_text()
    before = json.loads(text)
    lines = text.split('\n')
    start = next(i for i, l in enumerate(lines) if l.lstrip().startswith(KEY))
    indent = lines[start][:len(lines[start]) - len(lines[start].lstrip())]
    body = '\n'.join(indent + ' ' + l for l in json.dumps(record, indent=1, ensure_ascii=False).split('\n'))
    if lines[start].rstrip().endswith('[],'):
        lines[start:start+1] = [indent + KEY, body, indent + '],']
    else:
        end = next(i for i in range(start+1, len(lines)) if lines[i].rstrip() in (indent + '],', indent + ']'))
        lines[end-1] = lines[end-1] + ','
        lines.insert(end, body)
    out = '\n'.join(lines)
    after = json.loads(out)
    added = after.pop('revisions_after_freeze')
    if added[:-1] != before.pop('revisions_after_freeze') or added[-1] != record or after != before:
        raise SystemExit('append changed more than the one record')
    path.write_text(out)
    print(f'{path}: {len(added)} revision record(s)')


if __name__ == '__main__':
    main()
