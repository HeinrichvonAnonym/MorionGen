#!/usr/bin/env python
"""Read members of the AddBiomechanics 1.0 archive (CC BY 4.0) over HTTP range requests, without the 417 GB download.

  index                 write the central directory to <data>/index.json
  fetch PATTERN [...]   download members whose names match any regex (resumable; size and CRC checked)
"""
import argparse
import json
import re
import struct
import sys
import urllib.request
import zlib
from pathlib import Path

URL = 'http://archive.simtk.org/addbiomechanics/addbiomechanics.zip'
DATA = Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c95839/MorionGen/datasets/addbiomechanics_v1')


def get(start, end):
    req = urllib.request.Request(URL, headers={'Range': f'bytes={start}-{end}'})
    with urllib.request.urlopen(req, timeout=120) as r:
        if r.status != 206:
            raise RuntimeError(f'range request returned {r.status}')
        return r.read()


def size():
    with urllib.request.urlopen(urllib.request.Request(URL, method='HEAD'), timeout=60) as r:
        return int(r.headers['Content-Length'])


def zip64_extra(extra, fields):
    """Replace 0xFFFFFFFF fields by their ZIP64 extra values, in the order size, compressed size, offset."""
    i = 0
    while i+4 <= len(extra):
        tag, n = struct.unpack_from('<HH', extra, i)
        if tag == 1:
            vals = list(struct.unpack_from(f'<{n//8}Q', extra, i+4)); out = []
            for v in fields:
                out.append(vals.pop(0) if v == 0xFFFFFFFF and vals else v)
            return out
        i += 4+n
    return fields


def index():
    total = size()
    tail = get(total-(1 << 16), total-1)
    loc = tail.rfind(b'PK\x06\x07')
    if loc >= 0:
        eocd64 = struct.unpack_from('<Q', tail, loc+8)[0]
        rec = get(eocd64, eocd64+55)
        count, cd_size, cd_off = struct.unpack_from('<QQQ', rec, 32)
    else:
        e = tail.rfind(b'PK\x05\x06')
        count, cd_size, cd_off = struct.unpack_from('<HII', tail, e+10)
    cd = get(cd_off, cd_off+cd_size-1)
    entries, i = [], 0
    while i < len(cd) and cd[i:i+4] == b'PK\x01\x02':
        (method, crc, csize, usize, nlen, xlen, clen, off) = (struct.unpack_from('<H', cd, i+10)[0], *struct.unpack_from('<III', cd, i+16),
                                                              *struct.unpack_from('<HHH', cd, i+28), struct.unpack_from('<I', cd, i+42)[0])
        name = cd[i+46:i+46+nlen].decode('utf-8', 'replace')
        usize, csize, off = zip64_extra(cd[i+46+nlen:i+46+nlen+xlen], [usize, csize, off])
        entries.append({'name': name, 'method': method, 'crc': crc, 'compressed': csize, 'size': usize, 'offset': off})
        i += 46+nlen+xlen+clen
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA/'index.json').write_text(json.dumps({'url': URL, 'bytes': total, 'entries': entries}))
    print(json.dumps({'entries': len(entries), 'declared': count, 'bytes': total}))


def fetch(patterns, limit_gb):
    entries = json.loads((DATA/'index.json').read_text())['entries']
    chosen = [e for e in entries if not e['name'].endswith('/') and any(re.search(p, e['name']) for p in patterns)]
    total = sum(e['compressed'] for e in chosen)
    print(json.dumps({'members': len(chosen), 'compressed_gb': round(total/1e9, 2)}), flush=True)
    if total > limit_gb*1e9:
        sys.exit(f'selection is {total/1e9:.1f} GB, above --limit-gb {limit_gb}')
    for e in chosen:
        out = DATA/'files'/e['name']
        if out.exists() and out.stat().st_size == e['size']:
            continue
        head = get(e['offset'], e['offset']+29)
        nlen, xlen = struct.unpack_from('<HH', head, 26)
        start = e['offset']+30+nlen+xlen
        raw = get(start, start+e['compressed']-1) if e['compressed'] else b''
        data = zlib.decompress(raw, -15) if e['method'] == 8 else raw
        if len(data) != e['size'] or zlib.crc32(data) != e['crc']:
            raise RuntimeError(f"size or CRC mismatch for {e['name']}")
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix+'.part'); tmp.write_bytes(data); tmp.replace(out)
    print(json.dumps({'fetched': len(chosen)}))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('index')
    f = sub.add_parser('fetch'); f.add_argument('patterns', nargs='+'); f.add_argument('--limit-gb', type=float, default=40.)
    a = p.parse_args()
    index() if a.cmd == 'index' else fetch(a.patterns, a.limit_gb)


if __name__ == '__main__':
    main()
