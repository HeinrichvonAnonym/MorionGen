"""Download public OrthoLoad motion attachments for local spine load records."""
import argparse
from motiogen.orthoload_download import Downloader


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, help='Existing orthoload_v1 dataset directory')
    p.add_argument('--jobs', type=int, choices=range(1,9), default=4)
    p.add_argument('--limit', type=int, help='Optional bounded initial verification')
    a=p.parse_args()
    summary=Downloader(a.root,a.jobs).run(a.limit)
    return int(any(x['statuses'].get('failed',0) for x in summary['by_implant'].values()))


if __name__=='__main__':
    raise SystemExit(main())
