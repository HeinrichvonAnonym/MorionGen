"""Fetch public motion attachments paired with locally held OrthoLoad AKFs.

Preserve original media, API responses, transport receipts and trial-level links.
Video is recorded as video evidence; no estimated 3-D trajectory is manufactured.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json
import shutil
import subprocess
import threading
import time
import urllib.parse
import requests

SITE = 'https://orthoload.com'
AJAX = SITE + '/wp-admin/admin-ajax.php'
IMPLANTS = {648: ('Vertebral Body Replacement', 'vertebral_body'),
            1309: ('Internal Spinal Fixator', 'internal_spinal_fixator')}
VIDEO_TYPES = {'SCREENVIDEO', 'AVI', 'RAW_MOV', 'PATIENT_VIDEO'}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class Downloader:
    def __init__(self, root, jobs=4):
        self.root = Path(root).resolve()
        if not (self.root/'index.json').is_file() or not (self.root/'akf').is_dir():
            raise ValueError('Dataset root must contain index.json and akf/')
        self.out = self.root/'motion'
        self.out.mkdir(exist_ok=True)
        self.jobs = jobs
        self.local = threading.local()
        self.index = json.loads((self.root/'index.json').read_text())['trials']
        self.selected = {k: v for k, v in self.index.items()
                         if v['implant'] in {x[0] for x in IMPLANTS.values()}
                         and (self.root/'akf'/f'{k}.akf').is_file()}

    def session(self):
        if not hasattr(self.local, 'session'):
            self.local.session = requests.Session()
            self.local.session.verify = '/etc/ssl/certs/ca-certificates.crt'
        return self.local.session

    def api(self, action, **query):
        query = dict(action=action, **query)
        key = hashlib.sha256(json.dumps(query, sort_keys=True).encode()).hexdigest()
        path = self.out/'metadata/api'/f'{key}.json'
        if path.exists():
            return json.loads(path.read_text())['response']['data']
        url = AJAX+'?'+urllib.parse.urlencode(query)
        for attempt in range(4):
            try:
                with self.session().get(url, timeout=(20,60)) as response:
                    response.raise_for_status()
                    if len(response.content)>12*1024*1024:
                        raise ValueError('Unexpected API response size')
                    result = response.json()
                if not result.get('success'):
                    raise ValueError('API success flag absent')
                save(path, dict(query=query, fetched_at=stamp(), response=result))
                return result['data']
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2**(attempt+1))

    def catalogue(self):
        tasks = []
        for iid, (name, folder) in IMPLANTS.items():
            acts = self.api('orthoload_get_activities', implantId=iid)
            for a in acts:
                if a['indentaionLevel'] == 0:
                    tasks.append(dict(implantId=iid, activityId=a['id'],
                                      activityIndentationLevel=0, parameterId='',
                                      parameterIndentationLevel='', patientId='all'))
        links = {}; additional = []; groups = []
        def read_group(query):
            return query, self.api('orthoload_get_files', **query)
        with ThreadPoolExecutor(max_workers=self.jobs) as pool:
            for query, data in pool.map(read_group, tasks):
                groups.append(dict(query=query, response=data))
                for f in data.get('files', []):
                    if f['id'] in self.selected:
                        links.setdefault(f['id'], dict(query, fileId=f['id'], fileType=f['filetype']))
                for f in data.get('additionalFiles', []):
                    additional.append(dict(query=query, file=f))
        missing = sorted(self.selected.keys()-links.keys())
        def lookup(tid):
            return tid, self.api('orthoload_searchData', term=tid)
        with ThreadPoolExecutor(max_workers=self.jobs) as pool:
            for i, (tid, found) in enumerate(pool.map(lookup, missing), 1):
                if found:
                    links[tid] = next(x for x in found if x['fileId'] == tid)
                if i % 50 == 0:
                    print(json.dumps(dict(stage='trial_lookup', completed=i, total=len(missing))), flush=True)
        # Parameter-specific trials may expose additional measurements outside
        # the root activity listing. Query every represented condition, all patients.
        known = {json.dumps(q, sort_keys=True) for q in tasks}
        conditions = {}
        for q in links.values():
            query = {k:v for k,v in q.items() if k not in {'fileId','fileType'}}
            query['patientId'] = 'all'
            key = json.dumps(query, sort_keys=True)
            if key not in known:
                conditions[key] = query
        with ThreadPoolExecutor(max_workers=self.jobs) as pool:
            for i, (query, data) in enumerate(pool.map(read_group, conditions.values()), 1):
                groups.append(dict(query=query, response=data))
                for f in data.get('additionalFiles', []):
                    additional.append(dict(query=query, file=f))
                if i % 50 == 0:
                    print(json.dumps(dict(stage='additional_lookup', completed=i, total=len(conditions))), flush=True)
        save(self.out/'metadata/catalogue.json', dict(fetched_at=stamp(), groups=groups,
             links=links, additional=additional, selected_records=len(self.selected),
             unmatched=sorted(self.selected.keys()-links.keys())))
        print(json.dumps(dict(stage='catalogue', selected=len(self.selected), matched=len(links),
                              root_activity_queries=len(tasks), condition_queries=len(conditions),
                              additional_files=len(additional))), flush=True)
        return links, additional

    def validate_video(self, path):
        p = subprocess.run(['ffprobe', '-v', 'error', '-show_entries',
            'format=duration,size:stream=index,codec_type,codec_name,width,height,r_frame_rate,nb_frames',
            '-of', 'json', str(path)], capture_output=True, text=True, timeout=90)
        if p.returncode:
            raise ValueError('ffprobe failed: '+p.stderr[-300:])
        data = json.loads(p.stdout)
        if not any(x.get('codec_type') == 'video' for x in data.get('streams', [])):
            raise ValueError('No video stream')
        p = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-threads', '1',
                            '-i', str(path), '-map', '0:v:0', '-f', 'null', '-'],
                           capture_output=True, text=True, timeout=180)
        if p.returncode or p.stderr.strip():
            raise ValueError('Full video decode failed: '+p.stderr[-400:])
        data['full_video_decode'] = 'passed'
        return data

    def asset(self, item, folder):
        url = urllib.parse.urljoin(SITE, item['path'])
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != 'https' or parsed.hostname != 'orthoload.com':
            raise ValueError('Unexpected attachment host')
        name = Path(parsed.path).name
        if not name or name in {'.', '..'}:
            raise ValueError('Unexpected attachment name')
        kind = 'videos' if item['type'] in VIDEO_TYPES else 'additional'
        relative = Path(kind)/folder/name
        path = self.out/relative
        key = hashlib.sha256(url.encode()).hexdigest()
        receipt = self.out/'metadata/assets'/f'{key}.json'
        if receipt.exists():
            old = json.loads(receipt.read_text())
            if old['status'] == 'downloaded' and path.exists() and sha(path) == old['sha256']:
                return old
            if old['status'] == 'remote_missing':
                return old
        result = dict(url=url, official_type=item['type'], official_name=item['name'],
                      local_path=str(relative), checked_at=stamp())
        if 'listed_path' in item:
            result.update(listed_url=urllib.parse.urljoin(SITE,item['listed_path']),
                          resolution_basis='Legacy AVI entry in pat_mp4 directory; same basename with .mp4 verified by download and decode')
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name+'.part')
        for attempt in range(4):
            try:
                if shutil.disk_usage(self.root).free < 80*1024**3:
                    raise RuntimeError('Data disk has less than 80 GiB reserve')
                with self.session().get(url, stream=True, timeout=(20,90)) as response:
                    response.raise_for_status()
                    expected = response.headers.get('Content-Length')
                    if expected and int(expected)>shutil.disk_usage(self.root).free-80*1024**3:
                        raise RuntimeError('File would cross the 80 GiB disk reserve')
                    result['content_type'] = response.headers.get('Content-Type')
                    n = 0
                    with partial.open('wb') as f:
                        for block in response.iter_content(1024*1024):
                            if shutil.disk_usage(self.root).free<80*1024**3:
                                raise RuntimeError('Download stopped to preserve 80 GiB of disk space')
                            f.write(block); n += len(block)
                if not n or expected and n != int(expected):
                    raise ValueError('Incomplete transfer')
                if kind == 'videos':
                    result['media'] = self.validate_video(partial)
                else:
                    if result.get('content_type', '').startswith('text/html'):
                        raise ValueError('Attachment returned HTML')
                result.update(status='downloaded', bytes=n, sha256=sha(partial))
                partial.replace(path)
                save(receipt, result)
                return result
            except requests.HTTPError as exc:
                code = exc.response.status_code
                result.update(status='remote_missing' if code in (404,410) else 'failed',
                              http_status=code, error=f'HTTP {code}')
                if code in (404,410,401,403):
                    break
            except Exception as exc:
                result.update(status='failed', error=type(exc).__name__+': '+str(exc)[:400])
            if attempt < 3:
                time.sleep(2**(attempt+1))
        partial.unlink(missing_ok=True)
        save(receipt, result)
        return result

    def trial(self, tid, query):
        folder = IMPLANTS[int(query['implantId'])][1]
        result = dict(trial_id=tid, implant=self.selected[tid]['implant'],
                      labels=self.selected[tid]['labels'], akf_path=f'akf/{tid}.akf',
                      akf_sha256=sha(self.root/'akf'/f'{tid}.akf'), assets=[])
        try:
            entries = self.api('orthoload_get_data', **query)
            unique = {(x['type'], x['path']): x for x in entries}
            result['official_attachments'] = list(unique.values())
            for (typ, _), item in sorted(unique.items()):
                if typ in {'AKF', 'SCREEN'}:
                    continue
                if typ == 'AVI' and '/pat_mp4/' in item['path'] and item['path'].endswith('.avi'):
                    candidate = dict(item, type='PATIENT_VIDEO', path=item['path'][:-4]+'.mp4',
                                     name=Path(item['path']).stem+'.mp4', listed_path=item['path'])
                    asset = self.asset(candidate, folder)
                    result['assets'].append(asset)
                    if asset['status']=='downloaded':
                        continue
                result['assets'].append(self.asset(item, folder))
            result['status'] = 'video_downloaded' if any(
                a['status']=='downloaded' and a['official_type'] in VIDEO_TYPES for a in result['assets']) else 'video_unavailable'
        except Exception as exc:
            result.update(status='failed', error=type(exc).__name__+': '+str(exc)[:300])
        save(self.out/'metadata/trials'/f'{tid}.json', result)
        return result

    def run(self, limit=None):
        links, additional = self.catalogue()
        ids = sorted(links)[:limit] if limit else sorted(links)
        results = []
        with ThreadPoolExecutor(max_workers=self.jobs) as pool:
            futures = {pool.submit(self.trial, tid, links[tid]): tid for tid in ids}
            for f in as_completed(futures):
                results.append(f.result())
                if len(results)%25 == 0 or len(results)==len(ids):
                    status = dict(stage='download', completed=len(results), total=len(ids),
                                  statuses=dict(Counter(x['status'] for x in results)),
                                  bytes=sum(a.get('bytes',0) for x in results for a in x['assets'] if a['status']=='downloaded'))
                    save(self.out/'progress.json', status)
                    print(json.dumps(status), flush=True)
        extras = []
        for entry in additional:
            f = entry['file'];query = dict(entry['query'], fileId=f['id'], fileType=f['filetype'])
            assets = self.api('orthoload_get_data', **query)
            folder = IMPLANTS[int(query['implantId'])][1]
            downloaded = [self.asset(x, folder) for x in {(a['type'],a['path']):a for a in assets}.values()
                          if x['type'] not in {'AKF','SCREEN'}]
            extras.append(dict(query=query, attachments=assets, assets=downloaded,
                               association='same official activity group; trial mapping must be checked'))
        results.sort(key=lambda x:x['trial_id'])
        by_implant = {}
        for name, folder in IMPLANTS.values():
            group = [r for r in results if r['implant']==name]
            by_implant[name] = dict(records=len(group), video_downloaded=sum(r['status']=='video_downloaded' for r in group),
                                    statuses=dict(Counter(r['status'] for r in group)))
        assets = {a['url']:a for r in results for a in r['assets']}
        summary = dict(completed_at=stamp(), local_records=len(self.selected), processed=len(results),
                       by_implant=by_implant, asset_statuses=dict(Counter(a['status'] for a in assets.values())),
                       bytes_downloaded=sum(a.get('bytes',0) for a in assets.values() if a['status']=='downloaded'),
                       downloaded_types=dict(Counter(a['official_type'] for a in assets.values() if a['status']=='downloaded')),
                       additional_group_files=len(additional), jobs=self.jobs,
                       kinematics_note='Video and any additional measurements are inventoried by original type. No inferred 3-D motion is claimed.')
        save(self.out/'manifest.json', dict(summary=summary, trials=results, additional=extras))
        save(self.out/'summary.json', summary)
        with (self.out/'trial_video_map.csv').open('w', newline='') as f:
            writer=csv.writer(f);writer.writerow(['trial_id','implant','akf_path','asset_type','asset_status','motion_path','source_url','bytes','sha256'])
            for r in results:
                for a in r['assets']:
                    writer.writerow([r['trial_id'],r['implant'],r['akf_path'],a['official_type'],a['status'],
                                     'motion/'+a['local_path'] if a['status']=='downloaded' else '',a['url'],a.get('bytes',''),a.get('sha256','')])
        print(json.dumps(summary), flush=True)
        return summary
