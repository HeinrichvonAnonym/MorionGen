"""Resumable official transfers; progress and errors never contain signed URLs."""
import importlib
import re
import shutil
import time
from pathlib import Path
from urllib.parse import urljoin

import requests

from .downloads import policy, validate_asset_host
from .runtime import json_write


def _get(session, url, headers):
    for _ in range(12):
        validate_asset_host(url)
        response = session.get(url, headers=headers, stream=True, allow_redirects=False, timeout=(30, 120))
        if response.status_code in (301, 302, 303, 307, 308):
            url = urljoin(url, response.headers['Location'])
            response.close()
            continue
        if 'text/html' in response.headers.get('Content-Type', '') and 'google.com' in url:
            # Public Drive virus-scan confirmation, never bypass authentication.
            chunks = []; size = 0
            for chunk in response.iter_content(65536):
                chunks.append(chunk); size += len(chunk)
                if size > 2 * 1024**2:
                    response.close(); raise RuntimeError('Drive confirmation exceeded metadata limit')
            response.close()
            try:
                url = importlib.import_module('gdown.download').get_url_from_gdrive_confirmation(b''.join(chunks).decode('utf-8'))
            except Exception:
                raise RuntimeError('Drive public download unavailable: permission, quota or confirmation failure') from None
            continue
        return response
    raise RuntimeError('Official download redirect/confirmation limit exceeded')


def proxy_file(url, destination, expected_size=None):
    destination = Path(destination); destination.parent.mkdir(parents=True, exist_ok=True)
    progress = destination.with_name(destination.name + '.download.json')
    partial = destination.with_name(destination.name + '.part')
    # gdown used random partial names. Retain every other partial and resume the largest.
    if not partial.exists():
        candidates = list(destination.parent.glob(destination.name + '*.part'))
        if candidates:
            max(candidates, key=lambda p: p.stat().st_size).replace(partial)
    settings = policy(); reserve = settings.get('disk_reserve_gib', 80) * 1024**3
    start = time.monotonic(); last_error = None
    for attempt in range(3):
        offset = partial.stat().st_size if partial.exists() else 0
        json_write(progress, {'status': 'connecting', 'bytes': offset, 'expected_bytes': expected_size,
                              'transport': 'inherited proxy; official HTTPS', 'attempt': attempt + 1})
        try:
            with requests.Session() as session:
                session.trust_env = True
                response = _get(session, url, {'Range': f'bytes={offset}-'} if offset else {})
                with response:
                    if response.status_code == 416 and expected_size and offset == expected_size:
                        partial.replace(destination)
                        json_write(progress, {'status': 'downloaded', 'bytes': offset, 'expected_bytes': expected_size})
                        return destination
                    if response.status_code not in (200, 206):
                        raise RuntimeError(f'Official download HTTP {response.status_code}')
                    content_range = response.headers.get('Content-Range', '')
                    if offset:
                        match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+|\*)', content_range)
                        if response.status_code != 206 or not match or int(match[1]) != offset:
                            # Never append a complete HTTP 200 response to a partial file.
                            if response.status_code == 200:
                                partial.replace(partial.with_name(partial.name + '.range_ignored_' + str(time.time_ns())))
                                offset = 0
                            else:
                                raise RuntimeError('Invalid resume Content-Range')
                    content_length = response.headers.get('Content-Length')
                    total = offset + int(content_length) if content_length else expected_size
                    if expected_size and total and expected_size != total:
                        raise RuntimeError('Official file size does not match pinned metadata')
                    if total and total - offset > shutil.disk_usage(destination.parent).free - reserve:
                        raise RuntimeError('Download would cross disk reserve')
                    written = offset; last = 0
                    with partial.open('ab' if offset else 'wb') as handle:
                        for chunk in response.iter_content(1024**2):
                            if time.monotonic() - start > settings.get('download_timeout_seconds', 21600):
                                raise RuntimeError('Official transfer timeout')
                            if shutil.disk_usage(destination.parent).free < reserve + len(chunk):
                                raise RuntimeError('Disk reserve crossed during download')
                            handle.write(chunk); written += len(chunk)
                            if time.monotonic() - last > 10:
                                json_write(progress, {'status': 'downloading', 'bytes': written, 'expected_bytes': total,
                                                      'elapsed_seconds': time.monotonic() - start,
                                                      'transport': 'inherited proxy; official HTTPS'})
                                last = time.monotonic()
                    if total and written != total:
                        raise RuntimeError('Incomplete official response body')
                    partial.replace(destination)
                    json_write(progress, {'status': 'downloaded', 'bytes': written, 'expected_bytes': total,
                                          'elapsed_seconds': time.monotonic() - start,
                                          'transport': 'inherited proxy; official HTTPS'})
                    return destination
        except (requests.RequestException, RuntimeError) as exc:
            # Request exceptions may include signed URLs and proxy credentials.
            last_error = type(exc).__name__ if isinstance(exc, requests.RequestException) else str(exc)
            json_write(progress, {'status': 'retrying' if attempt < 2 else 'failed', 'reason': last_error,
                                  'bytes': partial.stat().st_size if partial.exists() else 0, 'expected_bytes': expected_size})
    raise RuntimeError(last_error)
