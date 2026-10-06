"""Explicit transport policy for direct mirrors or authorized official proxy downloads."""
import json
import os
from urllib.parse import urljoin, urlsplit

from .config import ROOT


class DownloadPolicyError(RuntimeError):
    pass


def policy():
    return json.loads((ROOT/'config/downloads.json').read_text())


def direct_environment():
    return {k:v for k,v in os.environ.items()
            if k.lower() not in ('http_proxy','https_proxy','all_proxy')}


def validate_asset_host(url, settings=None):
    settings=settings or policy()
    parsed=urlsplit(url)
    approved=parsed.hostname in settings['asset_hosts'] or (
        settings.get('allow_overseas_asset_redirects',False) and
        any((parsed.hostname or '').endswith(suffix) for suffix in settings.get('asset_host_suffixes',[])))
    if parsed.scheme!='https' or parsed.username or parsed.password or not approved:
        # Do not log signed redirects, query strings or credentials.
        raise DownloadPolicyError(f'Asset transport blocked for unapproved host {parsed.hostname}')


def resolve_asset_url(url, settings=None):
    import requests
    settings=settings or policy()
    with requests.Session() as session:
        session.trust_env=bool(settings.get('proxy_for_assets',False))
        for _ in range(6):
            validate_asset_host(url,settings)
            try:
                response=session.head(url,allow_redirects=False,timeout=(10,20))
            except requests.RequestException as exc:
                raise DownloadPolicyError(f'Direct mirror probe failed for {urlsplit(url).hostname}: {type(exc).__name__}') from None
            if response.status_code in (301,302,303,307,308):
                url=urljoin(url,response.headers['Location'])
                continue
            if not 200<=response.status_code<300:
                raise DownloadPolicyError(f'Direct mirror returned HTTP {response.status_code} for {urlsplit(url).hostname}')
            return url
    raise DownloadPolicyError('Asset mirror redirect limit exceeded')
