import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

from .config import ROOT, asset_root, data_root, repo_root
from .runtime import checksum, json_write
from .downloads import direct_environment, policy, resolve_asset_url

def drive_sources():
    sources=json.loads((ROOT/'config/asset_sources.json').read_text())
    result={}
    for baseline in ('motionlcm','omnicontrol','chois'):
        result[baseline]=[]
        for filename,url in sources[baseline].items():
            if filename.endswith(('.zip','.tar.gz','.tgz')) and '/file/d/' in url:
                result[baseline].append((url.split('/file/d/',1)[1].split('/',1)[0],filename))
    return result


def curl_file(url,destination,timeout=600,metadata=False,expected_size=None):
    if not metadata and policy().get('proxy_for_assets'):
        if 'drive.google.com/' in url:
            # Large official checkpoints routinely exceed the generic curl
            # timeout; the process-tree watchdog still bounds the whole run.
            return gdown_file(url,destination,policy()['download_timeout_seconds'])
        from .transfer import proxy_file
        return proxy_file(url,destination,expected_size)
    destination=Path(destination); destination.parent.mkdir(parents=True,exist_ok=True)
    partial=destination.with_name(destination.name+'.part')
    # Only small official metadata may inherit the proxy. Asset redirects are
    # checked without downloading a body, and curl cannot follow a new redirect.
    resolved=url if metadata else resolve_asset_url(url)
    command=['curl','--fail','--silent','--show-error','--connect-timeout','20',
             '--max-time',str(timeout),'--continue-at','-','--output',str(partial),url]
    command[-1]=resolved
    if metadata:
        command[1:1]=['--location','--max-filesize',str(policy()['official_metadata_max_bytes'])]
    else:
        command[1:1]=['--noproxy','*']
    for attempt in range(3):
        result=subprocess.run(command+['--write-out','%{http_code}'],env=None if metadata else direct_environment(),capture_output=True,text=True)
        if result.stderr:
            print(result.stderr,file=sys.stderr,end='')
        if not result.returncode and result.stdout[-3:] not in ('200','206'):
            raise RuntimeError('Asset response changed after mirror probe; redirect/body not accepted')
        if result.returncode not in (35,60,33,28,18,56):
            break
        if result.returncode==33:
            # Preserve interrupted work even if this endpoint cannot resume.
            if partial.exists():
                partial.replace(partial.with_name(partial.name+'.resume_unsupported'))
    if result.returncode:
        raise RuntimeError(f'curl failed with exit {result.returncode} for public asset; see transport policy')
    partial.replace(destination)
    return destination


def gdown_file(url,destination,timeout=259200):
    """Use gdown's cookie-aware public Drive flow without exposing signed URLs."""
    destination=Path(destination); destination.parent.mkdir(parents=True,exist_ok=True)
    partial=destination.with_name(destination.name+'.gdown.part')
    resumable=destination.with_name(destination.name+'.part')
    if not partial.exists():
        candidates=list(destination.parent.glob(destination.name+'.gdown.part*.part'))
        if candidates:
            max(candidates,key=lambda path:path.stat().st_size).replace(partial)
    if resumable.is_file() and not partial.exists():
        resumable.replace(partial)
    tools=data_root(json.loads((ROOT/'config/baselines.json').read_text()))/'envs/motiogen-tools/bin/python'
    if not tools.is_file():
        raise RuntimeError('Drive confirmation requires the external motiogen-tools environment')
    environment=os.environ.copy()
    environment['TMPDIR']=str(tools.parents[3]/'package-cache/tmp')
    Path(environment['TMPDIR']).mkdir(parents=True,exist_ok=True)
    command=[str(tools),'-m','gdown',url,'--output',str(partial),'--no-cookies','--fuzzy']
    result=subprocess.run(command+['--continue'],capture_output=True,text=True,timeout=timeout,env=environment)
    valid=partial.is_file() and is_archive(partial,destination.name)
    if (result.returncode or not valid) and partial.is_file():
        partial.replace(partial.with_name(partial.name+'.resume_failed_'+str(time.time_ns())))
        result=subprocess.run(command,capture_output=True,text=True,timeout=timeout,env=environment)
    valid=partial.is_file() and is_archive(partial,destination.name)
    if result.returncode or not valid:
        if partial.is_file():
            partial.replace(partial.with_name(partial.name+'.invalid_'+str(time.time_ns())))
        raise RuntimeError('Official Drive download unavailable through cookie-aware public flow')
    partial.replace(destination)
    return destination


def curl_snapshot(repo,target,repo_type='model',patterns=None):
    import fnmatch
    target=Path(target); target.mkdir(parents=True,exist_ok=True)
    metadata=target/'source_metadata.json'
    curl_file(f'https://huggingface.co/api/{repo_type}s/{repo}?blobs=true',metadata,60,metadata=True)
    info=json.loads(metadata.read_text()); revision=info['sha']; receipts=[]
    prefix=policy()['hf_mirror'].rstrip('/')+'/' + ('datasets/' if repo_type=='dataset' else '') + repo
    siblings={entry['rfilename'] for entry in info['siblings']}
    for entry in info['siblings']:
        name=entry['rfilename']
        if name=='pytorch_model.bin' and 'model.safetensors' in siblings:
            continue
        if name=='.gitattributes' or (patterns and not any(fnmatch.fnmatch(name,p) for p in patterns)):
            continue
        path=target/name
        if not path.is_file():
            curl_file(f'{prefix}/resolve/{revision}/{name}?download=true',path,expected_size=(entry.get('lfs') or {}).get('size'))
        digest=checksum(path)
        lfs=entry.get('lfs') or {}
        expected=lfs.get('sha256')
        if expected and (digest!=expected or path.stat().st_size!=lfs['size']):
            raise RuntimeError(f'Official LFS checksum/size mismatch for {name}; no receipt issued')
        if path.stat().st_size>=64*1024**2 and not expected:
            raise RuntimeError(f'No official SHA-256 for large asset {name}; no receipt issued')
        receipts.append({'file':name,'bytes':path.stat().st_size,'sha256':digest,'official_sha256':expected})
    transport='inherited proxy; official HTTPS; official LFS checksum verification' if policy().get('proxy_for_assets') else 'direct approved mirror; TLS; official LFS checksum verification; no overseas redirects'
    json_write(target/'source_receipt.json',{'repo':repo,'revision':revision,'files':receipts,'transport':transport})
    return str(target)


def git_dataset_snapshot(repo,target):
    """Clone metadata-heavy HF datasets without asking the multi-megabyte API listing."""
    target=Path(target)
    if target.exists() and any(target.iterdir()) and not (target/'.git').is_dir():
        raise RuntimeError('Benchmark metadata target is non-empty and is not a verified checkout')
    if not (target/'.git').is_dir():
        target.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['git','clone','--depth','1',f'https://huggingface.co/datasets/{repo}',str(target)],check=True)
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=target,text=True).strip()
    status=subprocess.check_output(['git','status','--porcelain'],cwd=target,text=True).strip()
    if status:
        raise RuntimeError('Benchmark metadata checkout is dirty')
    files=[]
    for path in target.rglob('*'):
        if path.is_file() and '.git' not in path.parts:
            files.append({'file':str(path.relative_to(target)),'bytes':path.stat().st_size,'sha256':checksum(path)})
    json_write(target.with_name(target.name+'.source_receipt.json'),{
        'repo':repo,'revision':revision,'files':files,
        'transport':'official Hugging Face dataset git checkout; no LFS motion dataset requested'})
    return str(target)


def text_access(root,name):
    """Probe selected upstream text dependencies without downloading the 8B backbone."""
    records=[]
    repos=[('meta-llama/Meta-Llama-3-8B-Instruct','config.json'),
           ('McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp','config.json'),
           ('McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised','adapter_config.json')]
    target=root/'text/access'; target.mkdir(parents=True,exist_ok=True)
    for repo,probe in repos:
        file=target/(repo.replace('/','--')+'.json')
        response=subprocess.run(['curl','-sS','-L','--max-time','30','--max-filesize',str(policy()['official_metadata_max_bytes']),'-w','%{http_code}',
                                 f'https://huggingface.co/{repo}/resolve/main/{probe}','-o',str(file)],capture_output=True,text=True)
        records.append({'repo':repo,'http_status':response.stdout[-3:],'returncode':response.returncode,
                        'probe':probe,'body_saved':str(file),'backbone_downloaded':False})
    json_write(root/name/'text_access.json',{'probes':records,'login_command':'conda run -n motiogen-tools huggingface-cli login',
               'model_access_page':'https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct',
               'scope':'public probes do not consume or print local credentials; full text-model access is not established'})
    return records


def is_archive(path,name=None):
    """Validate a configured archive by its logical filename, not temp suffix."""
    logical=(name or Path(path).name).lower()
    if logical.endswith('.zip'):
        return zipfile.is_zipfile(path)
    if logical.endswith(('.tar.gz','.tgz')):
        return tarfile.is_tarfile(path)
    return True


def _safe_member(target,name,preserve_existing):
    raw=target/name
    if name in preserve_existing and raw.is_symlink():
        return None
    candidate=raw.resolve()
    if not candidate.is_relative_to(target):
        raise ValueError('Unsafe archive member: '+name)
    current=target/name
    while current!=target:
        if current.is_symlink():
            raise ValueError('Archive member crosses existing symlink: '+name)
        current=current.parent
    return candidate


def safe_extract(archive, target, preserve_existing=()):
    target = Path(target).resolve()
    preserve_existing=set(preserve_existing)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as source:
            members=[]
            for info in source.infolist():
                if _safe_member(target,info.filename,preserve_existing) is None:
                    continue
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('Unsafe archive member: ' + info.filename)
                members.append(info)
            expanded=sum(info.file_size for info in members)
            if expanded > shutil.disk_usage(target).free - 80 * 1024**3:
                raise RuntimeError('Extraction would cross 80 GiB disk reserve')
            source.extractall(target,members=members)
        return
    with tarfile.open(archive,'r:*') as source:
        members=[]
        for info in source.getmembers():
            if _safe_member(target,info.name,preserve_existing) is None:
                continue
            if info.issym() or info.islnk() or info.isdev():
                raise ValueError('Unsafe archive member: '+info.name)
            members.append(info)
        if sum(info.size for info in members) > shutil.disk_usage(target).free - 80 * 1024**3:
            raise RuntimeError('Extraction would cross 80 GiB disk reserve')
        source.extractall(target,members=members,filter='data')


def prepare(cfg, baseline, run_dir, part='assets',selected=None):
    run_dir = Path(run_dir)
    root = asset_root(cfg)
    root.mkdir(parents=True, exist_ok=True)
    target = root / baseline
    target.mkdir(exist_ok=True)
    observations = []
    if baseline=='chois':
        local_policy=json.loads((ROOT/'config/local_assets.json').read_text())['chois']
        source=Path(local_policy['smplx_source_directory']).resolve()
        body_target=target/'processed_data/smpl_all_models/smplx'
        body_target.mkdir(parents=True,exist_ok=True)
        files=[]
        for name in local_policy['files']:
            original=source/name
            link=body_target/name
            if not original.is_file():
                observations.append({'kind':'SMPL-X','status':'missing_resources','file':str(original),
                    'reason':'configured licensed local body model is missing'})
                continue
            if link.is_symlink() and link.resolve()!=original:
                link.unlink()
            if not link.exists():
                link.symlink_to(original)
            files.append({'file':name,'source':str(original),'bytes':original.stat().st_size,
                          'sha256':checksum(original),'link':str(link)})
        if len(files)==len(local_policy['files']):
            json_write(target/'smplx.local_receipt.json',{'mode':local_policy['mode'],'files':files})
            observations.append({'kind':'SMPL-X','status':'available','receipt':str(target/'smplx.local_receipt.json')})
    if part in ('source','all'):
        try:
            from .sources import prepare_source
            observations.append({'kind':'source','status':'available',**prepare_source(cfg,baseline)})
        except Exception as exc:
            observations.append({'kind':'source','status':'missing_resources','reason':f'{type(exc).__name__}: {exc}'})
            return {'status':'missing_resources','reason':'official source checkout failed','observations':observations}
    if part == 'source':
        json_write(run_dir/'asset_observations.json',observations)
        return {'status':'success','observations':observations}
    if baseline in ('ardy','kimodo'):
        observations.extend(text_access(root,baseline))
    if part=='text-access':
        return {'status':'success','observations':observations,'scope':'read-only public access probes'}
    if part in ('environment','all'):
        result = subprocess.run(['bash', str(ROOT/'scripts/setup_baselines.sh'), baseline], cwd=ROOT)
        observations.append({'kind':'environment','returncode':result.returncode})
        if result.returncode:
            return {'status':'failed','reason':'environment installation failed; see stderr/stdout','observations':observations}
    if part == 'environment':
        return {'status':'success','observations':observations}
    for drive_id, filename in drive_sources().get(baseline,[]):
        if selected and filename not in selected:
            continue
        archive = target / filename
        extracted = target / (filename + '.extracted.json')
        if extracted.exists():
            observations.append({'file':filename,'status':'cached','receipt':str(extracted)})
            continue
        mirror=policy()['archive_mirrors'].get(filename)
        if not mirror and not policy().get('proxy_for_assets'):
            observations.append({'file':filename,'status':'missing_resources','reason':'Google Drive download paused: no checksum-pinned mainland mirror registered; proxy fallback prohibited','source':f'https://drive.google.com/file/d/{drive_id}/view'})
            continue
        try:
            if archive.is_file() and not is_archive(archive,filename):
                canonical=archive.with_name(archive.name+'.gdown.part')
                if canonical.exists():
                    archive.replace(archive.with_name(archive.name+'.invalid_'+str(time.time_ns())))
                else:
                    archive.replace(canonical)
            if not archive.is_file():
                curl_file(mirror['url'] if mirror else f'https://drive.google.com/uc?export=download&id={drive_id}',archive)
            if mirror and checksum(archive)!=mirror['sha256']:
                raise RuntimeError('Registered archive checksum mismatch')
            returncode=0
        except Exception as exc:
            observations.append({'file':filename,'status':'missing_resources','reason':str(exc)})
            continue
        if returncode or not archive.is_file() or not is_archive(archive,filename):
            observations.append({'file':filename,'status':'missing_resources','returncode':returncode,'source':f'https://drive.google.com/file/d/{drive_id}/view'})
            continue
        receipt={'file':filename,'source':f'https://drive.google.com/file/d/{drive_id}/view','sha256':checksum(archive),'bytes':archive.stat().st_size}
        preserve=()
        if baseline=='chois':
            preserve=tuple('processed_data/smpl_all_models/smplx/'+name for name in
                           json.loads((ROOT/'config/local_assets.json').read_text())['chois']['files'])
        safe_extract(archive,target,preserve_existing=preserve)
        json_write(extracted,receipt)
        observations.append(receipt)
    if baseline in ('ardy','kimodo') and (not selected or 'model' in selected):
        repo=cfg['baselines'][baseline]['hf_repo']
        try:
            downloaded=curl_snapshot(repo,target/cfg['baselines'][baseline]['model'])
            observations.append({'repo':repo,'path':downloaded,'status':'downloaded'})
        except Exception as exc:
            observations.append({'repo':repo,'status':'missing_resources','reason':str(exc)})
    if baseline=='kimodo' and (not selected or 'tmr' in selected):
        try:
            repo='nvidia/TMR-SOMA-RP-v1'
            downloaded=curl_snapshot(repo,target/'TMR-SOMA-RP-v1')
            observations.append({'repo':repo,'path':downloaded,'status':'downloaded'})
        except Exception as exc:
            observations.append({'repo':'nvidia/TMR-SOMA-RP-v1','status':'missing_resources','reason':str(exc)})
    if baseline=='kimodo' and (not selected or 'benchmark-metadata' in selected):
        try:
            repo='nvidia/Kimodo-Motion-Gen-Benchmark'
            downloaded=git_dataset_snapshot(repo,target/'benchmark_metadata')
            observations.append({'repo':repo,'path':downloaded,'status':'downloaded',
                'scope':'benchmark metadata only; BONES-SEED motion data is not downloaded'})
        except Exception as exc:
            observations.append({'repo':'nvidia/Kimodo-Motion-Gen-Benchmark','status':'missing_resources','reason':str(exc)})
    if baseline=='motionlcm' and (not selected or 'text' in selected):
        try:
            path=curl_snapshot('sentence-transformers/sentence-t5-large',root/'text/sentence-t5-large',
                               patterns=['*.json','*.txt','*.model','*.safetensors','pytorch_model.bin','1_Pooling/*','2_Dense/*'])
            observations.append({'repo':'sentence-transformers/sentence-t5-large','path':path,'status':'downloaded'})
        except Exception as exc:
            observations.append({'repo':'sentence-transformers/sentence-t5-large','status':'missing_resources','reason':str(exc)})
    if baseline in ('ardy','kimodo') and (not selected or 'local-text' in selected):
        observations.extend(prepare_local_llama(cfg))
    # Folder download: checkpoints only. Dataset/body-model access is checked separately.
    if baseline=='chois' and (not selected or 'checkpoint' in selected):
        if policy().get('proxy_for_assets'):
            source_policy=json.loads((ROOT/'config/asset_sources.json').read_text())['chois']
            folders=[(source_policy[directory].rstrip('/').rsplit('/',1)[-1],directory)
                     for directory in ('pretrained_models','evaluators')]
            for folder_id,directory in folders:
                try:
                    tools=data_root(cfg)/'envs/motiogen-tools/bin/python'
                    folder=target/directory
                    folder.mkdir(parents=True,exist_ok=True)
                    environment=os.environ.copy(); environment['TMPDIR']=str(data_root(cfg)/'package-cache/tmp')
                    Path(environment['TMPDIR']).mkdir(parents=True,exist_ok=True)
                    result=subprocess.run([str(tools),'-m','gdown','--folder',
                        f'https://drive.google.com/drive/folders/{folder_id}','--output',str(folder),
                        '--no-cookies','--continue','--remaining-ok'],capture_output=True,text=True,timeout=policy()['download_timeout_seconds'],env=environment)
                    if result.returncode:
                        raise RuntimeError('Official Drive folder download unavailable')
                    receipts=[]
                    for path in folder.rglob('*'):
                        if path.is_file():
                            if not path.resolve().is_relative_to(target.resolve()):
                                raise ValueError('Unsafe Drive folder member')
                            receipts.append({'file':str(path.relative_to(target)),'bytes':path.stat().st_size,'sha256':checksum(path)})
                    if not receipts:
                        raise RuntimeError('Official Drive folder listing unavailable')
                    json_write(target/(directory+'.source_receipt.json'),{'folder_id':folder_id,'files':receipts,'transport':'inherited proxy; official Drive; upstream checksum not published'})
                    observations.append({'kind':directory,'status':'downloaded','files':receipts})
                except Exception as exc:
                    observations.append({'kind':directory,'status':'missing_resources',
                        'reason':'Official Drive folder incomplete or quota-limited',
                        'error_type':type(exc).__name__})
        else:
            observations.append({'kind':'checkpoints','status':'missing_resources','reason':'Google Drive folder paused: no checksum-pinned mainland mirror registered; proxy fallback prohibited'})
    if baseline=='omnicontrol' and (not selected or 'text' in selected):
        try:
            digest='40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
            path=target/'ViT-B-32.pt'
            if not path.is_file():
                curl_file(f'https://openaipublic.azureedge.net/clip/models/{digest}/ViT-B-32.pt',path)
            if checksum(path)!=digest:
                raise RuntimeError('Official CLIP checksum mismatch')
            json_write(target/'clip.source_receipt.json',{'sha256':digest,'file':str(path),'source':'OpenAI CLIP official ViT-B-32'})
            observations.append({'kind':'CLIP','status':'downloaded'})
        except Exception as exc:
            observations.append({'kind':'CLIP','status':'missing_resources','reason':str(exc)})
    inventory=[]
    for path in target.rglob('*'):
        if path.is_file() and '.cache' not in path.parts and '.git' not in path.parts:
            inventory.append({'path':str(path.relative_to(root)),'bytes':path.stat().st_size,
                              'sha256':checksum(path) if path.suffix in ('.pt','.ckpt','.safetensors','.yaml','.json') else None})
    json_write(target/'inventory.json',inventory)
    json_write(run_dir/'asset_observations.json',observations)
    missing=any(x.get('status')=='missing_resources' for x in observations)
    return {'status':'missing_resources' if missing else 'success','reason':'some official asset downloads failed' if missing else None,'observations':observations}


def prepare_local_llama(cfg):
    """Original user-supplied shards are read only; no fine-tuned directory is searched."""
    policy=json.loads((ROOT/'config/language_models.json').read_text())
    configured=os.environ.get(policy.get('llama_base_environment','MORIONGEN_LLAMA_BASE_PATH')) or policy.get('llama_base_path')
    if not configured:
        raise FileNotFoundError('Set MORIONGEN_LLAMA_BASE_PATH to the user-authorized original Meta-Llama-3-8B-Instruct directory')
    base=Path(configured).expanduser().resolve()
    index_path=base/'model.safetensors.index.json'
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    index=json.loads(index_path.read_text())
    if (base/'adapter_config.json').exists() or base.name in policy['excluded_sibling_directories']:
        raise ValueError('Refuse a fine-tuned or adapter checkpoint as the original base')
    config_path=base/'config.json'
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    original_config=json.loads(config_path.read_text())
    mismatches={key:{'expected':expected,'actual':original_config.get(key)}
                for key,expected in policy['expected_config'].items()
                if original_config.get(key)!=expected}
    if mismatches:
        raise ValueError(
            f'Local Llama config is not the required {policy["expected_base"]}: {mismatches}. '
            'Llama 3.1 is not compatible with the pinned LLM2Vec Llama 3 adapters.')
    tokenizer_config_path=base/'tokenizer_config.json'
    if not tokenizer_config_path.is_file():
        raise FileNotFoundError(tokenizer_config_path)
    tokenizer_config=json.loads(tokenizer_config_path.read_text())
    if not tokenizer_config.get('chat_template'):
        raise ValueError(
            f'Local Llama tokenizer is not {policy["expected_base"]}: the Instruct chat_template is missing. '
            'The Meta-Llama-3-8B base model cannot replace Meta-Llama-3-8B-Instruct.')
    target=asset_root(cfg)/'text/llm2vec'; target.mkdir(parents=True,exist_ok=True)
    virtual=target/'original_base'; virtual.mkdir(exist_ok=True)
    files=[]
    cached_path=target/'local_base_receipt.json'
    cached=json.loads(cached_path.read_text()) if cached_path.exists() else {}
    cached_files={entry['path']:entry for entry in cached.get('files',[])}
    # Only config is copied, to retain upstream instruction-format selection by _name_or_path.
    config=dict(original_config); config['_name_or_path']=policy['expected_base']
    json_write(virtual/'config.json',config)
    for name in sorted(set(index['weight_map'].values())|{'config.json','model.safetensors.index.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json'}):
        source=base/name
        if not source.is_file():
            raise FileNotFoundError(source)
        link=virtual/name
        if not link.exists():
            link.symlink_to(source)
        stat=source.stat(); previous=cached_files.get(str(source),{})
        # Old receipts have no mtime: perform one further verification then retain
        # stat identity. A changed source always forces a fresh content checksum.
        unchanged=(previous.get('bytes')==stat.st_size and previous.get('mtime_ns')==stat.st_mtime_ns)
        files.append({'path':str(source),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,
                      'sha256':previous['sha256'] if unchanged else checksum(source)})
    receipt={**policy,'files':files,'weight_bytes':index['metadata']['total_size'],'base_readonly':True,
             'config_patch':'copy only: _name_or_path set to expected original model for upstream token formatting'}
    json_write(target/'local_base_receipt.json',receipt)
    records=[{'kind':'local_original_base','status':'available','receipt':str(target/'local_base_receipt.json')}]
    for repo in (policy['mntp_repo'],policy['supervised_repo']):
        try:
            directory=curl_snapshot(repo,target/repo)
            records.append({'repo':repo,'path':directory,'status':'downloaded'})
        except Exception as exc:
            records.append({'repo':repo,'status':'missing_resources','reason':str(exc)})
    return records


def find_asset(cfg, baseline, name):
    roots=[asset_root(cfg)/baseline, repo_root(cfg,baseline)]
    # Shared HumanML/evaluator assets may be reused only after exact-file checks.
    if baseline=='omnicontrol':
        roots.append(asset_root(cfg)/'motionlcm')
    for root in roots:
        if root.exists():
            matches=sorted(p for p in root.rglob(name) if '.git' not in p.parts)
            if matches:
                return matches[0]
    return None
