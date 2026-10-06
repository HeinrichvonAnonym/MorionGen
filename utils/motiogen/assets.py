import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from .config import ROOT, asset_root
from .runtime import checksum, json_write
from .downloads import direct_environment, policy, resolve_asset_url

# Official download IDs from pinned repositories, never guessed mirrors.
DRIVE = {
    'motionlcm': [('15zFDitcOLhjbQ0CaOoM-QNKQUeyJw-Om','experiments_recons.zip'),
                  ('1U7homKobR2gaDLfL5flS3N0g7e0a_AQd','experiments_t2m.zip'),
                  ('1o6oFdH5dgQJNB5J2rDGKCMw3FDr9gUkW','experiments_control.zip'),
                  ('1Mg_3RnWmRt0tk_lyLRRiOZg1W-Fu4wLL','humanml3d_tiny.zip'),
                  ('16hyR4XlEyksVyNVjhIWK684Lrm_7_pvX','t2m.zip'),
                  ('1J2pTxrar_q689Du5r3jES343fZUmCs_y','smpl_models.zip'),
                  ('1cmXKUT31pqd7_XpJAiWEo1K81TMYHA5n','glove.zip')],
    'omnicontrol': [('1oTkBtArc3xjqkYD6Id7LksrTOn3e1Zud','omnicontrol_ckpt.zip'),('1INYlGA76ak_cKGzvpOV2Pe6RkYTlXTW2','smpl.zip'),('1DSaKqWX2HlwBtVH5l7DdW96jeYUIXsOP','t2m.zip')],
    'chois': [('1ZG-9--RfUWj5oWYnvcONNuRuxaH_Zpw1','processed_data.zip')]
}


def curl_file(url,destination,timeout=600,metadata=False,expected_size=None):
    if not metadata and policy().get('proxy_for_assets'):
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


def text_access(root,name):
    """Probe selected upstream text dependencies without downloading the 8B backbone."""
    records=[]
    repos=['meta-llama/Meta-Llama-3-8B-Instruct','McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp','McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised']
    target=root/'text/access'; target.mkdir(parents=True,exist_ok=True)
    for repo in repos:
        file=target/(repo.replace('/','--')+'.json')
        response=subprocess.run(['curl','-sS','-L','--max-time','30','--max-filesize',str(policy()['official_metadata_max_bytes']),'-w','%{http_code}',
                                 f'https://huggingface.co/{repo}/resolve/main/config.json','-o',str(file)],capture_output=True,text=True)
        # PEFT-only supervised repos may not have config.json: preserve that distinction from gating.
        records.append({'repo':repo,'http_status':response.stdout[-3:],'returncode':response.returncode,
                        'probe':'config.json','body_saved':str(file),'backbone_downloaded':False})
    json_write(root/name/'text_access.json',{'probes':records,'login_command':'conda run -n motiogen-tools huggingface-cli login',
               'model_access_page':'https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct',
               'scope':'public probes do not consume or print local credentials; full text-model access is not established'})
    return records


def safe_extract(archive, target):
    target = Path(target).resolve()
    with zipfile.ZipFile(archive) as source:
        for info in source.infolist():
            candidate = (target / info.filename).resolve()
            if not candidate.is_relative_to(target) or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Unsafe archive member: ' + info.filename)
        free = shutil.disk_usage(target).free
        if sum(info.file_size for info in source.infolist()) > free - 80 * 1024**3:
            raise RuntimeError('Extraction would cross 80 GiB disk reserve')
        source.extractall(target)


def prepare(cfg, baseline, run_dir, part='assets',selected=None):
    run_dir = Path(run_dir)
    root = asset_root(cfg)
    root.mkdir(parents=True, exist_ok=True)
    target = root / baseline
    target.mkdir(exist_ok=True)
    observations = []
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
    for drive_id, filename in DRIVE.get(baseline,[]):
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
            if not archive.is_file():
                curl_file(mirror['url'] if mirror else f'https://drive.google.com/uc?export=download&id={drive_id}',archive)
            if mirror and checksum(archive)!=mirror['sha256']:
                raise RuntimeError('Registered archive checksum mismatch')
            returncode=0
        except Exception as exc:
            observations.append({'file':filename,'status':'missing_resources','reason':str(exc)})
            continue
        if returncode or not archive.is_file() or not zipfile.is_zipfile(archive):
            observations.append({'file':filename,'status':'missing_resources','returncode':returncode,'source':f'https://drive.google.com/file/d/{drive_id}/view'})
            continue
        receipt={'file':filename,'source':f'https://drive.google.com/file/d/{drive_id}/view','sha256':checksum(archive),'bytes':archive.stat().st_size}
        safe_extract(archive,target)
        json_write(extracted,receipt)
        observations.append(receipt)
    if baseline in ('ardy','kimodo') and (not selected or 'model' in selected):
        repo=cfg['baselines'][baseline]['hf_repo']
        try:
            downloaded=curl_snapshot(repo,target/cfg['baselines'][baseline]['model'])
            observations.append({'repo':repo,'path':downloaded,'status':'downloaded'})
        except Exception as exc:
            observations.append({'repo':repo,'status':'missing_resources','reason':str(exc)})
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
            for folder_id,directory in [('1gqw3EHiEMqw1OXqH92Axoc5FJntA_E5x','pretrained_models'),('1hGDYEy91Tk7FC1U_8BouhlSp5RQkEJuY','evaluators')]:
                try:
                    import gdown
                    entries=gdown.download_folder(id=folder_id,output=str(target/directory)+'/',quiet=True,use_cookies=False,skip_download=True)
                    if not entries:
                        raise RuntimeError('Official Drive folder listing unavailable')
                    receipts=[]
                    for entry in entries:
                        path=target/directory/entry.path
                        if not path.resolve().is_relative_to(target.resolve()):
                            raise ValueError('Unsafe Drive folder member')
                        if not path.is_file():
                            curl_file(f'https://drive.google.com/uc?export=download&id={entry.id}',path)
                        receipts.append({'file':str(path.relative_to(target)),'bytes':path.stat().st_size,'sha256':checksum(path)})
                    json_write(target/(directory+'.source_receipt.json'),{'folder_id':folder_id,'files':receipts,'transport':'inherited proxy; official Drive; upstream checksum not published'})
                    observations.append({'kind':directory,'status':'downloaded','files':receipts})
                except Exception as exc:
                    observations.append({'kind':directory,'status':'missing_resources','reason':type(exc).__name__})
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
        if path.is_file() and '.cache' not in path.parts:
            inventory.append({'path':str(path.relative_to(root)),'bytes':path.stat().st_size,
                              'sha256':checksum(path) if path.suffix in ('.pt','.ckpt','.safetensors','.yaml','.json') else None})
    json_write(target/'inventory.json',inventory)
    json_write(run_dir/'asset_observations.json',observations)
    missing=any(x.get('status')=='missing_resources' for x in observations)
    return {'status':'missing_resources' if missing else 'success','reason':'some official asset downloads failed' if missing else None,'observations':observations}


def prepare_local_llama(cfg):
    """Original user-supplied shards are read only; no fine-tuned directory is searched."""
    policy=json.loads((ROOT/'config/language_models.json').read_text())
    base=Path(policy['llama_base_path']); index=json.loads((base/'model.safetensors.index.json').read_text())
    if (base/'adapter_config.json').exists() or base.name in policy['excluded_sibling_directories']:
        raise ValueError('Refuse a fine-tuned or adapter checkpoint as the original base')
    target=asset_root(cfg)/'text/llm2vec'; target.mkdir(parents=True,exist_ok=True)
    virtual=target/'original_base'; virtual.mkdir(exist_ok=True)
    files=[]
    cached_path=target/'local_base_receipt.json'
    cached=json.loads(cached_path.read_text()) if cached_path.exists() else {}
    cached_files={entry['path']:entry for entry in cached.get('files',[])}
    # Only config is copied, to retain upstream instruction-format selection by _name_or_path.
    config=json.loads((base/'config.json').read_text()); config['_name_or_path']=policy['expected_base']
    json_write(virtual/'config.json',config)
    for name in sorted(set(index['weight_map'].values())|{'model.safetensors.index.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json'}):
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
    roots=[asset_root(cfg)/baseline, ROOT/cfg['baselines'][baseline]['repo']]
    # Shared HumanML/evaluator assets may be reused only after exact-file checks.
    if baseline=='omnicontrol':
        roots.append(asset_root(cfg)/'motionlcm')
    for root in roots:
        if root.exists():
            matches=sorted(p for p in root.rglob(name) if '.git' not in p.parts)
            if matches:
                return matches[0]
    return None
