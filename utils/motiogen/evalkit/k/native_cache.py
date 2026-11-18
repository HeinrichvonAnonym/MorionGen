"""Strict reuse of a complete frozen D-only native candidate bank.

This provider exposes every successful candidate, never just the old winner.
Its contract pins the native implementation, configuration and assets. Reference
scores cannot influence reuse, candidate applicability or subsequent selection.
"""
import json
from pathlib import Path
import shutil
import numpy as np
from .config import digest


class FrozenNativeBank:
    def __init__(self,source,contract,settings):
        self.source=Path(source)
        self.receipt=json.loads((self.source/'receipt.json').read_text())
        if self.receipt['version']!='k4-candidate-pool-1':raise ValueError('Expected a completed frozen native bank')
        required={'config','implementation','input_config_sha256','biomech_sha256','model_sha256'}
        if set(contract)!=required:raise ValueError('Declare the complete native code/configuration/asset contract')
        contract={**contract,'config':dict(contract['config'])}
        if 'method' in settings and 'method' not in contract['config']:
            contract['config']['method']=settings['method']
        for key,value in contract.items():
            if self.receipt['key'].get(key)!=value:raise ValueError('Native bank contract mismatch: '+key)
        native_settings={k:v for k,v in settings.items() if k!='coordinated_export_variants'}
        if native_settings!=contract['config']:raise ValueError('Only export composition may change when reusing a frozen bank')
        self.selection=self.receipt['candidate_selection']
        names=set(native_settings['candidates'])
        categories=[set(self.selection.get(k,{})) for k in ('qualities','failures','inapplicable')]
        if set.union(*categories)!=names or sum(map(len,categories))!=len(names):
            raise ValueError('Native candidate accounting is incomplete or overlapping')
        self.files={}
        for name in self.selection['qualities']:
            folder=self.source/'candidates'/name
            child=json.loads((folder/'receipt.json').read_text())
            for key in required-{'config'}:
                if child['key'].get(key)!=contract[key]:raise ValueError('Native candidate code/asset contract mismatch')
            self.files[name]={str(p.relative_to(folder)):digest(p) for p in sorted(folder.rglob('*')) if p.is_file()}
        self.identity={'version':'frozen_native_bank_v1','source_bank':str(self.source.resolve()),
                       'source_receipt_sha256':digest(self.source/'receipt.json'),
                       'contract':contract,'candidate_sha256':self.files,
                       'reuse_scope':'Every saved native candidate; no reference scores or old winner filtering'}

    def __call__(self,name,run,folder,mass,floor,config):
        if name in self.selection['failures']:
            raise RuntimeError('Frozen native candidate failure retained: '+str(self.selection['failures'][name]))
        if name not in self.files:raise ValueError('Requested native candidate was not computed')
        key=self.receipt['key']
        if mass!=key['mass_kg'] or floor!=key['floor_m']:raise ValueError('Native mass/floor contract changed')
        source=self.source/'candidates'/name
        child=json.loads((source/'receipt.json').read_text())
        if child['key']['config']!=config:raise ValueError('Native candidate configuration changed')
        if child['key']['inputs']!=key['inputs']:raise ValueError('Native candidate input identity differs')
        for path,expected in self.files[name].items():
            if digest(source/path)!=expected:raise ValueError('Frozen native artifact changed')
        shutil.copytree(source,folder)
        for path,expected in self.files[name].items():
            if digest(folder/path)!=expected:raise ValueError('Native artifact copy mismatch')
        with np.load(folder/'fit.npz') as z:return {**dict(z),'receipt':child}
