"""Reuse the original local Llama base with the two official LLM2Vec adapters.

MNTP is merged, then the supervised adapter is attached, matching upstream's
LLM2Vec.from_pretrained special-case plus peft_model_name_or_path path. Exact
bfloat16 weights are dispatched across bounded GPU/CPU memory for prompt-only
encoding, then released before the motion model is constructed.
"""
import gc
import importlib
import json
import types
from pathlib import Path

from .config import asset_root


class FrozenTextEncoder:
    """Exact prompt embeddings kept after the 8B encoder is released.

    ARDY/Kimodo only need one embedding per requested prompt.  Holding Llama and
    the motion model on a 16 GiB GPU at the same time is unnecessary and can
    exceed the reviewed budget, so prompt encoding is a separate load stage.
    """
    def __init__(self, encoded, device='cuda'):
        self.encoded=encoded
        self._device=str(device)

    def to(self,device=None,dtype=None):
        if device is not None:
            self._device=str(device)
        return self

    def eval(self):
        return self

    def get_device(self):
        import torch
        return torch.device(self._device)

    def __call__(self,texts):
        import torch
        scalar=isinstance(texts,str)
        keys=[texts] if scalar else list(texts)
        missing=[text for text in keys if text not in self.encoded]
        if missing:
            raise KeyError(f'Prompt was not encoded during the reviewed text stage: {missing}')
        vectors=torch.cat([self.encoded[text] for text in keys],dim=0).to(self._device)
        lengths=[1]*len(keys)
        return (vectors[0],lengths[0]) if scalar else (vectors,lengths)


def load_encoder(cfg,baseline):
    import torch
    root=asset_root(cfg)/'text/llm2vec'
    wrapper=importlib.import_module(baseline+'.model.llm2vec.llm2vec_wrapper')
    policy=json.loads((root/'local_base_receipt.json').read_text())
    supervised=root/policy['supervised_repo']
    extra=json.loads((supervised/'llm2vec_config.json').read_text()) if (supervised/'llm2vec_config.json').exists() else {}
    supported={k:v for k,v in extra.items() if k in ('pooling_mode','max_length','doc_max_length','skip_instruction')}
    # Keep the configured 3 GiB VRAM reserve on a 16 GiB device, including the
    # CUDA context, kernels and the subsequently attached supervised adapter.
    # A nominal 11 GiB weight cap reached 12.61 GiB device-wide on the first
    # smoke, so weights are capped at 10 GiB and overflow is dispatched to RAM.
    # Do not call merge_and_unload on a dispatched model: PEFT temporarily
    # materializes the full merged model on GPU and crossed the reserve even
    # with a 10 GiB dispatch cap.  Activating both adapters simultaneously
    # computes exactly the same MNTP + supervised LoRA delta without that
    # transient allocation (and works across Accelerate's CPU/GPU dispatch).
    # After the finite prompt set is encoded, all 8B weights are released.
    model=wrapper.LLM2Vec.from_pretrained(str(root/'original_base'),str(root/policy['mntp_repo']),merge_peft=False,
            torch_dtype=torch.bfloat16,device_map='auto',max_memory={0:'10GiB','cpu':'12GiB'},
            low_cpu_mem_usage=True,**supported)
    model.model.load_adapter(str(supervised),adapter_name='supervised',low_cpu_mem_usage=True)
    # PeftModel.set_adapter exposes the single-adapter compatibility API, while
    # its BaseTuner supports an exact list of simultaneously active adapters.
    model.model.base_model.set_adapter(['default','supervised'],inference_mode=True)
    # Upstream encode() unconditionally calls self.to(device), which is invalid
    # for an Accelerate-dispatched model containing offload/meta placeholders.
    # Inputs are still moved to cuda by _encode and dispatch hooks move each
    # layer as needed; keep the reviewed device map intact.
    model.to=types.MethodType(lambda current,*args,**kwargs: current,model)
    encoder=wrapper.LLM2VecEncoder.__new__(wrapper.LLM2VecEncoder)
    encoder.model=model; encoder.llm_dim=4096; encoder._device='cuda:0'
    encoder.model.eval()
    for parameter in encoder.model.parameters():
        parameter.requires_grad=False
    return encoder


def precompute_encoder(cfg,baseline,prompts,cache_path=None):
    """Encode the finite scenario prompt set, then free all Llama weights."""
    import torch
    unique=list(dict.fromkeys(prompts)); encoded={}; cache=Path(cache_path) if cache_path else None
    if cache and cache.is_file():
        payload=torch.load(cache,map_location='cpu',weights_only=True)
        if payload.get('schema_version')==1 and payload.get('baseline')==baseline:
            encoded={str(key):value for key,value in payload.get('encoded',{}).items()}
    missing=[prompt for prompt in unique if prompt not in encoded]
    if not missing:
        return FrozenTextEncoder(encoded,'cuda')
    encoder=load_encoder(cfg,baseline)
    with torch.no_grad():
        for ordinal,prompt in enumerate(missing,1):
            tensor,lengths=encoder([prompt])
            if lengths!=[1] or tensor.shape!=(1,1,4096):
                raise RuntimeError(f'Unexpected LLM2Vec output contract: {tensor.shape}, {lengths}')
            encoded[prompt]=tensor.detach().cpu()
            if ordinal==1 or ordinal%25==0 or ordinal==len(missing):
                print(json.dumps({'stage':'text_precompute','completed':ordinal,'total_missing':len(missing),
                                  'cached':len(encoded)-ordinal}),flush=True)
    del encoder
    gc.collect()
    torch.cuda.empty_cache()
    if cache:
        cache.parent.mkdir(parents=True,exist_ok=True)
        temporary=cache.with_suffix(cache.suffix+'.tmp')
        torch.save({'schema_version':1,'baseline':baseline,'encoded':encoded},temporary)
        temporary.replace(cache)
    return FrozenTextEncoder(encoded,'cuda')
