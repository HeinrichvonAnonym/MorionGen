"""Reuse the original local Llama base with the two official LLM2Vec adapters.

The only loading change is device_map={'': 'cuda'}: avoid full CPU residency.
MNTP is merged, then the supervised adapter is attached, matching upstream's
LLM2Vec.from_pretrained special-case plus peft_model_name_or_path path.
"""
import importlib
import json

from .config import asset_root


def load_encoder(cfg,baseline):
    import torch
    from peft import PeftModel
    root=asset_root(cfg)/'text/llm2vec'
    wrapper=importlib.import_module(baseline+'.model.llm2vec.llm2vec_wrapper')
    policy=json.loads((root/'local_base_receipt.json').read_text())
    supervised=root/policy['supervised_repo']
    extra=json.loads((supervised/'llm2vec_config.json').read_text()) if (supervised/'llm2vec_config.json').exists() else {}
    supported={k:v for k,v in extra.items() if k in ('pooling_mode','max_length','doc_max_length','skip_instruction')}
    model=wrapper.LLM2Vec.from_pretrained(str(root/'original_base'),str(root/policy['mntp_repo']),merge_peft=True,
            torch_dtype=torch.bfloat16,device_map={'':'cuda'},low_cpu_mem_usage=True,**supported)
    model.model=PeftModel.from_pretrained(model.model,str(supervised))
    encoder=wrapper.LLM2VecEncoder.__new__(wrapper.LLM2VecEncoder)
    encoder.model=model; encoder.llm_dim=4096; encoder._device='cuda'
    encoder.model.eval()
    for parameter in encoder.model.parameters():
        parameter.requires_grad=False
    return encoder
