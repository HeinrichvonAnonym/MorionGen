"""Narrow runtime compatibility helpers for the pinned Kimodo checkout."""


def load_tmr_with_frozen_raw_encoder(load_module,modelname,device,frozen):
    """Inject LLM2Vec while preserving TMR's trained top text projection.

    The upstream generic ``text_encoder=`` loader hook assigns to
    ``model.text_encoder``.  That field is the raw encoder in Kimodo, but is a
    trained 4096-to-latent projection in TMR; TMR's raw encoder lives in
    ``model.raw_text_encoder``.  Capture and restore the projection around the
    official load so the pinned checkout can remain byte-for-byte clean.
    """
    original=load_module.instantiate_from_dict; captured={}

    def instantiate(*args,**kwargs):
        model=original(*args,**kwargs)
        if hasattr(model,'raw_text_encoder'):
            captured['top_text_encoder']=model.text_encoder
        return model

    load_module.instantiate_from_dict=instantiate
    try:
        model=load_module.load_model(modelname=modelname,device=device,
            default_family='TMR',text_encoder=frozen)
    finally:
        load_module.instantiate_from_dict=original
    if 'top_text_encoder' not in captured or not hasattr(model,'raw_text_encoder'):
        raise RuntimeError('Loaded model does not expose the expected TMR text encoder contract')
    model.text_encoder=captured['top_text_encoder']
    model.raw_text_encoder=frozen
    return model
