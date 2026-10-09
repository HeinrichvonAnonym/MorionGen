import types
import unittest

from motiogen.kimodo_compat import load_tmr_with_frozen_raw_encoder


class KimodoCompatibilityTest(unittest.TestCase):
    def test_tmr_injection_preserves_trained_projection(self):
        top=object(); original_raw=object(); frozen=object()
        module=types.SimpleNamespace()

        def instantiate(*args,**kwargs):
            return types.SimpleNamespace(text_encoder=top,raw_text_encoder=original_raw)

        module.instantiate_from_dict=instantiate
        def load_model(**kwargs):
            model=module.instantiate_from_dict()
            # Reproduce the pinned upstream generic assignment.
            model.text_encoder=kwargs['text_encoder']
            return model
        module.load_model=load_model

        model=load_tmr_with_frozen_raw_encoder(module,'TMR-SOMA-RP-v1','cuda',frozen)
        self.assertIs(model.text_encoder,top)
        self.assertIs(model.raw_text_encoder,frozen)
        self.assertIs(module.instantiate_from_dict,instantiate)

    def test_injection_restores_loader_after_failure(self):
        module=types.SimpleNamespace(instantiate_from_dict=lambda:None)
        original=module.instantiate_from_dict
        def fail(**kwargs): raise ValueError('load failed')
        module.load_model=fail
        with self.assertRaisesRegex(ValueError,'load failed'):
            load_tmr_with_frozen_raw_encoder(module,'tmr','cuda',object())
        self.assertIs(module.instantiate_from_dict,original)


if __name__=='__main__': unittest.main()
