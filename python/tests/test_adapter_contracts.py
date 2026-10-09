import inspect
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from motiogen.adapters import Chois,EmptyConstraint,Nvidia,SparseJointConstraint
from motiogen.assets import prepare_local_llama
from motiogen.chois_compat import PureTorchChamferDistance
from motiogen.config import ROOT, load_config
from motiogen.local_llm import FrozenTextEncoder, load_encoder


class AdapterContractTest(unittest.TestCase):
    def test_chois_inference_does_not_construct_training_state(self):
        source=inspect.getsource(Chois.load)
        self.assertNotIn('Adam(',source)
        self.assertNotIn('EMA(',source)
        self.assertIn('opt.batch_size=1',source)
        self.assertIn('class OneSample',source)
        self.assertIn("'optimizer_created':False",source)
        self.assertIn("'ema_model_copy_created':False",source)
        run_source=inspect.getsource(Chois.run)
        self.assertIn("kwargs['num_workers']=0",run_source)

    def test_kimodo_declares_actual_output_skeleton(self):
        spec=load_config()['baselines']['kimodo']
        self.assertEqual(spec['input_skeleton'],'somaskel30')
        self.assertEqual(spec['skeleton'],'somaskel77')

    def test_chois_local_body_models_are_read_only_references(self):
        policy=json.loads((ROOT/'config/local_assets.json').read_text())['chois']
        self.assertIn('read-only',policy['mode'])
        self.assertEqual(set(policy['files']),{
            'SMPLX_MALE.npz','SMPLX_FEMALE.npz','SMPLX_NEUTRAL.npz'})

    def test_chois_chamfer_fallback_matches_squared_nearest_neighbours(self):
        import torch
        x=torch.tensor([[[0.,0.,0.],[2.,0.,0.]]])
        y=torch.tensor([[[1.,0.,0.],[4.,0.,0.],[2.,3.,0.]]])
        x_dist,y_dist,x_idx,y_idx=PureTorchChamferDistance()(x,y)
        self.assertTrue(torch.equal(x_dist,torch.tensor([[1.,1.]])))
        self.assertTrue(torch.equal(y_dist,torch.tensor([[1.,4.,9.]])))
        self.assertTrue(torch.equal(x_idx,torch.tensor([[0,0]])))
        self.assertTrue(torch.equal(y_idx,torch.tensor([[0,1,1]])))

    def test_required_download_links_are_recorded(self):
        sources=json.loads((ROOT/'config/asset_sources.json').read_text())
        self.assertEqual(set(sources),{'motionlcm','omnicontrol','chois','ardy','kimodo','text'})
        for group in sources.values():
            for url in group.values():
                self.assertTrue(url.startswith('https://'))

    def test_llama_31_cannot_masquerade_as_pinned_llama_3(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder)
            policy=json.loads((ROOT/'config/language_models.json').read_text())
            config=dict(policy['expected_config'],max_position_embeddings=131072)
            (base/'config.json').write_text(json.dumps(config))
            (base/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{},'metadata':{'total_size':0}}))
            with patch.dict(os.environ,{'MORIONGEN_LLAMA_BASE_PATH':folder}):
                with self.assertRaisesRegex(ValueError,'Llama 3.1 is not compatible'):
                    prepare_local_llama(load_config())

    def test_llama_3_base_cannot_masquerade_as_instruct(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder)
            policy=json.loads((ROOT/'config/language_models.json').read_text())
            (base/'config.json').write_text(json.dumps(policy['expected_config']))
            (base/'tokenizer_config.json').write_text(json.dumps({'chat_template':None}))
            (base/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{},'metadata':{'total_size':0}}))
            with patch.dict(os.environ,{'MORIONGEN_LLAMA_BASE_PATH':folder}):
                with self.assertRaisesRegex(ValueError,'base model cannot replace'):
                    prepare_local_llama(load_config())

    def test_frozen_text_encoder_is_fail_closed(self):
        import torch
        encoder=FrozenTextEncoder({'walk':torch.zeros(1,1,4096)},device='cpu')
        embedding,lengths=encoder(['walk'])
        self.assertEqual(tuple(embedding.shape),(1,1,4096))
        self.assertEqual(lengths,[1])
        with self.assertRaisesRegex(KeyError,'not encoded'):
            encoder(['run'])

    def test_llama_load_preserves_vram_reserve(self):
        source=inspect.getsource(load_encoder)
        self.assertIn("device_map='auto'",source)
        self.assertIn("0:'10GiB'",source)
        self.assertIn("'cpu':'12GiB'",source)
        self.assertIn("merge_peft=False",source)
        self.assertIn("set_adapter(['default','supervised']",source)
        self.assertIn("types.MethodType",source)

    def test_kimodo_constraints_enable_official_postprocess(self):
        source=inspect.getsource(Nvidia.generate)
        self.assertIn("post_processing=bool(constraints)",source)

    def test_sparse_nvidia_constraint_crops_in_time(self):
        import torch
        constraint=SparseJointConstraint(12,3,torch.ones(3),torch.eye(3),torch.zeros(2))
        moved=constraint.crop_move(10,20)
        self.assertEqual(moved.frame,2)
        data={'global_joints_positions':[],'global_joints_rots':[],'smooth_root_2d':[]}
        index={'global_joints_positions':[],'global_joints_rots':[],'smooth_root_2d':[]}
        moved.update_constraints(data,index)
        self.assertEqual(index['smooth_root_2d'][0].tolist(),[2])
        self.assertEqual(data['smooth_root_2d'][0].tolist(),[[0.0,0.0]])
        self.assertIsInstance(constraint.crop_move(20,30),EmptyConstraint)


if __name__=='__main__':
    unittest.main()
