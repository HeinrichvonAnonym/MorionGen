import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from motiogen.datasets import (KIMODO_GROUP_COUNT,archive_format,bvh_members,discover_groups,
    kit21_motion,select_stratified,selection_sha,validate_kit)
from motiogen.stratified import _postprocess_subset


class DatasetPreparationTest(unittest.TestCase):
    def make_suite(self, root, cases=32):
        for group in range(KIMODO_GROUP_COUNT):
            for case in range(cases):
                folder=Path(root)/('content' if group<29 else 'repetition')/f'group_{group:02d}'/f'{case:04d}'
                folder.mkdir(parents=True)
                (folder/'meta.json').write_text(json.dumps({'duration':5,'seed':case}))
                (folder/'seed_motion.json').write_text(json.dumps({'bvh_path':f'BVH/240101/m{group}_{case}.bvh'}))

    def test_stratified_selection_is_exact_and_reproducible(self):
        with tempfile.TemporaryDirectory() as folder:
            self.make_suite(folder,35)
            a=select_stratified(folder,32,20261008); b=select_stratified(folder,32,20261008)
            self.assertEqual(a,b)
            self.assertEqual(len(a),KIMODO_GROUP_COUNT*32)
            self.assertEqual(len({x['case'] for x in a}),len(a))
            self.assertEqual(selection_sha(a),selection_sha(b))

    def test_wrong_group_count_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            case=Path(folder)/'content/only/0000'; case.mkdir(parents=True)
            (case/'meta.json').write_text('{}'); (case/'seed_motion.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'expected 58'):
                select_stratified(folder)

    def test_bvh_paths_are_lowercase_and_safe(self):
        with tempfile.TemporaryDirectory() as folder:
            case=Path(folder)/'content/group/0000'; case.mkdir(parents=True)
            (case/'seed_motion.json').write_text(json.dumps({'bvh_path':'BVH/240101/a.bvh'}))
            self.assertEqual(bvh_members(folder),['soma_uniform/bvh/240101/a.bvh'])
            (case/'seed_motion.json').write_text(json.dumps({'bvh_path':'BVH/../../escape.bvh'}))
            with self.assertRaisesRegex(ValueError,'unsafe'):
                bvh_members(folder)

    def test_archive_header_fails_closed_for_corrupt_input(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'broken.tar.gz'; path.write_bytes(b'not an archive')
            with self.assertRaisesRegex(ValueError,'corrupt archive'):
                archive_format(path)
            path.write_bytes(b'\x1f\x8b' + b'\0'*6)
            self.assertEqual(archive_format(path),'gzip')

    def test_kit_validation_requires_all_modalities(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); (root/'test.txt').write_text('00001\n')
            for name in ('new_joint_vecs','new_joints','texts'): (root/name).mkdir()
            (root/'new_joint_vecs/00001.npy').write_bytes(b'x')
            result=validate_kit(root,root/'test.txt')
            self.assertEqual(result['status'],'missing_resources'); self.assertEqual(result['missing_count'],2)

    def test_kit_conversion_contract_axes_units_fps_and_joint_names(self):
        source=np.zeros((2,21,3),dtype=np.float32)
        source[0,0]=[1000,2000,3000]
        motion=kit21_motion(source,12.5)
        np.testing.assert_allclose(motion['joints'][0,0],[1,-3,2])
        self.assertEqual(motion['joints'].shape,(2,21,3))
        self.assertEqual(float(motion['fps']),12.5)
        self.assertEqual(motion['joint_names'][5],'right_shoulder')
        self.assertEqual(motion['joint_names'][8],'left_shoulder')
        self.assertEqual(motion['metadata']['units'],'m')
        self.assertEqual(motion['metadata']['axes'],'right-handed Z-up')
        with self.assertRaisesRegex(ValueError,r'\[T,21,3\]'):
            kit21_motion(np.zeros((2,22,3)))

    def test_postprocess_ablation_is_exactly_52_groups_by_8(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); suite=root/'suite'; target=root/'post'; rows=[]
            for group in range(52):
                for case in range(8):
                    rel=f'content/constraints_group_{group:02d}/{case:04d}'
                    source=suite/rel; source.mkdir(parents=True)
                    (source/'meta.json').write_text('{}')
                    rows.append({'group':f'content/constraints_group_{group:02d}','case':rel})
            manifest=root/'manifest.json'; manifest.write_text(json.dumps({'cases':rows}))
            selected=_postprocess_subset(suite,manifest,target)
            self.assertEqual(len(selected),416)
            self.assertEqual(len(set(selected)),416)
            self.assertEqual(set(selected),{row['case'] for row in rows})
            self.assertEqual(len(list(target.rglob('meta.json'))),416)


if __name__=='__main__': unittest.main()
