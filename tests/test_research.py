"""Small regression check for parsers, held-out thresholds and NME mathematics."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from research.lfw import evaluate_scores,parse_pairs
from research.landmarks import nme
from research.data import wider_to_coco


class ResearchChecks(unittest.TestCase):
    def test_protocol_and_geometry(self):
        # Deliberately reversed second fold: held-out threshold cannot cheat.
        report=evaluate_scores([.9,.1,.1,.9],[1,0,1,0],[0,0,1,1])
        self.assertEqual(report['accuracy_mean'],.25)
        target=np.zeros((1,68,2)); target[:,45,0]=10
        self.assertEqual(float(nme(target,target)[0]),0.)
        self.assertAlmostEqual(float(nme(target+1,target)[0]),np.sqrt(2)/10)
        with self.assertRaises(ValueError): nme(target,np.zeros_like(target))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ['Alice','Bob']:
                (root/name).mkdir()
                for number in [1,2]: Image.new('RGB',(20,20)).save(root/name/f'{name}_{number:04d}.jpg')
            pairs=root/'pairs.txt'; pairs.write_text('2 1\nAlice 1 2\nAlice 1 Bob 1\nBob 1 2\nBob 1 Alice 1\n')
            self.assertEqual(len(parse_pairs(root,pairs,strict=False)),4)
            with self.assertRaises(ValueError): parse_pairs(root,pairs)
            pairs.write_text('2 1\n../Alice 1 2\nAlice 1 Bob 1\nBob 1 2\nBob 1 Alice 1\n')
            with self.assertRaises(ValueError): parse_pairs(root,pairs,strict=False)
            annotation=root/'wider.txt'
            annotation.write_text('Alice/Alice_0001.jpg\n2\n-2 -2 10 10 0 0 0 0 0 0\n1 1 3 3 0 0 0 1 0 0\nBob/Bob_0001.jpg\n0\n0 0 0 0 0 0 0 0 0 0\n')
            report=wider_to_coco(root,annotation,root/'coco.json')
            self.assertEqual(report,{'images':2,'valid_boxes':1,'excluded_boxes':1})
            self.assertEqual(json.loads((root/'coco.json').read_text())['annotations'][0]['bbox'],[0,0,8,8])


if __name__=='__main__': unittest.main()
