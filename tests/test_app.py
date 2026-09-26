"""One compact check of the real image pipeline and validation boundaries."""
import sys
from pathlib import Path
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import Vision, decode_image, encode_image, EFFECTS


class AppCheck(unittest.TestCase):
    def test_pipeline_and_boundaries(self):
        root = Path(__file__).resolve().parents[1]
        vision = Vision(root / 'models')
        sample = cv2.imread(str(root / 'assets/sample.jpg'))
        self.assertIsNotNone(sample)
        self.assertEqual(len(vision.detect(sample)), 1)
        self.assertEqual(decode_image(encode_image(sample)).shape, sample.shape)
        for effect in EFFECTS:
            out, metrics = vision.process(sample, effect, .5, False)
            self.assertEqual(out.shape, sample.shape)
            self.assertEqual(out.dtype, np.uint8)
            self.assertEqual(metrics['faces'], 1)
            self.assertGreater(metrics['pipeline_ms'], 0)
            if effect != 'none':
                self.assertTrue(np.any(out != sample))
        self.assertAlmostEqual(vision.verify(sample, sample, .363)['cosine'], 1, places=5)
        blank = np.zeros_like(sample)
        self.assertEqual(len(vision.detect(blank)), 0)
        with self.assertRaises(ValueError):
            vision.feature(blank)
        for malformed in [None, '', 'data:image/jpeg;base64,!', 'https://example.com/a.jpg']:
            with self.assertRaises(ValueError):
                decode_image(malformed)
        with self.assertRaises(ValueError):
            vision.process(sample, 'unknown', .5, False)
        for targets in [None, [0, 1], [1, 1, 0, 0, 1], [0, 0, 0, 0, 1], [0, 1, 0, 2, 1], [False, 1, 0, 0, 1]]:
            with self.assertRaises(ValueError):
                vision.edit_attributes(sample, targets)


if __name__ == '__main__':
    unittest.main()
