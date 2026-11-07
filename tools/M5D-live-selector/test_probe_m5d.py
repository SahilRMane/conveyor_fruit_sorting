from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from probe_m5d import image_features
from utility_core import assess_features


class ImageEvidenceTests(unittest.TestCase):
    def test_selector_uses_the_matching_capture_and_crop(self):
        rgb = np.arange(12 * 10 * 3, dtype=np.uint8).reshape(12, 10, 3)
        control = SimpleNamespace(deferred_images=[(Path('assess-70.png'), rgb)])
        sample = dict(capture_ns=70, crop_xyxy=[1, 2, 9, 10], class_evidence={'confidence': .51})
        np.testing.assert_array_equal(image_features(control, sample), assess_features(rgb[2:10, 1:9], .51))
        with self.assertRaisesRegex(ValueError, 'stamp differs'):
            image_features(control, dict(sample, capture_ns=71))
