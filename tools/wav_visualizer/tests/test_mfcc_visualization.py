"""
Unit Tests for C Reference vs Visualizer MFCC Parity.
Checks that the visualizer DSP pipeline matches the active C runtime (src/features/mfcc.c)
within the established numerical tolerance (< 0.05 absolute error).
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.signal_processing import VisualizerPipeline

# Path to training dataset extractor for baseline check
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "training")))
from dataset import SwaraFeatureExtractor


class TestVisualizerParity(unittest.TestCase):

    def test_01_parity_with_training_feature_extractor(self):
        """Visualizer pipeline must be bit-exact with training/dataset.py feature extractor."""
        pipeline = VisualizerPipeline()
        dataset_extractor = SwaraFeatureExtractor()

        fixture_wav = os.path.abspath("tests/data/test_16k_1s.wav")
        self.assertTrue(os.path.exists(fixture_wav))

        processed = pipeline.process_full_audio(fixture_wav)
        dataset_features = dataset_extractor.extract_window_mfcc(processed.raw_samples)[:, :, 0]

        np.testing.assert_allclose(processed.mfcc_matrix, dataset_features, rtol=1e-5, atol=1e-5)

    def test_02_parity_with_c_reference_binary(self):
        """Visualizer MFCC output must match C reference binary dump within 0.05 tolerance."""
        c_ref_bin = os.path.abspath("tests/data/test_c_mfcc_reference.bin")
        self.assertTrue(os.path.exists(c_ref_bin), f"C reference binary not found: {c_ref_bin}")

        with open(c_ref_bin, "rb") as f:
            c_mfcc = np.frombuffer(f.read(), dtype=np.float32).reshape(49, 10)

        pipeline = VisualizerPipeline()
        fixture_wav = os.path.abspath("tests/data/test_16k_1s.wav")
        processed = pipeline.process_full_audio(fixture_wav)

        max_err = float(np.max(np.abs(processed.mfcc_matrix - c_mfcc)))
        mean_err = float(np.mean(np.abs(processed.mfcc_matrix - c_mfcc)))

        print(f"\n[Visualizer vs C Parity] Max error: {max_err:.6f}, Mean error: {mean_err:.6f}")
        self.assertLess(max_err, 0.05, f"Visualizer MFCC differs from C reference by {max_err} >= 0.05")


if __name__ == "__main__":
    unittest.main()
