"""
Unit Tests for Visualizer Signal Processing & Stage Extraction (Desktop Tooling).
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.signal_processing import VisualizerPipeline


class TestSignalProcessing(unittest.TestCase):

    def setUp(self):
        self.pipeline = VisualizerPipeline()

    def test_01_frame_dimensions(self):
        """Verify frame length, hop size, FFT size, and bin counts."""
        self.assertEqual(self.pipeline.frame_len, 480)
        self.assertEqual(self.pipeline.frame_hop, 320)
        self.assertEqual(self.pipeline.fft_size, 512)
        self.assertEqual(self.pipeline.fft_bins, 257)
        self.assertEqual(self.pipeline.num_mel_filters, 20)
        self.assertEqual(self.pipeline.num_mfcc, 10)
        self.assertEqual(self.pipeline.preemphasis, 0.97)

    def test_02_single_frame_extraction(self):
        """Verify extraction of all stages for a single frame."""
        frame_samples = (np.sin(np.linspace(0, 10, 480)) * 15000).astype(np.int16)
        analysis = self.pipeline.process_frame(frame_samples, frame_idx=5)

        self.assertEqual(analysis.frame_index, 5)
        self.assertEqual(analysis.start_sample, 5 * 320)
        self.assertEqual(analysis.end_sample, 5 * 320 + 480)
        self.assertEqual(len(analysis.preemphasized_samples), 480)
        self.assertEqual(len(analysis.windowed_samples), 480)
        self.assertEqual(len(analysis.fft_magnitude), 257)
        self.assertEqual(len(analysis.power_spectrum), 257)
        self.assertEqual(len(analysis.mel_energies), 20)
        self.assertEqual(len(analysis.log_mel_energies), 20)
        self.assertEqual(len(analysis.mfcc_coefficients), 10)

        # Ensure no NaNs or Infs
        self.assertTrue(np.all(np.isfinite(analysis.fft_magnitude)))
        self.assertTrue(np.all(np.isfinite(analysis.power_spectrum)))
        self.assertTrue(np.all(np.isfinite(analysis.mel_energies)))
        self.assertTrue(np.all(np.isfinite(analysis.log_mel_energies)))
        self.assertTrue(np.all(np.isfinite(analysis.mfcc_coefficients)))

    def test_03_full_audio_49_frames(self):
        """Verify full 1.0s window produces exactly 49 frames and (49, 10) MFCC matrix."""
        fixture_wav = os.path.abspath("tests/data/test_16k_1s.wav")
        self.assertTrue(os.path.exists(fixture_wav), f"Fixture not found: {fixture_wav}")

        processed = self.pipeline.process_full_audio(fixture_wav)
        self.assertEqual(len(processed.frames), 49)
        self.assertEqual(processed.mfcc_matrix.shape, (49, 10))
        self.assertEqual(processed.spectrogram_matrix.shape, (49, 257))
        self.assertTrue(np.all(np.isfinite(processed.mfcc_matrix)))

    def test_04_deterministic_repeated_extraction(self):
        """Repeated feature extraction on identical audio must be 100% bit-exact."""
        fixture_wav = os.path.abspath("tests/data/test_16k_1s.wav")
        res1 = self.pipeline.process_full_audio(fixture_wav)
        res2 = self.pipeline.process_full_audio(fixture_wav)

        np.testing.assert_array_equal(res1.mfcc_matrix, res2.mfcc_matrix)
        np.testing.assert_array_equal(res1.spectrogram_matrix, res2.spectrogram_matrix)


if __name__ == "__main__":
    unittest.main()
