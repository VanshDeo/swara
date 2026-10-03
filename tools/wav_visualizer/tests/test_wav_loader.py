"""
Unit Tests for WAV Loader & Validation (Desktop Tooling).
"""

import os
import sys
import wave
import tempfile
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.wav_loader import inspect_wav_header, load_wav_samples


class TestWavLoader(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_test_wav(self, filename, sr=16000, ch=1, bd=16, num_samples=16000):
        filepath = os.path.join(self.temp_dir.name, filename)
        with wave.open(filepath, "wb") as wf:
            wf.setnchannels(ch)
            wf.setsampwidth(bd // 8)
            wf.setframerate(sr)
            if bd == 16:
                data = (np.sin(np.linspace(0, 10, num_samples)) * 10000).astype(np.int16).tobytes()
            else:
                data = b"\x00" * (num_samples * (bd // 8))
            wf.writeframes(data)
        return filepath

    def test_01_valid_16k_mono_16bit(self):
        """Valid 16 kHz mono 16-bit WAV passes validation."""
        path = self._create_test_wav("valid.wav", sr=16000, ch=1, bd=16, num_samples=16000)
        meta = inspect_wav_header(path)
        self.assertTrue(meta.is_valid_swara_contract)
        self.assertEqual(meta.sample_rate, 16000)
        self.assertEqual(meta.channels, 1)
        self.assertEqual(meta.bit_depth, 16)
        self.assertEqual(len(meta.validation_reasons), 0)

    def test_02_invalid_sample_rate(self):
        """Sample rate other than 16,000 Hz is flagged as INVALID."""
        path = self._create_test_wav("sr_44k.wav", sr=44100, ch=1, bd=16)
        meta = inspect_wav_header(path)
        self.assertFalse(meta.is_valid_swara_contract)
        self.assertTrue(any("Sample rate = 44100 Hz" in r for r in meta.validation_reasons))

    def test_03_invalid_channel_count(self):
        """Stereo (2 channels) is flagged as INVALID."""
        path = self._create_test_wav("stereo.wav", sr=16000, ch=2, bd=16)
        meta = inspect_wav_header(path)
        self.assertFalse(meta.is_valid_swara_contract)
        self.assertTrue(any("Channels = 2" in r for r in meta.validation_reasons))

    def test_04_invalid_bit_depth(self):
        """Bit depths other than 16-bit are flagged as INVALID."""
        path = self._create_test_wav("bd_8bit.wav", sr=16000, ch=1, bd=8)
        meta = inspect_wav_header(path)
        self.assertFalse(meta.is_valid_swara_contract)
        self.assertTrue(any("Bit depth = 8-bit" in r for r in meta.validation_reasons))

    def test_05_corrupt_wav(self):
        """Corrupt non-WAV headers are rejected cleanly."""
        corrupt_path = os.path.join(self.temp_dir.name, "corrupt.wav")
        with open(corrupt_path, "wb") as f:
            f.write(b"NOT_A_VALID_RIFF_WAV_HEADER_123456789")
        meta = inspect_wav_header(corrupt_path)
        self.assertFalse(meta.is_valid_swara_contract)
        self.assertTrue(any("Corrupt or non-PCM" in r for r in meta.validation_reasons))

    def test_06_empty_wav(self):
        """Empty WAV file with 0 samples is flagged as INVALID."""
        path = self._create_test_wav("empty.wav", sr=16000, ch=1, bd=16, num_samples=0)
        meta = inspect_wav_header(path)
        self.assertFalse(meta.is_valid_swara_contract)
        self.assertTrue(any("0 audio samples" in r for r in meta.validation_reasons))


if __name__ == "__main__":
    unittest.main()
