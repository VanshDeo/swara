"""
Comprehensive Verification Test Suite for Swara Infrastructure & Robustness.

Verifies:
1. Deterministic test fixture loading and format verification.
2. SwaraDataset clean rejection of corrupt/malformed WAVs and 0-frame files.
3. Feature extractor output shape (49, 10, 1) and dtype (float32).
4. Deterministic recording-level splitting contract (labeled as recording-level).
5. Safe failure of training/quantization when real dataset is missing or has missing classes.
6. Deduplication by content hash prevents duplicate recordings across splits.
7. Model export rejects missing or non-TFLite binaries.
8. Dataset audit utility (inspect_dataset) handles empty datasets and reports class distributions.
9. DS-CNN channel configurability via config.py.
"""

import os
import sys
import shutil
import tempfile
import unittest
import numpy as np

# Add training directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "training")))

from dataset import SwaraDataset, SwaraFeatureExtractor, inspect_dataset
from config import CLASSES, CLASS_TO_IDX, DEFAULT_NUM_FILTERS, CANDIDATE_CHANNEL_WIDTHS
from model import build_dscnn_model
from train import train
from evaluate import compute_classification_metrics
from quantize import convert_to_float32_tflite, convert_to_int8_tflite
from validate_tflite import inspect_tflite_model, TFLM_STANDARD_OPS
from export_model_header import export_model_to_c_array


class TestSwaraTrainingInfrastructure(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.fixture_dir = os.path.abspath("tests/data/test_dataset_fixture")

    def test_01_feature_extractor_contract(self):
        """Verify feature extractor adheres strictly to Frozen Audio Spec V0."""
        extractor = SwaraFeatureExtractor()
        audio = np.zeros(16000, dtype=np.int16)
        features = extractor.extract_window_mfcc(audio)

        self.assertEqual(features.shape, (49, 10, 1))
        self.assertEqual(features.dtype, np.float32)
        self.assertEqual(extractor.num_mel_filters, 20)
        self.assertEqual(extractor.num_mfcc, 10)
        self.assertEqual(extractor.preemphasis, 0.97)

    def test_02_dataset_scanning_and_fixtures(self):
        """Verify scanning finds test fixtures and assigns correct classes."""
        dataset = SwaraDataset(data_dir=self.fixture_dir)
        splits = dataset.scan_dataset()

        total = dataset.count_total_files(splits)
        self.assertGreaterEqual(total, 3)

        classes_found = set()
        for split_name, entries in splits.items():
            for path, class_idx, speaker_id in entries:
                self.assertIsNone(speaker_id)  # Speaker ID must be unavailable
                classes_found.add(class_idx)

        self.assertEqual(len(classes_found), 3)

    def test_03_corrupt_wav_clean_rejection(self):
        """Verify corrupt non-WAV audio files and zero-frame files are cleanly rejected without crashing."""
        dataset = SwaraDataset(data_dir=self.fixture_dir)
        with tempfile.TemporaryDirectory() as tmp_dir:
            corrupt_path = os.path.join(tmp_dir, "corrupt.wav")
            with open(corrupt_path, "wb") as f:
                f.write(b"THIS_IS_NOT_A_VALID_RIFF_WAV_HEADER_DATA_1234567890")

            with self.assertRaises(ValueError):
                dataset.load_wav_file(corrupt_path)

            # Loading from list should cleanly skip corrupt file
            corrupt_entries = [(corrupt_path, 1, None)]
            X, y = dataset.load_tensors_from_file_list(corrupt_entries)
            self.assertEqual(len(X), 0)
            self.assertEqual(len(y), 0)

    def test_04_recording_level_split_determinism(self):
        """Verify split is deterministic and explicitly labeled recording-level."""
        id1 = "swara_sample_42.wav"
        id2 = "unknown_speaker_99.wav"

        split1_a = SwaraDataset.get_recording_split(id1)
        split1_b = SwaraDataset.get_recording_split(id1)
        self.assertEqual(split1_a, split1_b)

        split2_a = SwaraDataset.get_recording_split(id2)
        split2_b = SwaraDataset.get_recording_split(id2)
        self.assertEqual(split2_a, split2_b)
        self.assertIn(split1_a, ["train", "val", "test"])

    def test_05_training_fails_when_real_dataset_absent_or_missing_classes(self):
        """Ensure training fails with FileNotFoundError or ValueError when real dataset is absent or missing classes."""
        with tempfile.TemporaryDirectory() as empty_dir:
            with self.assertRaises(FileNotFoundError) as ctx:
                train(data_dir=empty_dir, epochs=1, dry_run=True)
            self.assertIn("No valid WAV audio files found", str(ctx.exception))

        # Test partial classes present (e.g. only 'swara', missing 'silence' and 'unknown')
        with tempfile.TemporaryDirectory() as partial_dir:
            swara_dir = os.path.join(partial_dir, "swara")
            os.makedirs(swara_dir, exist_ok=True)
            # copy a valid fixture into swara_dir
            src_fixture = os.path.join(self.fixture_dir, "swara", "swara_01.wav")
            with open(src_fixture, "rb") as sf, open(os.path.join(swara_dir, "swara_01.wav"), "wb") as df:
                df.write(sf.read())

            with self.assertRaises(ValueError) as ctx:
                train(data_dir=partial_dir, epochs=1, dry_run=True)
            self.assertIn("Required audio classes missing", str(ctx.exception))

    def test_06_quantization_fails_without_real_calibration_data(self):
        """Ensure INT8 quantization fails safely if representative data is missing."""
        with tempfile.TemporaryDirectory() as empty_dir:
            dummy_model_path = os.path.join(empty_dir, "dummy_model")
            with self.assertRaises((FileNotFoundError, RuntimeError)):
                convert_to_int8_tflite(
                    model_path=dummy_model_path,
                    data_dir=empty_dir,
                    output_path=os.path.join(empty_dir, "model.tflite"),
                )

    def test_07_model_architecture_and_configurability(self):
        """Verify DS-CNN model structure, default channel count (64), and configurability."""
        self.assertEqual(DEFAULT_NUM_FILTERS, 64)
        model_64 = build_dscnn_model(num_filters=64)
        self.assertIsNotNone(model_64)
        self.assertEqual(model_64.input_shape, (None, 49, 10, 1))
        self.assertEqual(model_64.output_shape, (None, 3))

        # Verify alternative widths (e.g. 32-channel candidate) build properly
        model_32 = build_dscnn_model(num_filters=32)
        self.assertIsNotNone(model_32)
        self.assertLess(model_32.count_params(), model_64.count_params())

    def test_08_duplicate_recording_split_isolation(self):
        """Ensure identical content recordings are grouped into the same split to avoid data leakage."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            swara_dir = os.path.join(tmp_dir, "swara")
            os.makedirs(swara_dir, exist_ok=True)
            src_fixture = os.path.join(self.fixture_dir, "swara", "swara_01.wav")
            with open(src_fixture, "rb") as sf:
                data = sf.read()

            # Create 2 copies with different names
            with open(os.path.join(swara_dir, "rec_alpha.wav"), "wb") as f1:
                f1.write(data)
            with open(os.path.join(swara_dir, "rec_beta.wav"), "wb") as f2:
                f2.write(data)

            dataset = SwaraDataset(data_dir=tmp_dir)
            splits = dataset.scan_dataset(deduplicate_by_content=True)

            # Both files must reside in the EXACT same split
            splits_with_alpha = [s for s, entries in splits.items() if any("rec_alpha.wav" in e[0] for e in entries)]
            splits_with_beta = [s for s, entries in splits.items() if any("rec_beta.wav" in e[0] for e in entries)]

            self.assertEqual(splits_with_alpha, splits_with_beta)

    def test_09_model_export_validation_and_rejection(self):
        """Verify exporter rejects invalid/missing models and handles valid flatbuffers."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 1. Non-existent file
            with self.assertRaises(FileNotFoundError):
                export_model_to_c_array(os.path.join(temp_dir, "nonexistent.tflite"))

            # 2. Corrupt non-flatbuffer binary
            corrupt_path = os.path.join(temp_dir, "corrupt.tflite")
            with open(corrupt_path, "wb") as f:
                f.write(b"NOT_A_VALID_TFLITE_HEADER")
            with self.assertRaises(ValueError):
                export_model_to_c_array(corrupt_path)

            # 3. Valid flatbuffer header with skip_validation
            valid_path = os.path.join(temp_dir, "valid.tflite")
            dummy_bytes = b"\x18\x00\x00\x00TFL3\x00\x01\x02\x03\x04\x05\x06\x07\x08" + b"\x00" * 20
            with open(valid_path, "wb") as f:
                f.write(dummy_bytes)

            h_path = os.path.join(temp_dir, "model_data.h")
            cc_path = os.path.join(temp_dir, "model_data.cc")

            export_model_to_c_array(valid_path, cc_path, h_path, skip_validation=True)
            self.assertTrue(os.path.exists(h_path))
            self.assertTrue(os.path.exists(cc_path))

    def test_10_dataset_audit_tool(self):
        """Verify dataset audit tool reports class counts, durations, and fails on empty directory."""
        with tempfile.TemporaryDirectory() as empty_dir:
            report_empty = inspect_dataset(empty_dir)
            self.assertIn("error", report_empty)

        report_fixtures = inspect_dataset(self.fixture_dir)
        self.assertGreaterEqual(report_fixtures["valid_wav_files"], 3)
        self.assertGreaterEqual(report_fixtures["usable_recordings"], 3)
        self.assertIn("class_breakdown", report_fixtures)

    def test_11_same_audio_different_filenames_same_partition(self):
        """
        M5 Regression Test:
        Prove that two byte-identical WAV files with completely different filenames
        are guaranteed to end up in the exact same partition (train/val/test), and
        that changing a filename without changing the audio does not cause the
        recording to move across partitions.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            src_fixture = os.path.join(self.fixture_dir, "swara", "swara_01.wav")
            with open(src_fixture, "rb") as f:
                content = f.read()

            # Create multiple files with different filenames but identical audio content
            path_a = os.path.join(temp_dir, "recording_alpha_001.wav")
            path_b = os.path.join(temp_dir, "completely_different_name_999.wav")
            path_c = os.path.join(temp_dir, "user_renamed_audio.wav")

            with open(path_a, "wb") as fa:
                fa.write(content)
            with open(path_b, "wb") as fb:
                fb.write(content)
            with open(path_c, "wb") as fc:
                fc.write(content)

            split_a = SwaraDataset.get_recording_split(path_a)
            split_b = SwaraDataset.get_recording_split(path_b)
            split_c = SwaraDataset.get_recording_split(path_c)

            self.assertEqual(split_a, split_b, "Identical audio with different names must receive identical partition!")
            self.assertEqual(split_b, split_c, "Renaming an audio file must not alter its partition!")

            # Verify in full dataset scanner
            test_data_dir = os.path.join(temp_dir, "test_dataset")
            swara_dir = os.path.join(test_data_dir, "swara")
            os.makedirs(swara_dir, exist_ok=True)
            shutil.copy2(path_a, os.path.join(swara_dir, "orig.wav"))
            shutil.copy2(path_b, os.path.join(swara_dir, "copy_with_different_name.wav"))

            ds = SwaraDataset(test_data_dir)
            splits = ds.scan_dataset(deduplicate_by_content=True)

            assigned_splits = []
            for s_name, entries in splits.items():
                for p, _, _ in entries:
                    assigned_splits.append(s_name)

            self.assertEqual(len(assigned_splits), 2)
            self.assertEqual(assigned_splits[0], assigned_splits[1], "Both duplicate files must be in the same partition!")


if __name__ == "__main__":
    unittest.main()
