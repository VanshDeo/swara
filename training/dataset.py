"""
Swara Dataset Loader, Feature Preprocessing & Inspection Pipeline (Offline Tooling).

This module provides offline dataset utilities for Swara keyword spotting:
- Ingestion of 16-bit PCM WAV audio
- Robust format validation and clean rejection of corrupt / non-compliant files
- Recording-level deterministic train/val/test partitioning (clearly labeled)
- Detection and avoidance of duplicate recordings across splits
- Comprehensive dataset inspection & integrity auditing
- Exact feature extraction matching the active C runtime (src/features/mfcc.c)
"""

import os
import wave
import struct
import hashlib
from typing import Tuple, List, Dict, Optional, Any, Set
import numpy as np

from config import CLASSES, CLASS_TO_IDX, IDX_TO_CLASS


def hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + (hz / 700.0))


def mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


class SwaraFeatureExtractor:
    """
    Feature extractor synchronized with Swara's active C implementation (mfcc.c, fft.c).
    NOTE: Pre-emphasis is currently active in C (SWARA_PREEMPHASIS_COEFF = 0.97).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_length_samples: int = 480,
        frame_step_samples: int = 320,
        fft_size: int = 512,
        num_mel_filters: int = 20,
        num_mfcc_coeffs: int = 10,
        low_freq_hz: float = 20.0,
        high_freq_hz: float = 8000.0,
        preemphasis_coeff: float = 0.97,
        use_preemphasis: bool = True,
    ):
        self.sample_rate = sample_rate
        self.frame_length = frame_length_samples
        self.frame_step = frame_step_samples
        self.fft_size = fft_size
        self.fft_bins = (fft_size // 2) + 1
        self.num_mel_filters = num_mel_filters
        self.num_mfcc = num_mfcc_coeffs
        self.low_freq_hz = low_freq_hz
        self.high_freq_hz = high_freq_hz
        self.preemphasis = preemphasis_coeff
        self.use_preemphasis = use_preemphasis

        n = np.arange(self.frame_length, dtype=np.float32)
        self.hamming_window = 0.54 - 0.46 * np.cos(2.0 * np.pi * n / float(self.frame_length - 1))
        self.mel_filterbank = self._build_mel_filterbank()
        self.dct_matrix = self._build_dct_matrix()

    def _build_mel_filterbank(self) -> np.ndarray:
        mel_min = hz_to_mel(self.low_freq_hz)
        mel_max = hz_to_mel(self.high_freq_hz)
        mel_step = (mel_max - mel_min) / float(self.num_mel_filters + 1)

        filter_bins = []
        for i in range(self.num_mel_filters + 2):
            mel = mel_min + (float(i) * mel_step)
            hz = mel_to_hz(mel)
            bin_idx = int(np.floor(hz * float(self.fft_size) / float(self.sample_rate)))
            bin_idx = max(0, min(bin_idx, self.fft_bins - 1))
            filter_bins.append(bin_idx)

        weights = np.zeros((self.num_mel_filters, self.fft_bins), dtype=np.float32)
        for m in range(self.num_mel_filters):
            left = filter_bins[m]
            center = filter_bins[m + 1]
            right = filter_bins[m + 2]

            if center == left:
                center = left + 1
            if right <= center:
                right = center + 1
            if right >= self.fft_bins:
                right = self.fft_bins - 1

            for k in range(left, center):
                weights[m, k] = float(k - left) / float(center - left)
            for k in range(center, right + 1):
                weights[m, k] = float(right - k) / float(right - center)

        return weights

    def _build_dct_matrix(self) -> np.ndarray:
        dct = np.zeros((self.num_mfcc, self.num_mel_filters), dtype=np.float32)
        for i in range(self.num_mfcc):
            for m in range(self.num_mel_filters):
                angle = (np.pi * float(i) * (float(m) + 0.5)) / float(self.num_mel_filters)
                dct[i, m] = np.cos(angle)
        return dct

    def extract_frame_mfcc(self, frame_samples: np.ndarray) -> np.ndarray:
        assert len(frame_samples) == self.frame_length

        if self.use_preemphasis and self.preemphasis > 0.0:
            pre = np.empty(self.frame_length, dtype=np.int16)
            pre[0] = frame_samples[0]
            for i in range(1, self.frame_length):
                val = float(frame_samples[i]) - (self.preemphasis * float(frame_samples[i - 1]))
                val = max(-32768.0, min(32767.0, val))
                pre[i] = int(val)
        else:
            pre = frame_samples

        norm_samples = (pre.astype(np.float32) / 32768.0) * self.hamming_window
        fft_res = np.fft.rfft(norm_samples, n=self.fft_size)
        power_spectrum = (np.real(fft_res) ** 2 + np.imag(fft_res) ** 2).astype(np.float32)

        mel_energies = np.dot(self.mel_filterbank, power_spectrum)
        log_energies = np.log(mel_energies + 1e-6)

        mfcc = np.dot(self.dct_matrix, log_energies)
        return mfcc

    def extract_window_mfcc(self, window_samples: np.ndarray) -> np.ndarray:
        assert len(window_samples) == 16000
        features = np.zeros((49, self.num_mfcc), dtype=np.float32)

        for frame_idx in range(49):
            start = frame_idx * self.frame_step
            frame = window_samples[start : start + self.frame_length]
            features[frame_idx] = self.extract_frame_mfcc(frame)

        return features[:, :, np.newaxis]


class SwaraDataset:
    """
    Dataset loader with deterministic recording-level splitting and strict WAV validation.
    """

    def __init__(
        self,
        data_dir: str = "data/raw",
        target_sample_rate: int = 16000,
        target_duration_samples: int = 16000,
        use_preemphasis: bool = True,
    ):
        self.data_dir = data_dir
        self.target_sample_rate = target_sample_rate
        self.target_duration = target_duration_samples
        self.feature_extractor = SwaraFeatureExtractor(use_preemphasis=use_preemphasis)

    def load_raw_samples(self, filepath: str) -> np.ndarray:
        """
        Loads unpadded, uncropped raw audio samples from 16kHz mono 16-bit PCM WAV.
        Rejects malformed headers, incorrect sample rate, channels, bit-depth, or zero-length cleanly.
        """
        if not os.path.exists(filepath):
            raise ValueError(f"{filepath}: File does not exist")
        if os.path.getsize(filepath) < 44:
            raise ValueError(f"{filepath}: File size too small for WAV header ({os.path.getsize(filepath)} bytes)")

        try:
            with wave.open(filepath, "rb") as wf:
                channels = wf.getnchannels()
                sample_rate = wf.getframerate()
                sample_width = wf.getsampwidth()
                num_frames = wf.getnframes()

                if sample_rate != self.target_sample_rate:
                    raise ValueError(f"{filepath}: Unsupported sample rate {sample_rate} Hz (expected {self.target_sample_rate} Hz)")
                if channels != 1:
                    raise ValueError(f"{filepath}: Unsupported channels {channels} (expected mono)")
                if sample_width != 2:
                    raise ValueError(f"{filepath}: Unsupported sample width {sample_width} bytes (expected 16-bit)")
                if num_frames == 0:
                    raise ValueError(f"{filepath}: Empty audio file (0 frames)")

                raw_data = wf.readframes(num_frames)
                return np.frombuffer(raw_data, dtype=np.int16)
        except (wave.Error, EOFError, struct.error) as e:
            raise ValueError(f"{filepath}: Malformed or corrupt WAV file: {e}")

    @staticmethod
    def find_peak_energy_window(samples: np.ndarray, window_size: int = 16000, step: int = 320) -> int:
        """Find start index of window_size that has highest RMS energy using frame-step hop."""
        if len(samples) <= window_size:
            return 0
        step = min(step, max(1, len(samples) - window_size))
        best_start = 0
        max_energy = -1.0
        for s in range(0, len(samples) - window_size + 1, step):
            chunk = samples[s : s + window_size].astype(np.float32)
            energy = float(np.mean(chunk ** 2))
            if energy > max_energy:
                max_energy = energy
                best_start = s
        return best_start

    def load_wav_file(self, filepath: str) -> np.ndarray:
        """
        Loads and validates a 16kHz mono 16-bit PCM WAV file.
        Uses energy-aligned slicing for recordings longer than target duration (16,000 samples).
        """
        samples = self.load_raw_samples(filepath)

        # Normalize duration to 1.0 second (16000 samples)
        if len(samples) < self.target_duration:
            pad_left = (self.target_duration - len(samples)) // 2
            pad_right = self.target_duration - len(samples) - pad_left
            samples = np.pad(samples, (pad_left, pad_right), mode="constant", constant_values=0)
        elif len(samples) > self.target_duration:
            start = self.find_peak_energy_window(samples, window_size=self.target_duration)
            samples = samples[start : start + self.target_duration]

        return samples

    @staticmethod
    def get_file_content_hash(filepath: str) -> str:
        """Compute SHA-256 hash of raw file content to identify duplicates."""
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def get_recording_split(
        file_or_content_hash: str,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> str:
        """
        DETERMINISTIC CONTENT-BASED RECORDING-LEVEL SPLIT.
        NOTE: This partition is strictly RECORDING-LEVEL and NOT speaker-independent
        because speaker identities are not available in the real dataset.

        Partition is computed deterministically from the SHA-256 content hash of the
        WAV audio file (or from an explicit SHA-256 hex string).
        
        Guarantees:
        1. Two byte-identical WAV files with different filenames ALWAYS receive the exact same split.
        2. Renaming a file without altering its audio content NEVER causes it to change partitions.
        3. Completely independent of machine, OS, or absolute filesystem paths.
        """
        if os.path.isfile(file_or_content_hash):
            c_hash = SwaraDataset.get_file_content_hash(file_or_content_hash)
        else:
            c_hash = file_or_content_hash.strip().lower()

        # If already a 64-character SHA-256 hex string, convert directly to integer
        if len(c_hash) == 64 and all(c in "0123456789abcdef" for c in c_hash):
            hash_val = int(c_hash, 16) % 100
        else:
            # Fallback for mock test identifiers / non-file strings
            hash_val = int(hashlib.sha256(c_hash.encode("utf-8")).hexdigest(), 16) % 100

        val_threshold = int(val_ratio * 100)
        test_threshold = val_threshold + int(test_ratio * 100)

        if hash_val < val_threshold:
            return "val"
        elif hash_val < test_threshold:
            return "test"
        else:
            return "train"

    def scan_dataset(self, deduplicate_by_content: bool = True) -> Dict[str, List[Tuple[str, int, Optional[str]]]]:
        """
        Scan data_dir for class subdirectories and assign each file deterministically
        Strictly ignores AppleDouble (._*) and hidden files.
        via content-hash recording-level splitting.
        If deduplicate_by_content is True, identical duplicate recordings are identified
        by SHA-256 content hash and locked to the same partition to prevent train/val leakage.
        """
        splits: Dict[str, List[Tuple[str, int, Optional[str]]]] = {"train": [], "val": [], "test": []}
        seen_content_hashes: Dict[str, str] = {}  # sha256_hash -> split_name

        # 1. Check if physical split directories already exist (e.g., data/train, data/val, data/test)
        train_dir = os.path.join(self.data_dir, "train")
        val_dir = os.path.join(self.data_dir, "val")
        test_dir = os.path.join(self.data_dir, "test")

        if os.path.exists(train_dir) or os.path.exists(val_dir) or os.path.exists(test_dir):
            for split_name, split_path in [("train", train_dir), ("val", val_dir), ("test", test_dir)]:
                if not os.path.exists(split_path):
                    continue
                for class_name, class_idx in CLASS_TO_IDX.items():
                    c_dir = os.path.join(split_path, class_name)
                    if not os.path.exists(c_dir):
                        continue
                    for root, dirs, files in os.walk(c_dir):
                        dirs[:] = [d for d in dirs if not d.startswith(".")]
                        for f in sorted(files):
                            if f.lower().endswith(".wav"):
                                splits[split_name].append((os.path.join(root, f), class_idx, None))
            if any(len(v) > 0 for v in splits.values()):
                return splits

        # 2. Fall back to scanning raw class directories and partitioning on-the-fly by content hash
        for class_name, class_idx in CLASS_TO_IDX.items():
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                raw_dir = os.path.join(self.data_dir, "raw", class_name)
                if os.path.exists(raw_dir):
                    class_dir = raw_dir
                else:
                    continue

            for root, dirs, files in os.walk(class_dir):
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                for f in sorted(files):
                    if f.lower().endswith(".wav"):
                        path = os.path.join(root, f)

                        if deduplicate_by_content:
                            try:
                                c_hash = self.get_file_content_hash(path)
                                if c_hash in seen_content_hashes:
                                    # Duplicate found! Assign to SAME split as original to prevent train/val leakage
                                    target_split = seen_content_hashes[c_hash]
                                    splits[target_split].append((path, class_idx, None))
                                    continue
                                else:
                                    target_split = self.get_recording_split(path)
                                    seen_content_hashes[c_hash] = target_split
                                    splits[target_split].append((path, class_idx, None))
                            except Exception:
                                target_split = self.get_recording_split(path)
                                splits[target_split].append((path, class_idx, None))
                        else:
                            split = self.get_recording_split(path)
                            splits[split].append((path, class_idx, None))

        return splits

    def count_total_files(self, splits: Dict[str, List[Tuple[str, int, Optional[str]]]]) -> int:
        return sum(len(entries) for entries in splits.values())

    def get_classes_present(self, splits: Dict[str, List[Tuple[str, int, Optional[str]]]]) -> Set[int]:
        classes = set()
        for entries in splits.values():
            for _, class_idx, _ in entries:
                classes.add(class_idx)
        return classes

    def load_tensors_from_file_list(
        self,
        file_entries: List[Tuple[str, int, Optional[str]]],
        augment_swara: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        num_samples = len(file_entries)
        if num_samples == 0:
            return (
                np.zeros((0, 49, 10, 1), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
            )

        valid_X = []
        valid_y = []
        rng = np.random.RandomState(42)

        for fpath, class_idx, _ in file_entries:
            try:
                raw_samples = self.load_raw_samples(fpath)
                swara_class_idx = CLASS_TO_IDX["swara"]
                unknown_class_idx = CLASS_TO_IDX["unknown"]

                if augment_swara and class_idx == swara_class_idx:
                    # Multi-slice & jitter augmentation for swara wake word
                    if len(raw_samples) > self.target_duration:
                        best_start = self.find_peak_energy_window(raw_samples, window_size=self.target_duration, step=320)
                        offsets = [0, -1200, +1200, -2400, +2400]
                        for off in offsets:
                            s = best_start + off
                            if 0 <= s and s + self.target_duration <= len(raw_samples):
                                chunk = raw_samples[s : s + self.target_duration].copy()
                                features = self.feature_extractor.extract_window_mfcc(chunk)
                                valid_X.append(features)
                                valid_y.append(class_idx)

                                # Mild noise injection on center peak slice (SNR ~25dB)
                                if off == 0:
                                    noise = rng.normal(0, 20.0, self.target_duration).astype(np.float32)
                                    noisy_chunk = np.clip(chunk.astype(np.float32) + noise, -32768, 32767).astype(np.int16)
                                    n_features = self.feature_extractor.extract_window_mfcc(noisy_chunk)
                                    valid_X.append(n_features)
                                    valid_y.append(class_idx)
                    else:
                        # Audio <= 1.0s: center pad
                        audio = self.load_wav_file(fpath)
                        valid_X.append(self.feature_extractor.extract_window_mfcc(audio))
                        valid_y.append(class_idx)

                        # Small time shifts with zero padding
                        for shift in [-800, 800]:
                            shifted = np.roll(audio, shift)
                            if shift > 0:
                                shifted[:shift] = 0
                            else:
                                shifted[shift:] = 0
                            valid_X.append(self.feature_extractor.extract_window_mfcc(shifted))
                            valid_y.append(class_idx)

                elif augment_swara and class_idx == unknown_class_idx and len(raw_samples) >= (self.target_duration * 2):
                    # Multi-slice for long negative/unknown speech to balance vocabulary
                    s1 = self.find_peak_energy_window(raw_samples, window_size=self.target_duration, step=320)
                    chunk1 = raw_samples[s1 : s1 + self.target_duration]
                    valid_X.append(self.feature_extractor.extract_window_mfcc(chunk1))
                    valid_y.append(class_idx)

                    # Second slice at least 1 second away if available
                    rem_start = s1 + self.target_duration
                    if rem_start + self.target_duration <= len(raw_samples):
                        chunk2 = raw_samples[rem_start : rem_start + self.target_duration]
                        if np.sqrt(np.mean(chunk2.astype(np.float32) ** 2)) > 300.0:
                            valid_X.append(self.feature_extractor.extract_window_mfcc(chunk2))
                            valid_y.append(class_idx)
                else:
                    audio = self.load_wav_file(fpath)
                    features = self.feature_extractor.extract_window_mfcc(audio)
                    valid_X.append(features)
                    valid_y.append(class_idx)

            except ValueError as e:
                print(f"[Warning] Skipping invalid/corrupt audio file {fpath}: {e}")

        if len(valid_X) == 0:
            return (
                np.zeros((0, 49, 10, 1), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
            )

        X = np.stack(valid_X, axis=0).astype(np.float32)
        y = np.array(valid_y, dtype=np.int32)
        return X, y


def inspect_dataset(data_dir: str) -> Dict[str, Any]:
    """
    Comprehensive dataset inspection and validation audit.
    Inspects all WAV files under data_dir and reports class distributions, durations,
    class percentages, sample rates, channel formats, duplicates, recording-level
    split counts, and a dataset sufficiency assessment.
    """
    report: Dict[str, Any] = {
        "target_directory": os.path.abspath(data_dir),
        "total_wav_files": 0,
        "valid_wav_files": 0,
        "invalid_corrupt_files": [],
        "sample_rates": {},
        "channel_counts": {},
        "bit_depths": {},
        "durations_sec": [],
        "files_shorter_than_1s": 0,
        "files_longer_than_1s": 0,
        "files_exact_1s": 0,
        "duplicate_files_count": 0,
        "duplicate_groups": [],
        "filename_patterns": set(),
        "class_breakdown": {
            cls: {
                "total": 0,
                "valid": 0,
                "duration_sec": 0.0,
                "percent_files": 0.0,
                "percent_duration": 0.0,
                "train_count": 0,
                "val_count": 0,
                "test_count": 0,
            }
            for cls in CLASSES
        },
        "recording_level_splits": {"train": 0, "val": 0, "test": 0},
        "class_imbalance_ratio": 1.0,
        "usable_recordings": 0,
        "duration_stats": {
            "min_sec": 0.0,
            "max_sec": 0.0,
            "mean_sec": 0.0,
            "median_sec": 0.0,
            "std_sec": 0.0,
        },
        "sufficiency": {
            "is_sufficient": False,
            "verdict": "DATASET NOT READY FOR TRAINING.",
            "deficiencies": [],
            "warnings": [],
        },
    }

    if not os.path.exists(data_dir):
        report["error"] = f"Directory not found: {data_dir}"
        return report

    has_physical_splits = any(os.path.exists(os.path.join(data_dir, s)) for s in ["train", "val", "test"])

    all_wavs = []
    for root, dirs, files in os.walk(data_dir):
        # Ignore hidden and staging directories (e.g., .gdrive_staging, .git)
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        if has_physical_splits:
            # If physical train/val/test splits exist in data_dir, do not recurse into raw/ or processed/
            if os.path.abspath(root) == os.path.abspath(data_dir):
                dirs[:] = [d for d in dirs if d in ["train", "val", "test"]]
        for f in sorted(files):
            if f.startswith(".") or f.startswith("._") or not f.lower().endswith(".wav"):
                continue
            all_wavs.append(os.path.join(root, f))

    report["total_wav_files"] = len(all_wavs)
    if len(all_wavs) == 0:
        report["error"] = f"No .wav files found under: {os.path.abspath(data_dir)}"
        report["sufficiency"]["deficiencies"].append(f"No .wav files found in {data_dir}")
        return report

    content_hashes: Dict[str, List[str]] = {}

    for wav_path in all_wavs:
        # Determine class if within class folder
        path_lower = wav_path.lower().replace("\\", "/")
        assigned_class = None
        for c in CLASSES:
            if f"/{c}/" in path_lower or path_lower.endswith(f"/{c}"):
                assigned_class = c
                break

        if assigned_class:
            report["class_breakdown"][assigned_class]["total"] += 1

        basename = os.path.basename(wav_path)
        pattern = "numeric" if any(c.isdigit() for c in basename) else "alpha"
        if "_" in basename:
            pattern += f"_underscore_{len(basename.split('_'))-1}"
        report["filename_patterns"].add(pattern)

        try:
            with open(wav_path, "rb") as raw_f:
                file_bytes = raw_f.read()
                file_sha256 = hashlib.sha256(file_bytes).hexdigest()
                content_hashes.setdefault(file_sha256, []).append(wav_path)

            with wave.open(wav_path, "rb") as wf:
                channels = wf.getnchannels()
                sample_rate = wf.getframerate()
                sample_width = wf.getsampwidth()
                num_frames = wf.getnframes()
                bit_depth = sample_width * 8

                if num_frames == 0:
                    raise ValueError("Audio file contains 0 frames")

                report["valid_wav_files"] += 1
                report["sample_rates"][sample_rate] = report["sample_rates"].get(sample_rate, 0) + 1
                report["channel_counts"][channels] = report["channel_counts"].get(channels, 0) + 1
                report["bit_depths"][bit_depth] = report["bit_depths"].get(bit_depth, 0) + 1

                duration_sec = float(num_frames) / float(sample_rate) if sample_rate > 0 else 0.0
                report["durations_sec"].append(duration_sec)

                # Determine recording split:
                # If physical directory structure exists (in /train/, /val/, /test/), use it;
                # otherwise compute deterministic recording split from filename.
                if "/train/" in path_lower:
                    split = "train"
                elif "/val/" in path_lower:
                    split = "val"
                elif "/test/" in path_lower:
                    split = "test"
                else:
                    split = SwaraDataset.get_recording_split(file_sha256)

                report["recording_level_splits"][split] += 1

                if assigned_class:
                    report["class_breakdown"][assigned_class]["valid"] += 1
                    report["class_breakdown"][assigned_class]["duration_sec"] += duration_sec
                    if split == "train":
                        report["class_breakdown"][assigned_class]["train_count"] += 1
                    elif split == "val":
                        report["class_breakdown"][assigned_class]["val_count"] += 1
                    elif split == "test":
                        report["class_breakdown"][assigned_class]["test_count"] += 1

                if num_frames < sample_rate:
                    report["files_shorter_than_1s"] += 1
                elif num_frames > sample_rate:
                    report["files_longer_than_1s"] += 1
                else:
                    report["files_exact_1s"] += 1

                # Check if fully usable per Swara Audio Contract (16kHz, mono, 16-bit)
                if sample_rate == 16000 and channels == 1 and bit_depth == 16:
                    report["usable_recordings"] += 1

        except Exception as e:
            report["invalid_corrupt_files"].append({"file": wav_path, "error": str(e)})

    # Calculate class percentages
    total_valid = report["valid_wav_files"]
    total_duration = sum(info["duration_sec"] for info in report["class_breakdown"].values())
    for cls, info in report["class_breakdown"].items():
        if total_valid > 0:
            info["percent_files"] = (info["valid"] / total_valid) * 100.0
        if total_duration > 0:
            info["percent_duration"] = (info["duration_sec"] / total_duration) * 100.0

    # Calculate duration statistics
    if report["durations_sec"]:
        d_arr = np.array(report["durations_sec"])
        report["duration_stats"] = {
            "min_sec": float(np.min(d_arr)),
            "max_sec": float(np.max(d_arr)),
            "mean_sec": float(np.mean(d_arr)),
            "median_sec": float(np.median(d_arr)),
            "std_sec": float(np.std(d_arr)),
        }

    # Detect duplicates
    for sha256_h, paths in content_hashes.items():
        if len(paths) > 1:
            report["duplicate_files_count"] += len(paths) - 1
            report["duplicate_groups"].append({"sha256": sha256_h, "copies": paths})

    # Class imbalance calculation
    counts = [info["valid"] for info in report["class_breakdown"].values() if info["valid"] > 0]
    if len(counts) > 1:
        report["class_imbalance_ratio"] = float(max(counts)) / float(min(counts))
    else:
        report["class_imbalance_ratio"] = 1.0

    report["filename_patterns"] = sorted(list(report["filename_patterns"]))

    # Dataset Sufficiency Assessment
    deficiencies = []
    warnings = []

    # Check for missing classes
    missing_classes = [c for c in CLASSES if report["class_breakdown"][c]["valid"] == 0]
    if missing_classes:
        deficiencies.append(f"Missing required classes with 0 recordings: {', '.join(missing_classes)}")

    # Check class sample quantities
    for c in CLASSES:
        val_c = report["class_breakdown"][c]["val_count"]
        test_c = report["class_breakdown"][c]["test_count"]
        total_c = report["class_breakdown"][c]["valid"]

        if total_c > 0:
            if val_c < 5:
                warnings.append(f"Class '{c}' has only {val_c} validation sample(s); too few for statistically meaningful validation.")
            if test_c < 5:
                warnings.append(f"Class '{c}' has only {test_c} test sample(s); too few for statistically sound evaluation.")
            if total_c < 50:
                warnings.append(f"Class '{c}' has only {total_c} total recordings (recommended >= 50 for edge wake-word training).")

    if total_valid < 30:
        deficiencies.append(f"Total usable recordings ({total_valid}) is insufficient for deep learning (minimum recommended >= 100).")

    is_sufficient = (len(deficiencies) == 0 and len(missing_classes) == 0)
    verdict = "READY FOR TRAINING." if is_sufficient else "DATASET NOT READY FOR TRAINING."

    report["sufficiency"] = {
        "is_sufficient": is_sufficient,
        "verdict": verdict,
        "deficiencies": deficiencies,
        "warnings": warnings,
    }

    return report
