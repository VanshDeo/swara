"""
Exact Swara Signal Processing & Multi-Stage Feature Inspector (Desktop Tooling).
Matches C reference (src/features/mfcc.c, fft.c) and training/dataset.py step-by-step:
PCM -> Pre-emphasis (0.97) -> Hamming -> 512-pt FFT -> Power Spectrum ->
20 Mel Filters -> 20 Mel Energies -> Natural Log -> DCT-II -> 10 MFCCs.
"""

import numpy as np
from typing import List, Tuple, Dict, Any

from .models import FrameAnalysis, ProcessedAudio, AudioMetadata, AudioStatistics
from .wav_loader import inspect_wav_header, load_wav_samples

# Import deterministic split function from dataset infrastructure
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "training")))
try:
    from dataset import SwaraDataset, SwaraFeatureExtractor
except ImportError:
    SwaraDataset = None
    SwaraFeatureExtractor = None


def hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + (hz / 700.0))


def mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


class VisualizerPipeline:
    """
    Step-by-step audio pipeline synchronized with Swara's frozen audio contract.
    Preserves intermediate representations for multi-stage visualization.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_len: int = 480,
        frame_hop: int = 320,
        fft_size: int = 512,
        num_mel_filters: int = 20,
        num_mfcc: int = 10,
        preemphasis_coeff: float = 0.97,
    ):
        self.sample_rate = sample_rate
        self.frame_len = frame_len
        self.frame_hop = frame_hop
        self.fft_size = fft_size
        self.fft_bins = (fft_size // 2) + 1  # 257 bins
        self.num_mel_filters = num_mel_filters
        self.num_mfcc = num_mfcc
        self.preemphasis = preemphasis_coeff

        # Hamming window matching C implementation: w[n] = 0.54 - 0.46 * cos(2*pi*n / 479)
        n = np.arange(self.frame_len, dtype=np.float32)
        self.hamming_window = 0.54 - 0.46 * np.cos(2.0 * np.pi * n / float(self.frame_len - 1))

        self.mel_filterbank = self._build_mel_filterbank()
        self.dct_matrix = self._build_dct_matrix()
        self.bin_frequencies = np.linspace(0.0, float(self.sample_rate) / 2.0, self.fft_bins)

    def _build_mel_filterbank(self) -> np.ndarray:
        mel_min = hz_to_mel(20.0)
        mel_max = hz_to_mel(8000.0)
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

    def process_frame(self, frame_samples: np.ndarray, frame_idx: int) -> FrameAnalysis:
        """Execute step-by-step extraction on a single 480-sample frame."""
        assert len(frame_samples) == self.frame_len

        # Step 1: Pre-emphasis (0.97)
        pre = np.empty(self.frame_len, dtype=np.int16)
        pre[0] = frame_samples[0]
        for i in range(1, self.frame_len):
            val = float(frame_samples[i]) - (self.preemphasis * float(frame_samples[i - 1]))
            val = max(-32768.0, min(32767.0, val))
            pre[i] = int(val)

        # Step 2: Normalization & Hamming Windowing
        norm_windowed = (pre.astype(np.float32) / 32768.0) * self.hamming_window

        # Step 3: 512-point Real FFT
        fft_res = np.fft.rfft(norm_windowed, n=self.fft_size)
        fft_mag = np.abs(fft_res).astype(np.float32)

        # Step 4: Power Spectrum
        power_spec = (np.real(fft_res) ** 2 + np.imag(fft_res) ** 2).astype(np.float32)

        # Step 5: Mel Filterbank Energies
        mel_energies = np.dot(self.mel_filterbank, power_spec)

        # Step 6: Natural Log with Numerical Floor (1e-6)
        log_mel = np.log(mel_energies + 1e-6)

        # Step 7: DCT-II -> 10 MFCC Coefficients
        mfcc = np.dot(self.dct_matrix, log_mel)

        start_sample = frame_idx * self.frame_hop
        end_sample = start_sample + self.frame_len
        start_ms = (float(start_sample) / float(self.sample_rate)) * 1000.0
        end_ms = (float(end_sample) / float(self.sample_rate)) * 1000.0

        return FrameAnalysis(
            frame_index=frame_idx,
            start_sample=start_sample,
            end_sample=end_sample,
            start_time_ms=start_ms,
            end_time_ms=end_ms,
            raw_samples=frame_samples,
            preemphasized_samples=pre,
            windowed_samples=norm_windowed,
            fft_magnitude=fft_mag,
            power_spectrum=power_spec,
            mel_energies=mel_energies,
            log_mel_energies=log_mel,
            mfcc_coefficients=mfcc,
        )

    def process_full_audio(self, filepath: str) -> ProcessedAudio:
        """Loads WAV, runs validation, signal statistics, and 49-frame windowing."""
        metadata = inspect_wav_header(filepath)
        if not metadata.is_valid_swara_contract:
            # If invalid, attempt basic load for visualization or return empty
            try:
                samples, stats = load_wav_samples(filepath)
            except Exception:
                samples = np.zeros(16000, dtype=np.int16)
                stats = AudioStatistics(0.0, 0.0, 0.0, 0, 0.0, 100.0, 0.0, ["File cannot be decoded."])
        else:
            samples, stats = load_wav_samples(filepath)

        # Split determination (recording-level)
        split_cat = None
        if SwaraDataset:
            split_cat = SwaraDataset.get_recording_split(filepath)

        norm_audio = samples.astype(np.float32) / 32768.0

        frames: List[FrameAnalysis] = []
        mfcc_matrix = np.zeros((49, self.num_mfcc), dtype=np.float32)
        spectrogram_matrix = np.zeros((49, self.fft_bins), dtype=np.float32)

        for f_idx in range(49):
            start = f_idx * self.frame_hop
            frame_samples = samples[start : start + self.frame_len]
            frame_analysis = self.process_frame(frame_samples, f_idx)
            frames.append(frame_analysis)
            mfcc_matrix[f_idx] = frame_analysis.mfcc_coefficients
            spectrogram_matrix[f_idx] = frame_analysis.power_spectrum

        return ProcessedAudio(
            metadata=metadata,
            statistics=stats,
            raw_samples=samples,
            normalized_audio=norm_audio,
            frames=frames,
            mfcc_matrix=mfcc_matrix,
            spectrogram_matrix=spectrogram_matrix,
            split_category=split_cat,
        )
