"""
Swara WAV Visualizer Models & Data Structures (Desktop Tooling).
Defines dataclasses for audio file metadata, signal statistics, frame features, and validation flags.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import numpy as np


@dataclass
class AudioMetadata:
    filepath: str
    filename: str
    file_size_bytes: int
    duration_sec: float
    sample_rate: int
    channels: int
    bit_depth: int
    num_samples: int
    is_pcm: bool = True
    is_valid_swara_contract: bool = False
    validation_reasons: List[str] = field(default_factory=list)


@dataclass
class AudioStatistics:
    peak_amplitude: float
    rms_amplitude: float
    dynamic_range_db: float
    clipping_sample_count: int
    clipping_percentage: float
    near_zero_percentage: float
    dc_offset: float
    quality_warnings: List[str] = field(default_factory=list)


@dataclass
class FrameAnalysis:
    frame_index: int
    start_sample: int
    end_sample: int
    start_time_ms: float
    end_time_ms: float
    raw_samples: np.ndarray             # (480,) int16
    preemphasized_samples: np.ndarray   # (480,) int16
    windowed_samples: np.ndarray        # (480,) float32
    fft_magnitude: np.ndarray           # (257,) float32
    power_spectrum: np.ndarray          # (257,) float32
    mel_energies: np.ndarray            # (20,) float32
    log_mel_energies: np.ndarray        # (20,) float32
    mfcc_coefficients: np.ndarray       # (10,) float32


@dataclass
class ProcessedAudio:
    metadata: AudioMetadata
    statistics: AudioStatistics
    raw_samples: np.ndarray             # (N,) int16
    normalized_audio: np.ndarray        # (N,) float32 (-1.0 to 1.0)
    frames: List[FrameAnalysis]         # 49 frames for standard 1.0s window
    mfcc_matrix: np.ndarray             # (49, 10) float32
    spectrogram_matrix: np.ndarray      # (49, 257) float32
    split_category: Optional[str] = None # 'train', 'val', 'test' (RECORDING-LEVEL)
