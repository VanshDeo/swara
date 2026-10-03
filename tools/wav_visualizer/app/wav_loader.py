"""
WAV File Ingestion and Header Validator (Desktop Tooling).
Validates WAV files strictly against the Swara Audio Specification without modifying files.
"""

import os
import wave
import struct
from typing import Tuple, List, Optional
import numpy as np

from .models import AudioMetadata, AudioStatistics


def inspect_wav_header(filepath: str) -> AudioMetadata:
    """
    Reads WAV header and checks compliance with Swara Audio Contract:
    - Sample rate: 16,000 Hz
    - Channels: 1 (Mono)
    - Bit depth: 16-bit signed PCM
    - Min length: > 0 samples
    """
    if not os.path.exists(filepath):
        return AudioMetadata(
            filepath=filepath,
            filename=os.path.basename(filepath),
            file_size_bytes=0,
            duration_sec=0.0,
            sample_rate=0,
            channels=0,
            bit_depth=0,
            num_samples=0,
            is_pcm=False,
            is_valid_swara_contract=False,
            validation_reasons=[f"File does not exist: {filepath}"]
        )

    file_size = os.path.getsize(filepath)
    reasons: List[str] = []
    is_pcm = True
    sr = 0
    ch = 0
    bd = 0
    num_samples = 0
    duration_sec = 0.0

    try:
        with wave.open(filepath, "rb") as wf:
            ch = wf.getnchannels()
            sr = wf.getframerate()
            sampwidth = wf.getsampwidth()
            bd = sampwidth * 8
            num_samples = wf.getnframes()
            duration_sec = float(num_samples) / float(sr) if sr > 0 else 0.0

            # Validate constraints
            if sr != 16000:
                reasons.append(f"INVALID: Sample rate = {sr} Hz (Expected 16,000 Hz)")
            if ch != 1:
                reasons.append(f"INVALID: Channels = {ch} (Expected mono / 1 channel)")
            if bd != 16:
                reasons.append(f"INVALID: Bit depth = {bd}-bit (Expected 16-bit signed PCM)")
            if num_samples == 0:
                reasons.append("INVALID: File contains 0 audio samples (Empty WAV)")
            if duration_sec < 0.2:
                reasons.append(f"WARNING: Duration {duration_sec:.2f}s is unusually short (< 200 ms)")
            elif duration_sec > 5.0:
                reasons.append(f"WARNING: Duration {duration_sec:.2f}s is unusually long (> 5.0 s)")

    except wave.Error as we:
        is_pcm = False
        reasons.append(f"INVALID: Corrupt or non-PCM WAV format ({we})")
    except Exception as e:
        is_pcm = False
        reasons.append(f"INVALID: Cannot parse audio file ({e})")

    is_valid = (len([r for r in reasons if r.startswith("INVALID")]) == 0) and is_pcm

    return AudioMetadata(
        filepath=os.path.abspath(filepath),
        filename=os.path.basename(filepath),
        file_size_bytes=file_size,
        duration_sec=duration_sec,
        sample_rate=sr,
        channels=ch,
        bit_depth=bd,
        num_samples=num_samples,
        is_pcm=is_pcm,
        is_valid_swara_contract=is_valid,
        validation_reasons=reasons,
    )


def load_wav_samples(filepath: str, target_length: int = 16000) -> Tuple[np.ndarray, AudioStatistics]:
    """
    Reads raw audio samples and computes signal statistics.
    Returns:
    - samples: np.ndarray (int16), centered/padded to target_length (16,000) for standard analysis
    - statistics: AudioStatistics object with clipping, RMS, and dynamic range
    """
    with wave.open(filepath, "rb") as wf:
        n_frames = wf.getnframes()
        raw_data = wf.readframes(n_frames)
        samples = np.frombuffer(raw_data, dtype=np.int16)

    # Calculate statistics on unmodified raw samples
    abs_samples = np.abs(samples.astype(np.float32))
    peak = float(np.max(abs_samples)) if len(samples) > 0 else 0.0
    rms = float(np.sqrt(np.mean(abs_samples ** 2))) if len(samples) > 0 else 0.0
    dc_offset = float(np.mean(samples.astype(np.float32))) if len(samples) > 0 else 0.0

    # Clipping detection (|sample| >= 32760)
    clipping_count = int(np.sum(abs_samples >= 32760))
    clipping_pct = (float(clipping_count) / float(len(samples)) * 100.0) if len(samples) > 0 else 0.0

    # Near zero samples (|sample| < 20)
    near_zero_count = int(np.sum(abs_samples < 20))
    near_zero_pct = (float(near_zero_count) / float(len(samples)) * 100.0) if len(samples) > 0 else 0.0

    # Dynamic range in dB
    if rms > 1e-6:
        dynamic_range_db = float(20.0 * np.log10(max(1.0, peak) / max(1e-6, rms)))
    else:
        dynamic_range_db = 0.0

    warnings: List[str] = []
    if clipping_pct > 0.05:
        warnings.append(f"WARNING: Audio clipping detected ({clipping_pct:.2f}% of samples near saturation)")
    if peak < 1000.0:
        warnings.append("WARNING: Extremely low amplitude signal (peak < 1000 / 32767)")
    if near_zero_pct > 90.0:
        warnings.append("WARNING: Excessive silence detected (> 90% of samples near zero)")
    if abs(dc_offset) > 500.0:
        warnings.append(f"WARNING: Noticeable DC offset detected ({dc_offset:.1f})")

    stats = AudioStatistics(
        peak_amplitude=peak,
        rms_amplitude=rms,
        dynamic_range_db=dynamic_range_db,
        clipping_sample_count=clipping_count,
        clipping_percentage=clipping_pct,
        near_zero_percentage=near_zero_pct,
        dc_offset=dc_offset,
        quality_warnings=warnings,
    )

    # Standardize length to 16,000 for standard 49-frame windowing
    if len(samples) < target_length:
        pad_left = (target_length - len(samples)) // 2
        pad_right = target_length - len(samples) - pad_left
        normalized_window = np.pad(samples, (pad_left, pad_right), mode="constant", constant_values=0)
    elif len(samples) > target_length:
        start = (len(samples) - target_length) // 2
        normalized_window = samples[start : start + target_length]
    else:
        normalized_window = samples

    return normalized_window, stats
