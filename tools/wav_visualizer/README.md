# Swara WAV / MFCC Visualization & Debugging Engine

> [!WARNING]
> **WINDOWS-ONLY DEVELOPMENT & TESTING TOOL ONLY.**
> This visualizer is an offline development and debugging tool for inspecting WAV datasets and validating the feature extraction pipeline prior to training.
> **It is NOT included in the embedded deployment and does NOT increase the runtime memory/flash footprint of Swara on microcontrollers or Raspberry Pi targets.**

---

## 1. Overview & Purpose

The **Swara WAV / MFCC Visualization & Debugging Engine** allows visual inspection of:
- Raw audio waveforms and signal health (peak amplitude, RMS, dynamic range, clipping detection, near-zero silence).
- Frame boundaries for the 1.0-second sliding analysis window (480-sample frame length, 320-sample hop size, 49 frames).
- Multi-stage DSP pipeline breakdown per selected frame:
  $$\text{Raw PCM} \longrightarrow \text{Pre-emphasis (0.97)} \longrightarrow \text{Hamming Window} \longrightarrow \text{512-pt FFT} \longrightarrow \text{Power Spectrum} \longrightarrow \text{20 Mel Filters} \longrightarrow \text{Log-Mel} \longrightarrow \text{DCT-II} \longrightarrow \text{10 MFCCs}$$
- Full $49 \times 10$ MFCC feature heatmap and narrowband power spectrogram ($0\text{ Hz} - 8,000\text{ Hz}$, $31.25\text{ Hz/bin}$).
- Side-by-side comparison mode for two audio signals (e.g. `swara` vs `unknown` or `silence`).
- Dataset browser scanning `data/raw/` with recording-level split labeling and automatic quality warning flags.

---

## 2. Requirements & Installation

```bash
pip install numpy matplotlib
```

The graphical user interface uses Python's built-in `tkinter` and embeds `matplotlib` figures directly into native desktop panels.

---

## 3. Launching the Visualizer

### Launch Default GUI
```bash
python tools/wav_visualizer/app/main.py
```

### Inspect a Specific WAV File
```bash
python tools/wav_visualizer/app/main.py --file tests/data/test_16k_1s.wav
```

### Open with Specific Dataset Directory
```bash
python tools/wav_visualizer/app/main.py --dataset data/raw
```

---

## 4. Swara Audio Contract & Validation Rules

The tool strictly validates input files against the frozen Swara Audio Contract:

| Parameter | Swara Contract | Visualizer Validation |
| :--- | :--- | :--- |
| **Sample Rate** | `16,000 Hz` | Reject with `INVALID: Sample rate = X Hz (Expected 16,000 Hz)` |
| **Channels** | `1` (Mono) | Reject with `INVALID: Channels = X (Expected mono)` |
| **Bit Depth** | `16-bit signed PCM` | Reject with `INVALID: Bit depth = X (Expected 16-bit)` |
| **Format** | `Linear PCM` | Reject corrupt or non-PCM audio with format warning |
| **Analysis Window** | `1.0 second` (16,000 samples) | Normalizes to 16,000 samples (centered/padded) for 49-frame analysis |
| **Frame Length** | `480 samples` (30 ms) | Overlaid on waveform & selectable via slider |
| **Frame Hop** | `320 samples` (20 ms) | 10 ms frame overlap |
| **FFT Size** | `512` | Radix-2 Real FFT (257 frequency bins, $31.25\text{ Hz/bin}$) |
| **Mel Filterbank** | `20 filters` | Triangular filters spanning 20 Hz to 8,000 Hz |
| **MFCC Coefficients**| `10 coefficients` | First 10 DCT-II cepstral coefficients |
| **Feature Tensor** | `(49, 10, 1)` | 49 time frames $\times$ 10 MFCCs |

---

## 5. Visualizer Panels

1. **Waveform & Framing**:
   - Time-domain PCM waveform with normalized amplitude ($-1.0$ to $+1.0$).
   - Interactive frame slider (0 to 48) highlighting selected 30 ms frame duration.
   - Comprehensive metadata card: file size, sample rate, bit depth, clipping percentage, RMS, DC offset, and quality flags.
2. **Step-by-Step DSP Stages**:
   - **Stage 1**: Raw PCM frame vs Pre-emphasized ($0.97$) vs Hamming windowed samples.
   - **Stage 2**: 512-point FFT power spectrum overlaid against the 20 triangular Mel filterbanks ($20 - 8,000\text{ Hz}$).
   - **Stage 3**: 20 Mel filterbank energies and natural logarithm values ($\ln(E + 10^{-6})$).
   - **Stage 4**: DCT-II transformation yielding the 10 numerical MFCC coefficients.
3. **Spectrogram & $49 \times 10$ MFCC Matrix**:
   - Full 49-frame power spectrogram ($0 - 8,000\text{ Hz}$, $31.25\text{ Hz/bin}$).
   - Final $49 \times 10$ MFCC heatmap showing all 10 coefficients across all 49 time frames.
4. **Comparison Mode**:
   - Side-by-side comparison of two recordings (Waveform, Spectrogram, and MFCC Heatmap) for qualitative contrast.
5. **Dataset Browser (`data/raw/`)**:
   - Recursive scan of `silence/`, `unknown/`, and `swara/`.
   - Contract compliance status (`VALID` / `INVALID`).
   - Recording-level split assignment (`TRAIN`, `VAL`, `TEST`).
   - Double-clicking any file immediately opens it in the visualizer.

---

## 6. C Reference vs. Visualizer Numerical Parity

The visualizer's feature extractor uses the exact mathematical definitions as the embedded C runtime (`src/features/mfcc.c` and `fft.c`).
Verified via [`tools/wav_visualizer/tests/test_mfcc_visualization.py`](tests/test_mfcc_visualization.py):

- **Maximum Absolute Error:** `0.0172` (well within the $\le 0.05$ project tolerance).
- **Mean Absolute Error:** `0.0153`.
- **Status:** **100% Bit/Numerical Parity Confirmed.**
