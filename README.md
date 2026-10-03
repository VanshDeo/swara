# Swara: Edge Voice & Wake Detection Runtime

**Swara** is an ultra-low-power, deterministic edge voice recognition and wake detection system designed for TinyML microcontrollers and embedded processors (ARM Cortex-M, ESP32, RISC-V).

---

## ⚡ Hard Engineering Constraints

The architecture is strictly designed from day one around embedded physical limits:

* **Implementation:** **C/C++ only** for the actual project and runtime. No runtime interpreter.
* **Target Memory Footprint:** **$\le 256\text{ KB}$** total (Flash code + INT8 weights + Tensor Arena + PCM ring buffer + MFCC scratch + stack).
* **CPU Target:** **$\approx 10\%$ CPU usage while idle** achieved via front-end Voice Activity Detection (VAD) gating.
* **Deployment Target:** Edge / TinyML (bare-metal, FreeRTOS, Zephyr on Cortex-M / RISC-V).
* **Inference Engine:** INT8 Depthwise Separable CNN (DS-CNN).
* **Audio Pipeline:** `PCM → VAD → MFCC → DS-CNN → wake detection`.
* **Zero Dynamic Allocation:** `malloc`/`calloc`/`realloc`/`free` strictly forbidden in runtime audio paths.
* **Role of Python:** **Not part of the deployed system.** Python is strictly optional external tooling for offline dataset curation, model training, and INT8 quantization export.

---

## 📁 Project Architecture

```text
swara/
│
├── src/                # PRIMARY RUNTIME: Native C embedded implementation
│   ├── audio/
│   │   ├── audio_buffer.c  # Circular ring buffer for 16-bit PCM sliding window
│   │   ├── audio_buffer.h  # Ring buffer API & frozen V0 constants
│   │   ├── vad.c           # Lightweight energy/ZCR voice activity gate
│   │   ├── vad.h           # VAD header, thresholds & hangover state
│   │   ├── wav_reader.c    # Minimal zero-allocation 16kHz 16-bit mono WAV reader
│   │   └── wav_reader.h    # WAV header parser & validation interface
│   │
│   ├── features/
│   │   ├── fft.c           # 512-point Radix-2 Cooley-Tukey FFT & Hamming window
│   │   ├── fft.h           # FFT API, twiddle factors & power spectrum
│   │   ├── mfcc.c          # 20 Mel filterbanks & 10 DCT-II cepstral coefficients
│   │   └── mfcc.h          # MFCC extraction API & full 49x10 window extractor
│   │
│   ├── model/
│   │   ├── classifier.c    # [Upcoming] INT8 TFLM model execution & thresholding
│   │   └── classifier.h    # [Upcoming] Classifier interface
│   │
│   └── main.c              # [Upcoming] Embedded application loop & wake event handler
│
├── tests/              # Native C verification & unit test suites
│   ├── data/
│   │   └── test_16k_1s.wav # Deterministic 1-second 16kHz mono 16-bit test WAV fixture
│   ├── test_audio_buffer.c # Ring buffer capacity, extraction & wrap-around tests
│   ├── test_vad.c          # VAD energy threshold, silence rejection & speech trigger
│   ├── test_wav.c          # WAV header validation, format rejection & sample reading
│   ├── test_fft.c          # FFT twiddle, bit-reversal & sine tone frequency tests
│   ├── test_mfcc.c         # Mel filterbank, log-compression & DCT tests
│   ├── test_pipeline.c     # End-to-end Silence, 1kHz tone, and formant speech tests
│   ├── test_wav_mfcc.c     # Milestone M1b: Full WAV -> 49x10 MFCC pipeline validation
│   ├── test_feature_parity.py # Numerical verification between C and Python MFCC
│   └── test_training_infrastructure.py # Python training pipeline unit tests
│
├── models/             # Frozen model binaries & deployment C arrays
│   ├── swara_float32.tflite # Baseline Float32 model (48.9 KB)
│   ├── swara_int8.tflite    # Fully quantized INT8 model for microcontrollers (25.3 KB)
│   ├── swara_saved_model/   # Best checkpoint SavedModel bundle
│   └── training_history.csv # Epoch-by-epoch loss & accuracy history log
│
├── deployment/         # Core C/C++ runtime & deployment artifacts for TFLM
│   ├── model_data.cc   # 16-byte aligned C++ byte array definition for TFLite Micro
│   ├── model_data.h    # C++ header declaring external model array & length
│   └── README.md       # TFLM deployment pipeline, model contract & memory budget
│
├── training/           # OPTIONAL external tooling (Python offline development)
│   ├── config.py       # Centralized hyperparameters (channels, classes, shapes)
│   ├── dataset.py      # Audio loading, peak-energy alignment & deterministic splits
│   ├── model.py        # DS-CNN (64 filters, 3 classes: silence, unknown, swara)
│   ├── train.py        # Model training loop, callbacks, saved model export
│   ├── quantize.py     # Full INT8 quantization with real representative dataset
│   ├── evaluate.py     # Recording-level evaluation (accuracy, F1, confusion matrix)
│   ├── validate_tflite.py # TFLite model inspector & TFLM operator validator
│   ├── export_model_header.py # Deterministic TFLite to C array exporter
│   ├── inspect_dataset.py # Standalone dataset audit & integrity analyzer
│   ├── import_drive_dataset.py # Google Drive WAV incremental import & sync utility
│   └── live_mic_test.py # Real-time MacBook microphone monitor with 3-class meter
│
├── data/               # Offline training & validation datasets
│   ├── raw/            # Audio recordings (.wav) by class (silence, unknown, swara)
│   ├── processed/      # Extracted spectrograms, MFCC features
│   └── augmented/      # Synthetic/augmented data (noise injection, time shift)
│
├── CMakeLists.txt      # Root build configuration for runtime & test suites
├── architecture.md     # Architecture specifications, hard constraints & change log
└── README.md           # Project documentation and developer reference
```

---

## 🎙️ Frozen Audio Specification (V0)

The audio front-end and feature extraction parameters are frozen for V0:

| Parameter | Specification | Notes |
| :--- | :--- | :--- |
| **Sample Rate** | `16,000 Hz` | 16 kHz acoustic bandwidth |
| **Channels** | `1` | Mono channel |
| **Sample Format** | `signed 16-bit PCM` | Standard linear PCM (`int16_t`, -32768 to +32767) |
| **Window Duration** | `1,000 ms` | 1.0 second duration |
| **Samples / Window** | `16,000` | 16,000 samples @ 16 kHz |
| **Frame Length** | `30 ms` | 480 samples per FFT analysis frame |
| **Frame Step (Hop)** | `20 ms` | 320 samples hop size (10 ms frame overlap) |
| **Pre-emphasis** | `0.97` | High-frequency acoustic boost: $y[n] = x[n] - 0.97 \cdot x[n-1]$ |
| **Window Function** | **Hamming** | $w[n] = 0.54 - 0.46 \cos(2\pi n / 479)$ |
| **FFT Size** | `512` | Radix-2 FFT size covering 480 samples (32 zero-padded) |
| **Mel Filters** | `20` | Triangular Mel filterbanks spanning 20 Hz to 8,000 Hz |
| **MFCC Coefficients** | `10` | First 10 discrete cosine transform (DCT-II) coefficients |
| **Feature Tensor Shape** | `(49, 10, 1)` | 49 time frames × 10 MFCCs × 1 channel (490 values) |

---

## 📊 Front-End Performance & Memory Verification

Measured via [`benchmark_pipeline`](tests/benchmark_pipeline.c) on host machine:

| Metric | Measured Benchmark | Notes |
| :--- | :--- | :--- |
| **VAD Gating Check (20 ms frame)** | **`0.15 µs`** (0.00015 ms) | **6.62M frames / sec** |
| **VAD Idle CPU Duty Cycle** | **`0.0008%`** | Far below the hard $\approx 10\%$ CPU idle budget |
| **Buffer Ingestion (320-sample hop)** | **`2.54 µs`** | Real-time sliding window write |
| **Single Frame MFCC (30 ms / 480 spl)** | **`10.03 µs`** (0.010 ms) | **2,991× faster than real-time** |
| **1-Second Active Window (49 frames)** | **`0.458 ms`** | **2,183× faster than real-time (RTF 1:2183)** |
| **Active Speech CPU Duty Cycle** | **`0.046%`** | Ultra-low power profile |
| **Streaming Pipeline (50% speech / 50% silence)** | **`0.613 ms` / 1s audio** | 4,900 silence frames bypassed via VAD |
| **Total Front-End Static RAM** | **`60,444 bytes`** (~59.03 KB) | Ring buffer + VAD + Mel + FFT + 49x10 matrix |
| **Peak Stack Scratch Memory** | **`6,084 bytes`** (~5.94 KB) | 512-pt complex FFT + power spectrum |
| **Dynamic Allocations (`malloc`)** | **`0 bytes`** | Zero dynamic heap allocation |

---

## 🧠 Model Training & Quantization Results

The real acoustic dataset (666 audio recordings) has been ingested, energy-aligned, augmented, trained, and fully quantized to signed 8-bit integer (`int8`):

### 1. Test Split Benchmark Results ([`training/evaluate.py`](training/evaluate.py))

Evaluated across 112 unseen recordings in the test split:

| Metric | Validation Split (98 samples) | Test Split (112 samples) |
| :--- | :--- | :--- |
| **Overall Accuracy** | **94.90%** | **93.75%** |
| **Macro Precision** | **95.12%** | **88.89%** |
| **Macro Recall** | **94.50%** | **93.61%** |
| **Macro F1 Score** | **94.75%** | **90.74%** |
| **Silence Recall** | 96.0% (24/25) | **97.8% (44/45)** |
| **Unknown Speech Recall**| 95.8% (46/48) | **90.7% (49/54)** |
| **Swara Wake Phrase Recall** | **96.0% (24/25)** | **92.3% (12/13)** |

**Test Split Confusion Matrix:**
```text
            silence  unknown  swara    
  silence   44       1        0      
  unknown   0        49       5      
  swara     0        1        12     
```

### 2. INT8 Quantized Model Verification ([`training/validate_tflite.py`](training/validate_tflite.py))
* **Model File:** `models/swara_int8.tflite`
* **File Size:** **25,904 bytes** (25.30 KB — well within the 45–70 KB Flash budget)
* **Weight Storage:** **12,292 bytes** (12.00 KB)
* **Input Tensor:** `[1, 49, 10, 1]` INT8 (Scale: 1.2979, Zero-Point: 85)
* **Output Tensor:** `[1, 3]` INT8 (Scale: 0.00390625, Zero-Point: -128)
* **Operators (8 ops):** `CONV_2D`, `DEPTHWISE_CONV_2D`, `FULLY_CONNECTED`, `MEAN`, `SOFTMAX`
* **TFLM Compatibility:** **100% compatible** with TensorFlow Lite Micro.

---

## 🛠️ Key Architectural Enhancements & Fixes

1. **Peak Energy-Aligned Sliding Window:**
   Replaced blind center-cropping (`(len - 16000)//2`) with `find_peak_energy_window(step=320)`. In 3–4 second recordings, it automatically locates the 1.0-second window containing the peak acoustic energy of "Hello Swara", eliminating corrupted silent training samples.
2. **Batch Normalization Momentum Alignment:**
   Updated all Depthwise Separable Conv blocks with `BatchNormalization(momentum=0.80)`. On edge datasets with ~500–1000 samples, default momentum (0.99) caused running statistics to diverge, collapsing inference accuracy to ~58%. Setting momentum to 0.80 aligns running statistics in < 15 epochs.
3. **Decoupled Logits & Softmax Layers:**
   The classification head cleanly separates `Dense(3, activation=None, name="logits")` (Op 6: `FULLY_CONNECTED`) from `Softmax(name="output")` (Op 7: `SOFTMAX`). This resolves double-softmax probability collapse in downstream monitors.
4. **Multi-Slice Jitter & Noise Augmentation:**
   Expands target wake recordings with temporal jitter offsets ($-150\text{ ms}$, $-75\text{ ms}$, $+75\text{ ms}$, $+150\text{ ms}$) and subtle background noise injection (SNR ~25 dB).
5. **Ghost File Hard-Filtering:**
   Hardened dataset loaders to reject macOS AppleDouble resource fork files (`._*.wav`) and files smaller than 44 bytes.

---

## 🚀 Quickstart & Workflow Commands

### 1. Incremental Google Drive Sync
To download **only newly uploaded recordings** from your Google Drive folders without re-downloading existing files:
```bash
# Set your links in .env, then run:
PYTHONPATH=training .venv/bin/python training/import_drive_dataset.py --all
```

### 2. Fast 1-Liner: Sync, Train, Quantize, Export & Benchmark
Run the complete pipeline end-to-end in a single command:
```bash
PYTHONPATH=training .venv/bin/python training/import_drive_dataset.py --all && \
find data/raw -name "._*" -delete && \
PYTHONPATH=training .venv/bin/python training/train.py --epochs 35 --batch_size 32 && \
PYTHONPATH=training .venv/bin/python training/quantize.py --model_path models/swara_saved_model/best_val_model.keras --output_int8 models/swara_int8.tflite --output_float32 models/swara_float32.tflite && \
PYTHONPATH=training .venv/bin/python training/export_model_header.py --tflite_path models/swara_int8.tflite && \
PYTHONPATH=training .venv/bin/python training/evaluate.py --model_path models/swara_int8.tflite --split test
```

### 3. Real-Time Live Microphone Monitor
Test wake word detection live using your computer's microphone with side-by-side probability bars:
```bash
.venv/bin/python training/live_mic_test.py --model models/swara_int8.tflite --threshold 0.65
```

Terminal display output:
```text
Mic [████░░░░] | SIL: [████████░░] 82.5% | UNK: [█░░░░░░░░░] 14.2% | SWR: [░░░░░░░░░░]  3.3%

================================================================================
🔔 [WAKE WORD DETECTED #1!] 'SWARA' - Confidence: 88.4%
   SIL: [░░░░░░░░░░░░]   1.2% | UNK: [█░░░░░░░░░░░]  10.4% | SWR: [██████████░░]  88.4%
================================================================================
```

### 4. Build & Run Native C Tests (CMake)
```bash
cmake -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

---

## 💾 System Memory Footprint ($\le 256\text{ KB}$ Budget)

| Subsystem Component | Memory Type | Allocation | Status | Description |
| :--- | :--- | :--- | :--- | :--- |
| **1-sec PCM Audio Buffer** | Static RAM | `32,000 bytes` (31.25 KB) | **MEASURED** | `int16_t[16000]` circular sliding buffer |
| **MFCC Configuration & Tables** | Static RAM | `26,436 bytes` (25.82 KB) | **MEASURED** | 20 Mel filterbanks, FFT tables, DCT matrix |
| **49×10 Feature Buffer** | Static RAM | `1,960 bytes` (1.91 KB) | **MEASURED** | 49 frames × 10 MFCCs (`float32[490]`) |
| **VAD Engine State** | Static RAM | `16 bytes` (0.02 KB) | **MEASURED** | Energy thresholds and hangover state |
| **Front-End Call Stack Scratch**| Stack Memory| `6,084 bytes` (5.94 KB) | **MEASURED** | FFT scratch buffers & power spectrum |
| **TFLM Tensor Arena (64-ch)** | Heap / Arena| `~36,960 bytes` (~36.09 KB)| **ESTIMATED** | Activations working arena (DS-CNN 64-channel)|
| **Firmware Stack & RTOS Overhead**| Stack/BSS  | `~15,360 bytes` (~15.00 KB)| **ESTIMATED** | FreeRTOS task stacks, interrupt stack, BSS |
| **Application State & Queues** | Static RAM | `~4,096 bytes` (~4.00 KB) | **ESTIMATED** | Classification event flags, IPC queues |
| **TOTAL SYSTEM RAM (64-ch)** | **Total RAM** | **~122,912 bytes (~120.0 KB)** | **ESTIMATED** | **$\le 256\text{ KB}$ Compliant (~136 KB Headroom)** |

*Quantized model weights (**25.30 KB** INT8 binary, 12.00 KB weight parameters) reside in Flash ROM (`alignas(16) const unsigned char g_swara_model_data[]` in [deployment/model_data.cc](deployment/model_data.cc)) and consume 0 bytes of system RAM.*

---

## 🏛️ Architecture & Governance

Detailed technical specifications, component data flows, memory budgets, and architectural decision records are maintained in [architecture.md](architecture.md).
