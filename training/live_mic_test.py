"""
Real-time wake word detection test script using MacBook microphone (Swara).

Captures continuous 16kHz mono audio, extracts 49x10 MFCC features in a
sliding 1.0-second window, and runs inference using the quantized INT8
TFLite model (models/swara_int8.tflite) or Keras model.
"""

import os
import sys
import time
import argparse
import queue
import numpy as np

# Ensure training directory is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from dataset import SwaraFeatureExtractor
from config import CLASSES, CLASS_TO_IDX, IDX_TO_CLASS

try:
    import sounddevice as sd
except ImportError:
    print("[ERROR] sounddevice is not installed. Run: pip install sounddevice")
    sys.exit(1)


class SwaraLivePredictor:
    def __init__(self, model_path: str = "models/swara_int8.tflite"):
        self.model_path = model_path
        self.is_tflite = model_path.endswith(".tflite")
        self.extractor = SwaraFeatureExtractor(use_preemphasis=True)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        if self.is_tflite:
            import tensorflow as tf
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()[0]
            self.output_details = self.interpreter.get_output_details()[0]

            self.in_scale, self.in_zero = self.input_details.get("quantization", (0.0, 0))
            self.out_scale, self.out_zero = self.output_details.get("quantization", (0.0, 0))
            self.is_int8 = self.input_details["dtype"] == np.int8
        else:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(model_path)

    def predict(self, window_16k_pcm: np.ndarray) -> np.ndarray:
        """
        Runs inference on 16,000 samples of 16-bit PCM.
        Returns class probabilities for [silence, unknown, swara].
        """
        mfcc = self.extractor.extract_window_mfcc(window_16k_pcm)
        sample = np.expand_dims(mfcc, axis=0)  # Shape: (1, 49, 10, 1)

        if self.is_tflite:
            if self.is_int8 and self.in_scale > 0.0:
                sample_quant = np.clip(np.round(sample / self.in_scale) + self.in_zero, -128, 127).astype(np.int8)
                self.interpreter.set_tensor(self.input_details["index"], sample_quant)
            else:
                self.interpreter.set_tensor(self.input_details["index"], sample.astype(self.input_details["dtype"]))

            self.interpreter.invoke()
            out_tensor = self.interpreter.get_tensor(self.output_details["index"])

            if self.is_int8 and self.out_scale > 0.0:
                raw_out = (out_tensor[0].astype(np.float32) - self.out_zero) * self.out_scale
            else:
                raw_out = out_tensor[0].astype(np.float32)

            # Output is already a probability distribution from Softmax activation
            probs = np.clip(raw_out, 0.0, 1.0)
            total = np.sum(probs)
            if total > 0.0:
                probs = probs / total
            return probs
        else:
            pred = self.model.predict(sample, verbose=0)[0]
            pred = np.clip(pred, 0.0, 1.0)
            total = np.sum(pred)
            if total > 0.0:
                pred = pred / total
            return pred


def render_audio_bar(rms_level: float, max_rms: float = 4000.0, width: int = 10) -> str:
    """ASCII volume meter bar."""
    ratio = min(1.0, max(0.0, rms_level / max_rms))
    filled = int(round(ratio * width))
    return "█" * filled + "░" * (width - filled)


def render_meter(prob: float, width: int = 8) -> str:
    """ASCII percentage meter bar for class probabilities."""
    ratio = min(1.0, max(0.0, prob))
    filled = int(round(ratio * width))
    return "█" * filled + "░" * (width - filled)


def main():
    parser = argparse.ArgumentParser(description="Live MacBook Microphone Wake Word Test for Swara")
    parser.add_argument("--model", type=str, default="models/swara_int8.tflite", help="Path to .tflite or .keras model")
    parser.add_argument("--threshold", type=float, default=0.65, help="Wake word activation threshold (0.0 to 1.0, default 0.65)")
    parser.add_argument("--cooldown", type=float, default=1.2, help="Cooldown in seconds after a wake detection")
    parser.add_argument("--device", type=int, default=None, help="Input device ID (default: system default microphone)")
    parser.add_argument("--step_ms", type=int, default=100, help="Inference update interval in milliseconds (default: 100ms)")
    args = parser.parse_args()

    # Fallback to keras model if int8 doesn't exist
    model_path = args.model
    if not os.path.exists(model_path) and os.path.exists("models/swara_model.keras"):
        print(f"[Notice] {model_path} not found. Falling back to models/swara_model.keras")
        model_path = "models/swara_model.keras"

    print("=" * 80)
    print("           Swara Real-Time Wake Word Microphone Monitor                ")
    print("=" * 80)
    print(f"Model File:          {os.path.abspath(model_path)}")
    print(f"Sample Rate:         16,000 Hz Mono (Signed 16-bit PCM)")
    print(f"Sliding Window:      1.0 second (16,000 samples)")
    print(f"Inference Rate:      Every {args.step_ms} ms (10 frames/sec)")
    print(f"Wake Word:           'Hello Swara' / 'Swara' (Class 2)")
    print(f"Activation Cutoff:   {args.threshold * 100:.1f}% confidence")
    print(f"Trigger Cooldown:    {args.cooldown:.1f} s")
    print("Probability Monitor: [SIL = Silence, UNK = Unknown/Negative, SWR = Swara]")
    print("-" * 80)

    # Initialize predictor
    try:
        predictor = SwaraLivePredictor(model_path=model_path)
    except Exception as e:
        print(f"[FATAL] Failed to initialize model predictor: {e}")
        sys.exit(1)

    # Audio stream parameters
    SAMPLE_RATE = 16000
    WINDOW_SIZE = 16000
    CHUNK_SIZE = int(SAMPLE_RATE * (args.step_ms / 1000.0))

    audio_queue = queue.Queue()
    ring_buffer = np.zeros(WINDOW_SIZE, dtype=np.int16)

    def audio_callback(indata, frames, time_info, status):
        if status:
            pass
        # Indata shape is (frames, 1) int16
        audio_queue.put(indata[:, 0].copy())

    dev_info = sd.query_devices(args.device, kind="input")
    print(f"Audio Input Device:  [{dev_info.get('index', 0)}] {dev_info.get('name')}")
    print("-" * 80)
    print("Listening... Speak 'Hello Swara' into your microphone.")
    print("Press Ctrl+C to stop.\n")

    last_trigger_time = 0.0
    detected_count = 0

    try:
        with sd.InputStream(
            device=args.device,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=CHUNK_SIZE,
            callback=audio_callback,
        ):
            while True:
                # Get newest chunk of audio
                chunk = audio_queue.get()
                chunk_len = len(chunk)

                # Slide ring buffer
                ring_buffer[:-chunk_len] = ring_buffer[chunk_len:]
                ring_buffer[-chunk_len:] = chunk

                # Calculate volume / RMS
                rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
                vol_bar = render_audio_bar(rms, max_rms=4000.0, width=8)

                # Run inference
                probs = predictor.predict(ring_buffer)
                p_silence, p_unknown, p_swara = float(probs[0]), float(probs[1]), float(probs[2])

                now = time.time()
                time_since_trigger = now - last_trigger_time

                # Check if wake word triggered
                if p_swara >= args.threshold and time_since_trigger >= args.cooldown:
                    last_trigger_time = now
                    detected_count += 1
                    print("\n" + "=" * 80)
                    print(f"🔔 [WAKE WORD DETECTED #{detected_count}!] 'SWARA' - Confidence: {p_swara * 100:.1f}%")
                    print(f"   SIL: [{render_meter(p_silence, 12)}] {p_silence*100:5.1f}% | "
                          f"UNK: [{render_meter(p_unknown, 12)}] {p_unknown*100:5.1f}% | "
                          f"SWR: [{render_meter(p_swara, 12)}] {p_swara*100:5.1f}%")
                    print("=" * 80 + "\n")
                else:
                    # Multi-class side-by-side visual progress monitoring
                    status = (
                        f"\rMic [{vol_bar}] | "
                        f"SIL: [{render_meter(p_silence, 8)}] {p_silence*100:5.1f}% | "
                        f"UNK: [{render_meter(p_unknown, 8)}] {p_unknown*100:5.1f}% | "
                        f"SWR: [{render_meter(p_swara, 8)}] {p_swara*100:5.1f}%"
                    )
                    sys.stdout.write(status)
                    sys.stdout.flush()

    except KeyboardInterrupt:
        print(f"\n\n[Stopped] Live microphone monitor exited. Total detections: {detected_count}")


if __name__ == "__main__":
    main()
