"""
Swara Dataset Curation & Balance Script.

1. Extracts 35 diverse negative speech samples per word (280 total) from
   data/mini_speech_commands.zip into data/raw/unknown/.
2. Quarantines noisy non-silence files from data/raw/silence/ (RMS > 1000).
3. Slices clean 4-second silence files into 1.0-second compliant WAVs.
4. Audits the final dataset balance across all 3 classes (silence, unknown, swara).
"""

import os
import glob
import wave
import zipfile
import shutil
import numpy as np


def extract_negative_speech_commands(zip_path: str = "data/mini_speech_commands.zip", target_dir: str = "data/raw/unknown", count_per_word: int = 35):
    """Extract diverse negative words from mini_speech_commands.zip into target_dir."""
    os.makedirs(target_dir, exist_ok=True)
    words = ["down", "go", "left", "no", "right", "stop", "up", "yes"]
    extracted = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        for word in words:
            prefix = f"mini_speech_commands/{word}/"
            word_files = [f for f in namelist if f.startswith(prefix) and f.endswith(".wav") and not os.path.basename(f).startswith("._")]
            selected = word_files[:count_per_word]
            for member in selected:
                fname = f"unknown_sc_{word}_{os.path.basename(member)}"
                dest_path = os.path.join(target_dir, fname)
                with zf.open(member) as src, open(dest_path, "wb") as dst:
                    dst.write(src.read())
                extracted += 1

    print(f"[Speech Commands] Extracted {extracted} negative speech files into {target_dir}")


def clean_and_slice_silence(silence_dir: str = "data/raw/silence", rms_thresh: float = 1000.0):
    """
    Quarantines noisy files (RMS > rms_thresh) and slices clean long files
    into 1.0s (16,000 samples) non-overlapping chunks.
    """
    quarantine_dir = os.path.join("data", "raw", ".noisy_silence_quarantine")
    os.makedirs(quarantine_dir, exist_ok=True)

    files = [f for f in glob.glob(f"{silence_dir}/*.wav") if not os.path.basename(f).startswith("._")]
    quarantined = 0
    sliced_count = 0

    for f in files:
        if "_slice" in os.path.basename(f):
            continue  # Already sliced in previous run

        try:
            with wave.open(f, "rb") as wf:
                sr = wf.getframerate()
                ch = wf.getnchannels()
                sw = wf.getsampwidth()
                num_frames = wf.getnframes()
                raw = wf.readframes(num_frames)
                samples = np.frombuffer(raw, dtype=np.int16)

            rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
            if rms > rms_thresh:
                # Move to quarantine
                shutil.move(f, os.path.join(quarantine_dir, os.path.basename(f)))
                quarantined += 1
                continue

            # If clean and longer than 1.5 seconds, slice into 1.0s chunks
            if len(samples) >= 32000:  # >= 2.0s
                base_name = os.path.splitext(os.path.basename(f))[0]
                num_slices = len(samples) // 16000
                for s_idx in range(num_slices):
                    chunk = samples[s_idx * 16000 : (s_idx + 1) * 16000]
                    chunk_rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
                    if chunk_rms <= rms_thresh:
                        slice_name = f"{base_name}_slice{s_idx}.wav"
                        slice_path = os.path.join(silence_dir, slice_name)
                        with wave.open(slice_path, "wb") as out_wf:
                            out_wf.setnchannels(1)
                            out_wf.setsampwidth(2)
                            out_wf.setframerate(16000)
                            out_wf.writeframes(chunk.tobytes())
                        sliced_count += 1
                # Remove original long file after slicing
                os.remove(f)

        except Exception as e:
            print(f"[Warning] Error processing {f}: {e}")

    print(f"[Silence] Quarantined {quarantined} noisy files, created {sliced_count} 1.0s silence slices.")


def audit_final_dataset(data_dir: str = "data/raw"):
    """Print the final balance of files per class."""
    print("=" * 60)
    print("Final Dataset Audit:")
    for cls in ["silence", "unknown", "swara"]:
        files = [f for f in glob.glob(f"{data_dir}/{cls}/*.wav") if not os.path.basename(f).startswith("._")]
        print(f"  * Class '{cls}': {len(files)} files")
    print("=" * 60)


if __name__ == "__main__":
    extract_negative_speech_commands()
    clean_and_slice_silence()
    audit_final_dataset()
