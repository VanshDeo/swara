"""
CLI tool for Swara dataset inspection and audit.

Usage:
    python training/inspect_dataset.py --data_dir data/raw
    python training/inspect_dataset.py --data_dir /path/to/google_drive_dataset
"""

import sys
import os
import argparse
import numpy as np

# Ensure training directory is in Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from dataset import inspect_dataset
from config import CLASSES


def main():
    parser = argparse.ArgumentParser(description="Swara Dataset Integrity Inspector & Audit Tool")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Directory containing audio dataset")
    args = parser.parse_args()

    print("=" * 68)
    print("           Swara Dataset Audit & Integrity Report                ")
    print("=" * 68)
    print(f"Target Directory: {os.path.abspath(args.data_dir)}\n")

    report = inspect_dataset(args.data_dir)

    if "error" in report:
        print(f"[FATAL AUDIT ERROR] {report['error']}")
        sys.exit(1)

    print(f"1. Total WAV Files Found:        {report['total_wav_files']}")
    print(f"2. Valid WAV Files:              {report['valid_wav_files']}")
    print(f"3. Usable Recordings (Contract): {report['usable_recordings']} (16kHz, mono, 16-bit PCM)")
    print(f"4. Invalid / Corrupt Files:      {len(report['invalid_corrupt_files'])}")
    if report["invalid_corrupt_files"]:
        for item in report["invalid_corrupt_files"][:10]:
            print(f"     - {item['file']}: {item['error']}")
        if len(report["invalid_corrupt_files"]) > 10:
            print(f"     ... and {len(report['invalid_corrupt_files']) - 10} more.")

    print("\n5. Class Breakdown & Percentages:")
    print(f"     {'Class':<10} {'Status':<10} {'Recordings':<12} {'% Files':<10} {'Duration':<18} {'% Duration':<10}")
    print("     " + "-" * 72)
    for cls_name in CLASSES:
        info = report["class_breakdown"][cls_name]
        valid = info["valid"]
        dur = info["duration_sec"]
        status = "[OK]" if valid > 0 else "[MISSING]"
        p_files = f"{info['percent_files']:.1f}%"
        p_dur = f"{info['percent_duration']:.1f}%"
        dur_str = f"{dur/60.0:.2f}m ({dur:.1f}s)"
        print(f"     {cls_name:<10} {status:<10} {valid:<12} {p_files:<10} {dur_str:<18} {p_dur:<10}")

    print(f"\n     * Class Imbalance Ratio:   {report['class_imbalance_ratio']:.2f}x")

    print("\n6. Recording-Level Partition Counts:")
    splits = report["recording_level_splits"]
    print(f"     * Train:      {splits['train']} recordings")
    print(f"     * Validation: {splits['val']} recordings")
    print(f"     * Test:       {splits['test']} recordings")
    print("     (Note: Partitioning is strictly recording-level and NOT speaker-independent)")

    print("\n7. Audio Specification V0 Compliance:")
    if report["sample_rates"]:
        for sr, count in sorted(report["sample_rates"].items()):
            comp = "[COMPLIANT]" if sr == 16000 else "[NON-COMPLIANT]"
            print(f"     * Sample Rate {sr} Hz:  {count} files {comp}")
        for ch, count in sorted(report["channel_counts"].items()):
            ch_str = "Mono (1-ch)" if ch == 1 else f"{ch}-ch"
            comp = "[COMPLIANT]" if ch == 1 else "[NON-COMPLIANT]"
            print(f"     * Channels {ch_str}:     {count} files {comp}")
        for bd, count in sorted(report["bit_depths"].items()):
            comp = "[COMPLIANT]" if bd == 16 else "[NON-COMPLIANT]"
            print(f"     * Bit Depth {bd}-bit:      {count} files {comp}")

    print("\n8. Duration Statistics:")
    d_stats = report["duration_stats"]
    if report["durations_sec"]:
        print(f"     * Min Duration:     {d_stats['min_sec']:.3f} s")
        print(f"     * Max Duration:     {d_stats['max_sec']:.3f} s")
        print(f"     * Mean Duration:    {d_stats['mean_sec']:.3f} s")
        print(f"     * Median Duration:  {d_stats['median_sec']:.3f} s")
        print(f"     * Std Dev:          {d_stats['std_sec']:.3f} s")
        print(f"     * Files < 1.0s:     {report['files_shorter_than_1s']} (padded)")
        print(f"     * Files == 1.0s:    {report['files_exact_1s']}")
        print(f"     * Files > 1.0s:     {report['files_longer_than_1s']} (centered/windowed)")
    else:
        print("     (No audio files found)")

    print("\n9. Duplication & Leakage Analysis:")
    print(f"     * Duplicate Copies: {report['duplicate_files_count']}")
    if report["duplicate_groups"]:
        for grp in report["duplicate_groups"][:5]:
            print(f"         SHA-256 {grp['sha256'][:16]}...: {len(grp['copies'])} identical copies")

    print("\n" + "=" * 68)
    print("                 DATASET SUFFICIENCY REPORT                      ")
    print("=" * 68)
    print(f"  {'Class':<10} | {'Recordings':<10} | {'Duration':<12} | {'Train':<6} | {'Val':<6} | {'Test':<6}")
    print("  " + "-" * 60)
    for c in CLASSES:
        info = report["class_breakdown"][c]
        d_str = f"{info['duration_sec']:.1f}s"
        print(f"  {c:<10} | {info['valid']:<10} | {d_str:<12} | {info['train_count']:<6} | {info['val_count']:<6} | {info['test_count']:<6}")

    suff = report["sufficiency"]
    if suff["warnings"]:
        print("\n  Sufficiency Warnings:")
        for w in suff["warnings"]:
            print(f"    [!] {w}")

    if suff["deficiencies"]:
        print("\n  Specific Deficiencies:")
        for d in suff["deficiencies"]:
            print(f"    [x] {d}")

    print("\n" + "=" * 68)
    if suff["is_sufficient"]:
        print(f">> AUDIT VERDICT: {suff['verdict']}")
    else:
        print(f">> AUDIT VERDICT: {suff['verdict']}")
    print("=" * 68)

    if not suff["is_sufficient"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
