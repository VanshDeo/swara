"""
Swara Physical Dataset Split Generator.

Partitions recordings from data/raw/ into physical directories:
    data/train/<class>/
    data/val/<class>/
    data/test/<class>/

Uses deterministic content/recording-level hashing to prevent data leakage.
Source files in data/raw/ are preserved untouched.
"""

import os
import sys
import shutil
import hashlib
from typing import Dict, List, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "training"))

from config import CLASSES
from dataset import SwaraDataset

get_recording_split = SwaraDataset.get_recording_split


def prepare_physical_splits(raw_dir: str = "data/raw", base_data_dir: str = "data") -> Tuple[Dict[str, Dict[str, int]], int]:
    """
    Scans raw_dir, performs duplicate detection across content hashes,
    and cleanly rebuilds base_data_dir/train, val, test using deterministic
    recording-level content hashing.
    """
    raw_path = os.path.abspath(raw_dir)
    base_path = os.path.abspath(base_data_dir)

    splits = ["train", "val", "test"]
    stats: Dict[str, Dict[str, int]] = {s: {c: 0 for c in CLASSES} for s in splits}

    # 1. Clean existing physical split directories to ensure clean rebuild
    for s in splits:
        for c in CLASSES:
            target_dir = os.path.join(base_path, s, c)
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            os.makedirs(target_dir, exist_ok=True)

    # 2. Collect all raw files and perform duplicate detection via SHA-256
    content_to_files: Dict[str, List[Tuple[str, str, str]]] = {}  # sha256 -> [(src_file, class_name, filename)]
    
    for root, dirs, files in os.walk(raw_path):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in sorted(files):
            if not f.lower().endswith(".wav"):
                continue

            src_file = os.path.join(root, f)
            rel_to_raw = os.path.relpath(src_file, raw_path).replace("\\", "/")

            matched_class = None
            for c in CLASSES:
                if rel_to_raw.startswith(f"{c}/") or f"/{c}/" in rel_to_raw or rel_to_raw == f"{c}":
                    matched_class = c
                    break

            if not matched_class and "swara" in rel_to_raw.lower():
                matched_class = "swara"

            if not matched_class:
                continue

            # Compute SHA-256
            hasher = hashlib.sha256()
            with open(src_file, "rb") as rf:
                while chunk := rf.read(65536):
                    hasher.update(chunk)
            sha256_hash = hasher.hexdigest()

            content_to_files.setdefault(sha256_hash, []).append((src_file, matched_class, f))

    duplicate_copies_count = 0
    assigned_hashes: Dict[str, str] = {}  # sha256 -> assigned split name

    # 3. Partition deterministically, guaranteeing identical content shares the same split
    for sha256_hash, file_list in sorted(content_to_files.items()):
        if len(file_list) > 1:
            duplicate_copies_count += (len(file_list) - 1)
            print(f"[DUPLICATE DETECTED] SHA-256 {sha256_hash[:16]} has {len(file_list)} copies across raw dataset.")

        # Determine split directly from the content SHA-256 hash
        target_split = get_recording_split(sha256_hash)
        assigned_hashes[sha256_hash] = target_split

        for src_file, matched_class, f_name in file_list:
            dest_folder = os.path.join(base_path, target_split, matched_class)
            dest_file = os.path.join(dest_folder, f_name)
            shutil.copy2(src_file, dest_file)
            stats[target_split][matched_class] += 1

    return stats, duplicate_copies_count


def main():
    raw_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    base_dir = sys.argv[2] if len(sys.argv) > 2 else "data"

    print("================================================================")
    print("          Swara Physical Dataset Split Generator                ")
    print("================================================================")
    print(f"Source:      {os.path.abspath(raw_dir)}")
    print(f"Destination: {os.path.abspath(base_dir)}/[train, val, test]\n")

    stats, dup_count = prepare_physical_splits(raw_dir, base_dir)

    print(f"Duplicates Detected: {dup_count} (Identical audio content locked to same split)")
    print("-" * 52)
    print(f"{'Split':<10} | {'swara':<8} | {'silence':<8} | {'unknown':<8} | {'Total':<8}")
    print("-" * 52)
    for s in ["train", "val", "test"]:
        swara_count = stats[s]["swara"]
        silence_count = stats[s]["silence"]
        unknown_count = stats[s]["unknown"]
        tot = swara_count + silence_count + unknown_count
        print(f"{s:<10} | {swara_count:<8} | {silence_count:<8} | {unknown_count:<8} | {tot:<8}")
    print("================================================================")
    print(f"Physical splits generated successfully under: {os.path.abspath(base_dir)}")


if __name__ == "__main__":
    main()
