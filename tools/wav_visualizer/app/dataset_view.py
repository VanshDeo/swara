"""
Dataset Browser and File Management View (Desktop Tooling).
Allows scanning data/raw/ directories, filtering classes, checking recording-level splits,
and viewing quality flags across audio collections.
"""

import os
import glob
from typing import List, Dict, Any, Optional
from .models import AudioMetadata
from .wav_loader import inspect_wav_header

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "training")))
try:
    from dataset import SwaraDataset, inspect_dataset
    from config import CLASSES
except ImportError:
    SwaraDataset = None
    inspect_dataset = None
    CLASSES = ["silence", "unknown", "swara"]


class DatasetBrowserModel:
    """Scans and manages dataset audio files with recording-level split labeling."""

    def __init__(self, root_dir: str = "data/raw"):
        self.root_dir = os.path.abspath(root_dir)
        self.file_records: List[Dict[str, Any]] = []
        self.audit_summary: Dict[str, Any] = {}

    def scan_directory(self, target_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        if target_dir:
            self.root_dir = os.path.abspath(target_dir)

        self.file_records.clear()
        if not os.path.exists(self.root_dir):
            return []

        # Find all WAV files recursively
        all_wavs = []
        for root, _, files in os.walk(self.root_dir):
            for f in sorted(files):
                if f.lower().endswith(".wav"):
                    all_wavs.append(os.path.join(root, f))

        for fpath in all_wavs:
            meta = inspect_wav_header(fpath)

            # Determine class
            assigned_class = "unassigned"
            path_lower = fpath.lower().replace("\\", "/")
            for c in CLASSES:
                if f"/{c}/" in path_lower or path_lower.endswith(f"/{c}"):
                    assigned_class = c
                    break

            # Recording-level split
            split_label = "train"
            if SwaraDataset:
                split_label = SwaraDataset.get_recording_split(fpath)

            record = {
                "filepath": fpath,
                "filename": os.path.basename(fpath),
                "class": assigned_class,
                "duration_sec": meta.duration_sec,
                "sample_rate": meta.sample_rate,
                "channels": meta.channels,
                "bit_depth": meta.bit_depth,
                "is_valid": meta.is_valid_swara_contract,
                "validation_reasons": meta.validation_reasons,
                "split": split_label.upper(),  # RECORDING-LEVEL SPLIT
            }
            self.file_records.append(record)

        # Run summary audit if tool available
        if inspect_dataset:
            try:
                self.audit_summary = inspect_dataset(self.root_dir)
            except Exception:
                self.audit_summary = {}

        return self.file_records

    def filter_by_class(self, class_name: str) -> List[Dict[str, Any]]:
        if class_name.lower() == "all":
            return self.file_records
        return [r for r in self.file_records if r["class"].lower() == class_name.lower()]

    def filter_by_split(self, split_name: str) -> List[Dict[str, Any]]:
        if split_name.lower() == "all":
            return self.file_records
        return [r for r in self.file_records if r["split"].lower() == split_name.lower()]
