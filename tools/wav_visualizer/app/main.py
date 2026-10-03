"""
Swara WAV / MFCC Visualization & Debugging Engine (Desktop Tooling).
Windows-only development and audio inspection tool.
"""

import os
import sys
import argparse
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

if __package__ is None or __package__ == "":
    tools_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    from tools.wav_visualizer.app.models import ProcessedAudio
    from tools.wav_visualizer.app.wav_loader import inspect_wav_header
    from tools.wav_visualizer.app.signal_processing import VisualizerPipeline
    from tools.wav_visualizer.app.mfcc_view import VisualizerPlotEngine
    from tools.wav_visualizer.app.dataset_view import DatasetBrowserModel
else:
    from .models import ProcessedAudio
    from .wav_loader import inspect_wav_header
    from .signal_processing import VisualizerPipeline
    from .mfcc_view import VisualizerPlotEngine
    from .dataset_view import DatasetBrowserModel


class SwaraVisualizerApp(tk.Tk):
    """Main desktop application window for Swara WAV/MFCC inspection."""

    def __init__(self, initial_file: str = None, dataset_dir: str = "data/raw"):
        super().__init__()
        self.title("Swara WAV & MFCC Visualization Engine [WINDOWS DEVELOPMENT TOOL ONLY]")
        self.geometry("1400x900")
        self.minsize(1050, 700)

        self.pipeline = VisualizerPipeline()
        self.dataset_model = DatasetBrowserModel(dataset_dir)
        self.current_audio: ProcessedAudio = None
        self.comparison_audio: ProcessedAudio = None
        self.selected_frame_idx: int = 0

        self._build_ui()

        # Load initial file if provided
        if initial_file and os.path.exists(initial_file):
            self.load_audio_file(initial_file)
        elif os.path.exists("tests/data/test_16k_1s.wav"):
            self.load_audio_file("tests/data/test_16k_1s.wav")

    def _build_ui(self):
        # Top Header Bar
        top_bar = ttk.Frame(self, padding=5)
        top_bar.pack(side=tk.TOP, fill=tk.X)

        btn_open = ttk.Button(top_bar, text="Open WAV File...", command=self._on_open_file)
        btn_open.pack(side=tk.LEFT, padx=5)

        btn_open_ds = ttk.Button(top_bar, text="Open Dataset Dir...", command=self._on_open_dataset)
        btn_open_ds.pack(side=tk.LEFT, padx=5)

        btn_compare = ttk.Button(top_bar, text="Load Comparison WAV...", command=self._on_load_comparison)
        btn_compare.pack(side=tk.LEFT, padx=5)

        self.lbl_title = ttk.Label(top_bar, text="No file loaded", font=("Segoe UI", 10, "bold"))
        self.lbl_title.pack(side=tk.LEFT, padx=15)

        self.lbl_validity = ttk.Label(top_bar, text="", font=("Segoe UI", 10, "bold"))
        self.lbl_validity.pack(side=tk.RIGHT, padx=10)

        # Notebook (Tabbed Views)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tab 1: Waveform & Frame Breakdown
        self.tab_waveform = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_waveform, text="1. Waveform & Framing")
        self._build_waveform_tab()

        # Tab 2: Full DSP Pipeline Stages
        self.tab_pipeline = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_pipeline, text="2. Step-by-Step DSP Stages")
        self._build_pipeline_tab()

        # Tab 3: Spectrogram & MFCC Heatmap
        self.tab_features = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_features, text="3. Spectrogram & 49x10 MFCC Matrix")
        self._build_features_tab()

        # Tab 4: Comparison Mode (Two Audio Signals)
        self.tab_compare = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_compare, text="4. Comparison Mode")
        self._build_compare_tab()

        # Tab 5: Dataset Browser
        self.tab_dataset = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_dataset, text="5. Dataset Browser (data/raw)")
        self._build_dataset_tab()

        # Bottom Status Bar
        self.status_bar = ttk.Label(self, text="Ready | Windows Development & Debugging Tool Only | Not part of deployment binary",
                                    relief=tk.SUNKEN, anchor=tk.W, font=("Segoe UI", 9))
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _build_waveform_tab(self):
        # Left Panel: Info & Statistics
        left_panel = ttk.LabelFrame(self.tab_waveform, text="Audio Metadata & Statistics", padding=10, width=320)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        left_panel.pack_propagate(False)

        self.txt_metadata = tk.Text(left_panel, wrap=tk.WORD, height=18, font=("Consolas", 9), relief=tk.FLAT)
        self.txt_metadata.pack(fill=tk.BOTH, expand=True)

        # Frame Slider
        slider_frame = ttk.LabelFrame(left_panel, text="Frame Selector (0 to 48)", padding=5)
        slider_frame.pack(fill=tk.X, pady=10)

        self.frame_slider = ttk.Scale(slider_frame, from_=0, to=48, orient=tk.HORIZONTAL, command=self._on_slider_change)
        self.frame_slider.pack(fill=tk.X, padx=5)

        self.lbl_frame_info = ttk.Label(slider_frame, text="Frame 0 (0-30 ms)", font=("Segoe UI", 9, "bold"))
        self.lbl_frame_info.pack(pady=3)

        # Right Panel: Plot Canvas
        right_panel = ttk.Frame(self.tab_waveform)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.fig_waveform = Figure(figsize=(8, 5), dpi=100)
        self.canvas_waveform = FigureCanvasTkAgg(self.fig_waveform, master=right_panel)
        self.canvas_waveform.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas_waveform, right_panel)
        toolbar.update()

    def _build_pipeline_tab(self):
        # Top toolbar with frame jump buttons
        ctrl_frame = ttk.Frame(self.tab_pipeline, padding=5)
        ctrl_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(ctrl_frame, text="< Previous Frame", command=self._prev_frame).pack(side=tk.LEFT, padx=5)
        ttk.Button(ctrl_frame, text="Next Frame >", command=self._next_frame).pack(side=tk.LEFT, padx=5)
        self.lbl_pipeline_frame = ttk.Label(ctrl_frame, text="Selected: Frame 0", font=("Segoe UI", 10, "bold"))
        self.lbl_pipeline_frame.pack(side=tk.LEFT, padx=10)

        self.fig_pipeline = Figure(figsize=(10, 6), dpi=100)
        self.canvas_pipeline = FigureCanvasTkAgg(self.fig_pipeline, master=self.tab_pipeline)
        self.canvas_pipeline.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_features_tab(self):
        self.fig_features = Figure(figsize=(10, 6), dpi=100)
        self.canvas_features = FigureCanvasTkAgg(self.fig_features, master=self.tab_features)
        self.canvas_features.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_compare_tab(self):
        self.fig_compare = Figure(figsize=(10, 6), dpi=100)
        self.canvas_compare = FigureCanvasTkAgg(self.fig_compare, master=self.tab_compare)
        self.canvas_compare.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_dataset_tab(self):
        top_ctrl = ttk.Frame(self.tab_dataset, padding=5)
        top_ctrl.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top_ctrl, text="Filter Class:").pack(side=tk.LEFT, padx=5)
        self.cmb_class = ttk.Combobox(top_ctrl, values=["ALL", "silence", "unknown", "swara"], state="readonly", width=12)
        self.cmb_class.set("ALL")
        self.cmb_class.pack(side=tk.LEFT, padx=5)
        self.cmb_class.bind("<<ComboboxSelected>>", self._on_filter_dataset)

        ttk.Button(top_ctrl, text="Refresh Dataset", command=self._refresh_dataset_view).pack(side=tk.LEFT, padx=10)

        self.lbl_dataset_stats = ttk.Label(top_ctrl, text="", font=("Segoe UI", 9))
        self.lbl_dataset_stats.pack(side=tk.RIGHT, padx=10)

        # Treeview Table
        columns = ("filename", "class", "duration", "sample_rate", "channels", "bits", "split", "status")
        self.tree_dataset = ttk.Treeview(self.tab_dataset, columns=columns, show="headings", height=15)
        self.tree_dataset.heading("filename", text="File Name")
        self.tree_dataset.heading("class", text="Class")
        self.tree_dataset.heading("duration", text="Duration")
        self.tree_dataset.heading("sample_rate", text="Sample Rate")
        self.tree_dataset.heading("channels", text="Channels")
        self.tree_dataset.heading("bits", text="Bit Depth")
        self.tree_dataset.heading("split", text="Split (Recording-Level)")
        self.tree_dataset.heading("status", text="Contract Status")

        self.tree_dataset.column("filename", width=260)
        self.tree_dataset.column("class", width=90)
        self.tree_dataset.column("duration", width=80)
        self.tree_dataset.column("sample_rate", width=90)
        self.tree_dataset.column("channels", width=70)
        self.tree_dataset.column("bits", width=70)
        self.tree_dataset.column("split", width=140)
        self.tree_dataset.column("status", width=140)

        self.tree_dataset.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.tree_dataset.bind("<Double-1>", self._on_dataset_item_double_click)

    def load_audio_file(self, filepath: str):
        try:
            self.current_audio = self.pipeline.process_full_audio(filepath)
            self._update_all_views()
        except Exception as e:
            messagebox.showerror("Audio Load Error", f"Failed to process {filepath}:\n{e}")

    def _update_all_views(self):
        if not self.current_audio:
            return

        meta = self.current_audio.metadata
        stats = self.current_audio.statistics

        # Title & validity
        self.lbl_title.config(text=f"Loaded: {meta.filename}")
        if meta.is_valid_swara_contract:
            self.lbl_validity.config(text="[VALID SWARA CONTRACT]", foreground="green")
        else:
            self.lbl_validity.config(text="[INVALID CONTRACT]", foreground="red")

        # Metadata Text Info
        self.txt_metadata.delete("1.0", tk.END)
        self.txt_metadata.insert(tk.END, f"File: {meta.filename}\n")
        self.txt_metadata.insert(tk.END, f"Size: {meta.file_size_bytes} bytes\n")
        self.txt_metadata.insert(tk.END, f"Duration: {meta.duration_sec:.3f} s ({meta.num_samples} spl)\n")
        self.txt_metadata.insert(tk.END, f"Sample Rate: {meta.sample_rate} Hz\n")
        self.txt_metadata.insert(tk.END, f"Channels: {meta.channels} ({'Mono' if meta.channels == 1 else 'Stereo'})\n")
        self.txt_metadata.insert(tk.END, f"Bit Depth: {meta.bit_depth}-bit signed PCM\n")
        self.txt_metadata.insert(tk.END, f"Split: {self.current_audio.split_category.upper() if self.current_audio.split_category else 'N/A'}\n")
        self.txt_metadata.insert(tk.END, f"------------------------------\n")
        self.txt_metadata.insert(tk.END, f"Signal Statistics:\n")
        self.txt_metadata.insert(tk.END, f"  Peak: {stats.peak_amplitude:.0f} / 32767\n")
        self.txt_metadata.insert(tk.END, f"  RMS:  {stats.rms_amplitude:.1f}\n")
        self.txt_metadata.insert(tk.END, f"  Dynamic Range: {stats.dynamic_range_db:.1f} dB\n")
        self.txt_metadata.insert(tk.END, f"  Clipping: {stats.clipping_percentage:.2f}%\n")
        self.txt_metadata.insert(tk.END, f"  Near-Zero: {stats.near_zero_percentage:.1f}%\n")
        self.txt_metadata.insert(tk.END, f"  DC Offset: {stats.dc_offset:.1f}\n")
        if stats.quality_warnings or meta.validation_reasons:
            self.txt_metadata.insert(tk.END, f"\nQuality Flags:\n")
            for r in meta.validation_reasons:
                self.txt_metadata.insert(tk.END, f"  ! {r}\n")
            for w in stats.quality_warnings:
                self.txt_metadata.insert(tk.END, f"  ! {w}\n")

        self._render_current_frame()

    def _render_current_frame(self):
        if not self.current_audio:
            return

        f_idx = self.selected_frame_idx
        sel_f = self.current_audio.frames[f_idx]

        self.lbl_frame_info.config(text=f"Frame {f_idx} ({sel_f.start_time_ms:.0f}-{sel_f.end_time_ms:.0f} ms)")
        self.lbl_pipeline_frame.config(text=f"Selected: Frame {f_idx} (Time: {sel_f.start_time_ms:.0f}-{sel_f.end_time_ms:.0f} ms)")

        # Tab 1: Waveform
        VisualizerPlotEngine.plot_waveform(self.fig_waveform, self.current_audio, f_idx)
        self.canvas_waveform.draw()

        # Tab 2: DSP Stages
        VisualizerPlotEngine.plot_frame_dsp_stages(
            self.fig_pipeline, sel_f, self.pipeline.mel_filterbank, self.pipeline.bin_frequencies
        )
        self.canvas_pipeline.draw()

        # Tab 3: Spectrogram & MFCC Heatmap
        VisualizerPlotEngine.plot_spectrogram(self.fig_features, self.current_audio, f_idx)
        self.canvas_features.draw()

    def _on_slider_change(self, val):
        self.selected_frame_idx = int(float(val))
        self._render_current_frame()

    def _prev_frame(self):
        if self.selected_frame_idx > 0:
            self.selected_frame_idx -= 1
            self.frame_slider.set(self.selected_frame_idx)
            self._render_current_frame()

    def _next_frame(self):
        if self.selected_frame_idx < 48:
            self.selected_frame_idx += 1
            self.frame_slider.set(self.selected_frame_idx)
            self._render_current_frame()

    def _on_open_file(self):
        path = filedialog.askopenfilename(
            title="Select WAV File",
            filetypes=[("WAV Audio Files", "*.wav"), ("All Files", "*.*")]
        )
        if path:
            self.load_audio_file(path)

    def _on_open_dataset(self):
        dpath = filedialog.askdirectory(title="Select Dataset Directory")
        if dpath:
            self.dataset_model.scan_directory(dpath)
            self._populate_dataset_tree(self.dataset_model.file_records)
            self.notebook.select(self.tab_dataset)

    def _on_load_comparison(self):
        path = filedialog.askopenfilename(
            title="Select Comparison WAV File",
            filetypes=[("WAV Audio Files", "*.wav"), ("All Files", "*.*")]
        )
        if path and self.current_audio:
            try:
                self.comparison_audio = self.pipeline.process_full_audio(path)
                VisualizerPlotEngine.plot_comparison(
                    self.fig_compare,
                    self.current_audio,
                    self.comparison_audio,
                    label_a=self.current_audio.metadata.filename,
                    label_b=self.comparison_audio.metadata.filename
                )
                self.canvas_compare.draw()
                self.notebook.select(self.tab_compare)
            except Exception as e:
                messagebox.showerror("Comparison Error", f"Failed to process comparison WAV:\n{e}")

    def _refresh_dataset_view(self):
        records = self.dataset_model.scan_directory()
        self._populate_dataset_tree(records)

    def _on_filter_dataset(self, event=None):
        cls_choice = self.cmb_class.get()
        records = self.dataset_model.filter_by_class(cls_choice)
        self._populate_dataset_tree(records)

    def _populate_dataset_tree(self, records):
        self.tree_dataset.delete(*self.tree_dataset.get_children())
        for r in records:
            status_text = "VALID" if r["is_valid"] else "INVALID"
            self.tree_dataset.insert("", tk.END, values=(
                r["filename"],
                r["class"],
                f"{r['duration_sec']:.2f}s",
                f"{r['sample_rate']} Hz",
                r["channels"],
                f"{r['bit_depth']}-bit",
                r["split"],
                status_text
            ), tags=(status_text.lower(),))

        self.tree_dataset.tag_configure("valid", foreground="black")
        self.tree_dataset.tag_configure("invalid", foreground="red")
        self.lbl_dataset_stats.config(text=f"Total: {len(records)} recordings")

    def _on_dataset_item_double_click(self, event):
        item = self.tree_dataset.selection()
        if item:
            vals = self.tree_dataset.item(item[0], "values")
            filename = vals[0]
            for r in self.dataset_model.file_records:
                if r["filename"] == filename:
                    self.load_audio_file(r["filepath"])
                    self.notebook.select(self.tab_waveform)
                    break


def main():
    parser = argparse.ArgumentParser(description="Swara WAV / MFCC Visualization & Debugging Engine")
    parser.add_argument("--file", type=str, default=None, help="Path to initial WAV file to inspect")
    parser.add_argument("--dataset", type=str, default="data/raw", help="Path to dataset directory")
    args = parser.parse_args()

    app = SwaraVisualizerApp(initial_file=args.file, dataset_dir=args.dataset)
    app.mainloop()


if __name__ == "__main__":
    main()
