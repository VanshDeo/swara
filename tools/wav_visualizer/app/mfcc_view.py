"""
MFCC, Spectrogram, Filterbank and Pipeline Plotting Routines (Desktop Tooling).
Embeds interactive matplotlib figures into Tkinter panels.
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk

from .models import ProcessedAudio, FrameAnalysis


class VisualizerPlotEngine:
    """Manages creation and updates of acoustic and feature visualization figures."""

    @staticmethod
    def plot_waveform(fig: Figure, audio: ProcessedAudio, selected_frame_idx: int = 0):
        fig.clear()
        ax = fig.add_subplot(111)

        t = np.linspace(0.0, 1.0, len(audio.normalized_audio), endpoint=False)
        ax.plot(t * 1000.0, audio.normalized_audio, color="#1f77b4", lw=1.0, label="PCM Audio")
        ax.axhline(0, color="gray", linestyle="--", alpha=0.6)

        # Highlight clipping if any (|sample| >= 32760 / 32768)
        clip_mask = np.abs(audio.normalized_audio) >= (32760.0 / 32768.0)
        if np.any(clip_mask):
            ax.plot(t[clip_mask] * 1000.0, audio.normalized_audio[clip_mask], "ro", markersize=3, label="Clipping")

        # Overlay all 49 frame boundaries
        for f in audio.frames:
            ax.axvline(f.start_time_ms, color="#aec7e8", alpha=0.35, lw=0.8)

        # Highlight currently selected frame
        if 0 <= selected_frame_idx < len(audio.frames):
            sel_f = audio.frames[selected_frame_idx]
            ax.axvspan(sel_f.start_time_ms, sel_f.end_time_ms, color="orange", alpha=0.35,
                       label=f"Frame {selected_frame_idx} ({sel_f.start_time_ms:.0f}-{sel_f.end_time_ms:.0f}ms)")

        ax.set_title(f"1.0s Analysis Window (16 kHz Mono Signed 16-bit PCM) — Frame {selected_frame_idx}/48", fontsize=10, fontweight="bold")
        ax.set_xlabel("Time (ms)", fontsize=9)
        ax.set_ylabel("Normalized Amplitude", fontsize=9)
        ax.set_xlim(0, 1000)
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()

    @staticmethod
    def plot_spectrogram(fig: Figure, audio: ProcessedAudio, selected_frame_idx: int = 0):
        fig.clear()
        ax = fig.add_subplot(111)

        # spectrogram_matrix is (49, 257) power spectrum
        power_spec_db = 10.0 * np.log10(np.maximum(audio.spectrogram_matrix, 1e-8)).T
        extent = [0, 1000, 0, 8000]

        im = ax.imshow(power_spec_db, origin="lower", aspect="auto", extent=extent, cmap="inferno")
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label("Power (dB)", fontsize=8)

        if 0 <= selected_frame_idx < len(audio.frames):
            sel_f = audio.frames[selected_frame_idx]
            ax.axvline(sel_f.start_time_ms + 15.0, color="cyan", linestyle="--", lw=1.5,
                       label=f"Frame {selected_frame_idx}")

        ax.set_title("Narrowband Spectrogram (0 - 8,000 Hz, 512-pt FFT, 31.25 Hz/bin)", fontsize=10, fontweight="bold")
        ax.set_xlabel("Time (ms)", fontsize=9)
        ax.set_ylabel("Frequency (Hz)", fontsize=9)
        ax.set_ylim(0, 8000)
        fig.tight_layout()

    @staticmethod
    def plot_mfcc_heatmap(fig: Figure, audio: ProcessedAudio, selected_frame_idx: int = 0):
        fig.clear()
        ax = fig.add_subplot(111)

        # mfcc_matrix is (49, 10)
        im = ax.imshow(audio.mfcc_matrix.T, origin="lower", aspect="auto", cmap="viridis")
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label("MFCC Amplitude", fontsize=8)

        ax.axvline(selected_frame_idx, color="red", linestyle="--", lw=1.5, label=f"Selected Frame {selected_frame_idx}")

        ax.set_title("Swara MFCC Feature Heatmap [49 Frames × 10 Coefficients]", fontsize=10, fontweight="bold")
        ax.set_xlabel("Frame Index (0 to 48)", fontsize=9)
        ax.set_ylabel("MFCC Coefficient (0 to 9)", fontsize=9)
        ax.set_yticks(range(10))
        ax.set_yticklabels([f"C{i}" for i in range(10)])
        fig.tight_layout()

    @staticmethod
    def plot_frame_dsp_stages(fig: Figure, frame: FrameAnalysis, mel_filterbank: np.ndarray, bin_freqs: np.ndarray):
        fig.clear()

        # 4 Subplots:
        # 1. Raw vs Pre-emphasized Frame (480 samples)
        # 2. FFT Power Spectrum vs Mel Filterbank (0 - 8000 Hz)
        # 3. 20 Mel Energies & 20 Log-Mel Values
        # 4. 10 MFCC Coefficients Bar Chart

        # Panel 1: Frame Waveform & Pre-emphasis
        ax1 = fig.add_subplot(221)
        ax1.plot(frame.raw_samples, color="#aec7e8", label="Raw PCM (480 spl)")
        ax1.plot(frame.preemphasized_samples, color="#1f77b4", label="Pre-emph (0.97)")
        ax1.plot(frame.windowed_samples * 32768.0, color="orange", alpha=0.7, label="Hamming Win")
        ax1.set_title(f"Stage 1: Pre-emphasis & Window (Frame {frame.frame_index})", fontsize=9, fontweight="bold")
        ax1.set_xlabel("Sample index (0-479)", fontsize=8)
        ax1.set_ylabel("Amplitude", fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper right", fontsize=7)

        # Panel 2: Power Spectrum and 20 Mel Filters
        ax2 = fig.add_subplot(222)
        # Plot 20 Mel filter triangles
        for m in range(mel_filterbank.shape[0]):
            ax2.plot(bin_freqs, mel_filterbank[m], color="green", alpha=0.5, lw=1.0)
        # Overlay normalized power spectrum
        norm_power = frame.power_spectrum / max(1e-6, np.max(frame.power_spectrum))
        ax2.plot(bin_freqs, norm_power, color="red", lw=1.2, label="Normalized Power Spec")
        ax2.set_title("Stage 2: 512-pt FFT & 20 Mel Filterbank (20-8000 Hz)", fontsize=9, fontweight="bold")
        ax2.set_xlabel("Frequency (Hz)", fontsize=8)
        ax2.set_ylabel("Filter Gain / Power", fontsize=8)
        ax2.set_xlim(20, 8000)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="upper right", fontsize=7)

        # Panel 3: 20 Mel Energies & Log-Mel Values
        ax3 = fig.add_subplot(223)
        x_mels = np.arange(len(frame.mel_energies))
        ax3.bar(x_mels - 0.2, frame.log_mel_energies, width=0.4, color="#2ca02c", label="Log-Mel (ln(E + 1e-6))")
        ax3.set_title("Stage 3: 20 Log-Mel Filterbank Energies", fontsize=9, fontweight="bold")
        ax3.set_xlabel("Mel Filter Index (0 to 19)", fontsize=8)
        ax3.set_ylabel("Log Energy", fontsize=8)
        ax3.set_xticks(range(0, 20, 2))
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc="lower right", fontsize=7)

        # Panel 4: 10 MFCC Coefficients Bar Chart
        ax4 = fig.add_subplot(224)
        x_mfcc = np.arange(len(frame.mfcc_coefficients))
        bars = ax4.bar(x_mfcc, frame.mfcc_coefficients, color="#9467bd", width=0.5)
        # Add value labels on top of bars
        for bar, val in zip(bars, frame.mfcc_coefficients):
            ax4.text(bar.get_x() + bar.get_width()/2.0, val, f"{val:.1f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=6)
        ax4.axhline(0, color="gray", lw=0.8)
        ax4.set_title("Stage 4: DCT-II -> 10 MFCC Coefficients", fontsize=9, fontweight="bold")
        ax4.set_xlabel("Coefficient Index (C0 to C9)", fontsize=8)
        ax4.set_ylabel("Value", fontsize=8)
        ax4.set_xticks(range(10))
        ax4.set_xticklabels([f"C{i}" for i in range(10)])
        ax4.grid(True, alpha=0.3)

        fig.tight_layout()

    @staticmethod
    def plot_comparison(fig: Figure, audio_a: ProcessedAudio, audio_b: ProcessedAudio, label_a: str = "Audio A", label_b: str = "Audio B"):
        fig.clear()

        # Top row: Audio A (Waveform, Spectrogram, MFCC)
        ax1 = fig.add_subplot(231)
        t_a = np.linspace(0, 1000, len(audio_a.normalized_audio))
        ax1.plot(t_a, audio_a.normalized_audio, color="#1f77b4")
        ax1.set_title(f"{label_a}: Waveform", fontsize=9, fontweight="bold")
        ax1.set_xlabel("Time (ms)", fontsize=8)
        ax1.set_ylim(-1.05, 1.05)
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(232)
        p_a = 10.0 * np.log10(np.maximum(audio_a.spectrogram_matrix, 1e-8)).T
        ax2.imshow(p_a, origin="lower", aspect="auto", extent=[0, 1000, 0, 8000], cmap="inferno")
        ax2.set_title(f"{label_a}: Spectrogram", fontsize=9, fontweight="bold")
        ax2.set_xlabel("Time (ms)", fontsize=8)

        ax3 = fig.add_subplot(233)
        ax3.imshow(audio_a.mfcc_matrix.T, origin="lower", aspect="auto", cmap="viridis")
        ax3.set_title(f"{label_a}: MFCC Heatmap (49x10)", fontsize=9, fontweight="bold")
        ax3.set_xlabel("Frame", fontsize=8)

        # Bottom row: Audio B (Waveform, Spectrogram, MFCC)
        ax4 = fig.add_subplot(234)
        t_b = np.linspace(0, 1000, len(audio_b.normalized_audio))
        ax4.plot(t_b, audio_b.normalized_audio, color="#ff7f0e")
        ax4.set_title(f"{label_b}: Waveform", fontsize=9, fontweight="bold")
        ax4.set_xlabel("Time (ms)", fontsize=8)
        ax4.set_ylim(-1.05, 1.05)
        ax4.grid(True, alpha=0.3)

        ax5 = fig.add_subplot(235)
        p_b = 10.0 * np.log10(np.maximum(audio_b.spectrogram_matrix, 1e-8)).T
        ax5.imshow(p_b, origin="lower", aspect="auto", extent=[0, 1000, 0, 8000], cmap="inferno")
        ax5.set_title(f"{label_b}: Spectrogram", fontsize=9, fontweight="bold")
        ax5.set_xlabel("Time (ms)", fontsize=8)

        ax6 = fig.add_subplot(236)
        ax6.imshow(audio_b.mfcc_matrix.T, origin="lower", aspect="auto", cmap="viridis")
        ax6.set_title(f"{label_b}: MFCC Heatmap (49x10)", fontsize=9, fontweight="bold")
        ax6.set_xlabel("Frame", fontsize=8)

        fig.tight_layout()
