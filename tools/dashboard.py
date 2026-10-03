"""
Swara CLI Model & Pipeline Dashboard (Desktop Development & Training Tool).

Interactive terminal UI featuring:
- ASCII Banner & System Status Box
- Dataset Audit & Recording-Level Health Check
- Architecture Inspection & Memory Estimator (16, 24, 32, 48, 64 channels)
- Feature Parity & CTest Suite Runner
- Safe Training Dry-Run & Dataset Verifier
- INT8 Quantization & Calibration Safety Inspector
- Flatbuffer & TFLM Operator Compatibility Audit
- Deterministic C Deployment Array Exporter
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, List

# Ensure working directory is always repository root
REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
training_dir = REPO_ROOT / "training"
if str(training_dir) not in sys.path:
    sys.path.insert(0, str(training_dir))

# Ensure standard output can print Unicode characters safely
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Check critical dependencies with clear diagnostic if invoked by non-main Python (e.g. MinGW Python)
try:
    import numpy as np
except ImportError:
    print("\n" + "=" * 70)
    print(" [!] SWARA DASHBOARD STARTUP ERROR: 'numpy' not found.")
    print("=" * 70)
    print(f"Current Python: {sys.executable}")
    print("\nIf you are using MSYS2 / MinGW, 'python' points to the MinGW interpreter.")
    print("Run the dashboard using the Windows Python launcher instead:")
    print("    py tools/dashboard.py")
    print("  or:")
    print(r"    C:\Python313\python.exe tools/dashboard.py")
    print("=" * 70 + "\n")
    sys.exit(1)

try:
    from config import (
        DEFAULT_NUM_FILTERS,
        DEFAULT_NUM_CLASSES,
        DEFAULT_INPUT_SHAPE,
        CANDIDATE_CHANNEL_WIDTHS,
        CLASSES
    )
    from dataset import inspect_dataset, SwaraDataset
    from model import build_dscnn_model
    from validate_tflite import inspect_tflite_model, TFLM_STANDARD_OPS
except ImportError as e:
    print(f"Error loading training modules: {e}")
    sys.exit(1)

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.layout import Layout
    from rich.text import Text
    from rich.prompt import Prompt, Confirm
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
SWARA_ASCII_BANNER = """
 ███████╗██╗    ██╗ █████╗ ██████╗  █████╗ 
 ██╔════╝██║    ██║██╔══██╗██╔══██╗██╔══██╗
 ███████╗██║ █╗ ██║███████║██████╔╝███████║
 ╚════██║██║███╗██║██╔══██║██╔══██╗██╔══██║
 ███████║╚███╔███╔╝██║  ██║██║  ██║██║  ██║
 ╚══════╝ ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝ 
"""

console = Console(highlight=False) if RICH_AVAILABLE else None



def print_banner(target_data_dir: str = "data/raw"):
    raw_status = "READY" if os.path.exists(target_data_dir) and len(os.listdir(target_data_dir)) > 0 else "NO REAL DATA"
    status_color = "green" if raw_status == "READY" else "red"

    banner_text = Text(SWARA_ASCII_BANNER, style="bold cyan")
    
    info_table = Table.grid(padding=(0, 2))
    info_table.add_column(style="bold yellow")
    info_table.add_column(style="bold white")
    
    info_table.add_row("Status:", f"[{status_color}]{raw_status}[/{status_color}]")
    info_table.add_row("Target:", f"swara@c-embedded (C99 / TFLM)")
    info_table.add_row("Memory Budget:", f"<= 256 KB Total RAM")
    info_table.add_row("Audio Spec V0:", f"16kHz Mono 16-bit PCM | (49, 10, 1)")
    info_table.add_row("Dataset Split:", f"RECORDING-LEVEL (NOT speaker-indep)")

    header_table = Table.grid(expand=True)
    header_table.add_column(ratio=6)
    header_table.add_column(ratio=5)
    header_table.add_row(banner_text, Panel(info_table, title="[bold white]System Context[/bold white]", border_style="cyan", box=box.ROUNDED))

    console.print(Panel(header_table, border_style="blue", box=box.DOUBLE))


def run_dataset_audit(data_dir: str = "data/raw"):
    console.print(f"\n[bold yellow]>>> Running Swara Dataset Audit on: {os.path.abspath(data_dir)}[/bold yellow]")
    report = inspect_dataset(data_dir)

    if "error" in report:
        console.print(Panel(f"[bold red]AUDIT FAILED:[/bold red] {report['error']}\n"
                            "Real dataset recordings are not yet placed in this directory.",
                            title="Dataset Audit Error", border_style="red"))
        return

    table = Table(title=f"Dataset Audit Results ({data_dir})", box=box.ROUNDED)
    table.add_column("Class", style="cyan", justify="left")
    table.add_column("Status", style="magenta", justify="center")
    table.add_column("Total Files", justify="right")
    table.add_column("Valid Audio", justify="right")
    table.add_column("Total Duration", justify="right")

    for cls in CLASSES:
        info = report["class_breakdown"][cls]
        status = "[green]OK[/green]" if info["valid"] > 0 else "[bold red]MISSING[/bold red]"
        dur_min = info["duration_sec"] / 60.0
        table.add_row(cls, status, str(info["total"]), str(info["valid"]), f"{dur_min:.2f} min ({info['duration_sec']:.1f}s)")

    console.print(table)

    summary_panel = Text()
    summary_panel.append(f"Total WAV Files:        {report['total_wav_files']}\n")
    summary_panel.append(f"Valid WAV Files:        {report['valid_wav_files']}\n")
    summary_panel.append(f"Usable Recordings:      {report['usable_recordings']} (16kHz mono 16-bit PCM)\n")
    summary_panel.append(f"Invalid / Corrupt:      {len(report['invalid_corrupt_files'])}\n")
    summary_panel.append(f"Duplicate Copies:       {report['duplicate_files_count']}\n")
    summary_panel.append(f"Class Imbalance Ratio:  {report['class_imbalance_ratio']:.2f}x\n")

    console.print(Panel(summary_panel, title="[bold white]Dataset Integrity & Contract Compliance[/bold white]", border_style="green"))


def explore_architectures():
    console.print("\n[bold yellow]>>> DS-CNN Architecture Exploration & Memory Estimation[/bold yellow]")
    console.print("Comparing DS-CNN channel candidates against Swara's [bold green]<= 256 KB[/bold green] total RAM budget:\n")

    table = Table(title="Swara DS-CNN Channel Width Comparison", box=box.ROUNDED)
    table.add_column("Candidate", justify="left", style="cyan")
    table.add_column("Channels", justify="right", style="bold")
    table.add_column("Params", justify="right")
    table.add_column("INT8 Weights", justify="right")
    table.add_column("Est. Arena", justify="right")
    table.add_column("Est. Peak RAM", justify="right")
    table.add_column("RAM Budget Status", justify="center")

    front_end_measured_ram = 60444  # Measured static front-end C RAM: 59.03 KB

    for width in CANDIDATE_CHANNEL_WIDTHS:
        model = build_dscnn_model(num_filters=width)
        params = model.count_params()
        int8_weights = params  # 1 byte per quantized weight
        
        # Intermediate activation tensor peak estimate
        # Conv1: 25 * 10 * width, DW1: 25 * 10 * width, PW1: 25 * 10 * width
        arena_estimate = int(width * 25 * 10 * 2.3) + 2048  # working activation arena
        total_ram = front_end_measured_ram + arena_estimate + 15360 + 4096

        tag = "CURRENT BASELINE" if width == DEFAULT_NUM_FILTERS else ("PREFERRED LOW-RAM" if width == 32 else "")
        status = "[green]COMPLIANT (<=256KB)[/green]" if total_ram <= 256 * 1024 else "[red]EXCEEDS BUDGET[/red]"

        table.add_row(
            f"{width} Channels {f'({tag})' if tag else ''}",
            str(width),
            f"{params:,}",
            f"{int8_weights/1024.0:.1f} KB",
            f"{arena_estimate/1024.0:.1f} KB",
            f"{total_ram/1024.0:.1f} KB",
            status
        )

    console.print(table)
    console.print("[dim]* Note: Front-end DSP RAM (59.03 KB) is MEASURED. Tensor Arena is an ESTIMATE pending tflite::MicroInterpreter profiling.[/dim]\n")


def run_pipeline_benchmark():
    console.print("\n[bold yellow]>>> Running Native C End-to-End Audio Pipeline Benchmark...[/bold yellow]")
    bench_exe = "build/benchmark_pipeline.exe" if sys.platform == "win32" else "build/benchmark_pipeline"
    if not os.path.exists(bench_exe):
        console.print(f"[bold red]Benchmark executable not found at {bench_exe}. Compiling...[/bold red]")
        subprocess.run(["cmake", "--build", "build", "--target", "benchmark_pipeline"], check=True)

    res = subprocess.run([bench_exe], capture_output=True, text=True)
    console.print(Panel(res.stdout, title="[bold green]Native C Benchmark Output[/bold green]", border_style="cyan"))


def run_parity_test():
    console.print("\n[bold yellow]>>> Running C vs Python Numerical Feature Parity Test...[/bold yellow]")
    res = subprocess.run([sys.executable, "tests/test_feature_parity.py"], capture_output=True, text=True)
    console.print(res.stdout)


def test_training_dry_run():
    console.print("\n[bold yellow]>>> Simulating Model Training Ingestion Verification (Dry-Run)...[/bold yellow]")
    console.print("Testing against isolated test fixtures: [cyan]tests/data/test_dataset_fixture[/cyan]")

    from train import train
    try:
        train(
            data_dir="tests/data/test_dataset_fixture",
            epochs=1,
            batch_size=2,
            dry_run=True,
            num_filters=64
        )
        console.print("[bold green]>> SUCCESS: Training harness verified. Correctly validated shapes, splits, and classes.[/bold green]")
    except Exception as e:
        console.print(f"[bold red]>> FAILED:[/bold red] {e}")


def launch_wav_visualizer():
    console.print("\n[bold yellow]>>> Launching Desktop WAV & MFCC Visualization Engine...[/bold yellow]")
    subprocess.Popen([sys.executable, "-m", "tools.wav_visualizer.app.main"])
    console.print("[green]Visualizer GUI started in background window.[/green]")


def main_dashboard():
    if not RICH_AVAILABLE:
        print("Please install 'rich' to run the CLI dashboard: pip install rich")
        sys.exit(1)

    while True:
        os.system("cls" if os.name == "nt" else "clear")
        print_banner()

        menu_table = Table(title="[bold white]Swara Model Creation & Architecture Dashboard[/bold white]", box=box.ROUNDED)
        menu_table.add_column("Option", style="bold cyan", justify="center", width=8)
        menu_table.add_column("Action", style="bold white")
        menu_table.add_column("Description", style="dim")

        menu_table.add_row("1", "Audit Dataset (data/raw)", "Inspect class counts, durations, and audio contract compliance")
        menu_table.add_row("2", "Audit Test Fixtures", "Run inspection on isolated unit test fixtures (tests/data/test_dataset_fixture)")
        menu_table.add_row("3", "Explore DS-CNN Channels", "Compare parameters, weights, and memory estimates for 16, 24, 32, 48, 64 channels")
        menu_table.add_row("4", "Run C Pipeline Benchmark", "Run native C performance profiler (VAD, ring buffer, FFT, MFCC)")
        menu_table.add_row("5", "Run C vs Python Parity Test", "Verify numerical equivalence between C runtime and offline Python DSP")
        menu_table.add_row("6", "Run All Infrastructure Tests", "Execute complete 10-test verification suite (tests/test_training_infrastructure.py)")
        menu_table.add_row("7", "Simulate Training Dry-Run", "Verify training setup and data loading without model.fit")
        menu_table.add_row("8", "Launch WAV / MFCC Visualizer", "Open desktop GUI to inspect waveforms, FFT spectra, and MFCC heatmaps")
        menu_table.add_row("Q", "Quit", "Exit CLI dashboard")

        console.print(menu_table)

        try:
            choice = Prompt.ask("\n[bold yellow]Select an option[/bold yellow]", choices=["1", "2", "3", "4", "5", "6", "7", "8", "q", "Q"], default="3")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[bold cyan]Exiting Swara Dashboard.[/bold cyan]\n")
            break

        if choice in ["q", "Q"]:
            console.print("\n[bold cyan]Exiting Swara Dashboard. Keep embedded constraints front-of-mind![/bold cyan]\n")
            break
        elif choice == "1":
            run_dataset_audit("data/raw")
        elif choice == "2":
            run_dataset_audit("tests/data/test_dataset_fixture")
        elif choice == "3":
            explore_architectures()
        elif choice == "4":
            run_pipeline_benchmark()
        elif choice == "5":
            run_parity_test()
        elif choice == "6":
            subprocess.run([sys.executable, "tests/test_training_infrastructure.py"])
        elif choice == "7":
            test_training_dry_run()
        elif choice == "8":
            launch_wav_visualizer()

        try:
            Prompt.ask("\n[dim]Press Enter to return to main menu...[/dim]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[bold cyan]Exiting Swara Dashboard.[/bold cyan]\n")
            break


if __name__ == "__main__":
    main_dashboard()
