"""Simple desktop launcher for primary RoboSim utilities."""

from __future__ import annotations

import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk


class UtilityLauncher:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("RoboSim Utility Launcher")
        self.root.geometry("460x300")
        self.root.resizable(False, False)

        self.project_root = Path(__file__).resolve().parent
        self.status_var = tk.StringVar(value="Ready")

        self._build_ui()

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=16)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            frame,
            text="Launch a main utility",
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w", pady=(0, 8))

        ttk.Label(
            frame,
            text="This opens each tool in its own Python process.",
        ).pack(anchor="w", pady=(0, 14))

        ttk.Button(
            frame,
            text="Open Simulator",
            command=lambda: self._launch_script("simulator.py"),
        ).pack(fill="x", pady=4)

        ttk.Button(
            frame,
            text="Open MLP Trainer",
            command=lambda: self._launch_script("train_mlp.py"),
        ).pack(fill="x", pady=4)

        ttk.Button(
            frame,
            text="Open Benchmark Utility",
            command=lambda: self._launch_script("benchmark_gui.py"),
        ).pack(fill="x", pady=4)

        ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=14)
        ttk.Label(frame, textvariable=self.status_var).pack(anchor="w")

    def _launch_script(self, script_name: str) -> None:
        script_path = self.project_root / script_name
        if not script_path.exists():
            messagebox.showerror("Launch failed", f"Script not found: {script_name}")
            self.status_var.set(f"Missing script: {script_name}")
            return

        try:
            subprocess.Popen(
                [sys.executable, str(script_path)],
                cwd=str(self.project_root),
            )
            self.status_var.set(f"Launched {script_name}")
        except Exception as exc:
            messagebox.showerror("Launch failed", str(exc))
            self.status_var.set(f"Failed to launch {script_name}")

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    UtilityLauncher().run()
