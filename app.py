from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from infer import (
    DEFAULT_CHECKPOINT,
    DEFAULT_MIN_SEGMENT_SECONDS,
    DEFAULT_SMOOTHING_WINDOW,
    run_inference,
)


class ChordInferenceApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Chord Recognition")
        self.root.geometry("980x680")

        self.audio_path_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Choose an audio file to begin.")
        self.output_dir_var = tk.StringVar(value="")
        self.smoothing_var = tk.IntVar(value=DEFAULT_SMOOTHING_WINDOW)
        self.min_segment_var = tk.DoubleVar(value=DEFAULT_MIN_SEGMENT_SECONDS)

        self._build_ui()

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        top = ttk.Frame(self.root, padding=16)
        top.grid(row=0, column=0, sticky="nsew")
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="Local Chord Inference", font=("TkDefaultFont", 16, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w"
        )
        ttk.Label(
            top,
            text="Upload a song, run the pretrained LSTM locally, and inspect the merged chord timeline.",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 12))

        ttk.Label(top, text="Audio File").grid(row=2, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.audio_path_var).grid(row=2, column=1, sticky="ew", padx=(8, 8))
        ttk.Button(top, text="Browse", command=self._browse_audio).grid(row=2, column=2, sticky="ew")

        ttk.Label(top, text="Smoothing Window").grid(row=3, column=0, sticky="w", pady=(10, 0))
        ttk.Spinbox(top, from_=1, to=51, increment=2, textvariable=self.smoothing_var, width=10).grid(
            row=3, column=1, sticky="w", padx=(8, 8), pady=(10, 0)
        )

        ttk.Label(top, text="Min Segment Seconds").grid(row=4, column=0, sticky="w", pady=(10, 0))
        ttk.Spinbox(top, from_=0.0, to=10.0, increment=0.05, textvariable=self.min_segment_var, width=10).grid(
            row=4, column=1, sticky="w", padx=(8, 8), pady=(10, 0)
        )

        self.run_button = ttk.Button(top, text="Run Inference", command=self._run_inference)
        self.run_button.grid(row=5, column=0, pady=(14, 0), sticky="w")

        ttk.Label(top, textvariable=self.status_var, foreground="#1f3a5f").grid(
            row=5, column=1, columnspan=2, sticky="w", padx=(12, 0), pady=(14, 0)
        )

        middle = ttk.Frame(self.root, padding=(16, 0, 16, 16))
        middle.grid(row=1, column=0, sticky="nsew")
        middle.columnconfigure(0, weight=1)
        middle.rowconfigure(1, weight=1)

        ttk.Label(middle, text="Predicted Chord Segments", font=("TkDefaultFont", 12, "bold")).grid(
            row=0, column=0, sticky="w"
        )

        columns = ("start", "end", "duration", "label")
        self.tree = ttk.Treeview(middle, columns=columns, show="headings", height=18)
        self.tree.heading("start", text="Start (s)")
        self.tree.heading("end", text="End (s)")
        self.tree.heading("duration", text="Duration (s)")
        self.tree.heading("label", text="Chord")
        self.tree.column("start", width=120, anchor="center")
        self.tree.column("end", width=120, anchor="center")
        self.tree.column("duration", width=120, anchor="center")
        self.tree.column("label", width=180, anchor="center")
        self.tree.grid(row=1, column=0, sticky="nsew", pady=(8, 0))

        scrollbar = ttk.Scrollbar(middle, orient="vertical", command=self.tree.yview)
        scrollbar.grid(row=1, column=1, sticky="ns", pady=(8, 0))
        self.tree.configure(yscrollcommand=scrollbar.set)

        bottom = ttk.Frame(self.root, padding=(16, 0, 16, 16))
        bottom.grid(row=2, column=0, sticky="ew")
        bottom.columnconfigure(1, weight=1)

        ttk.Label(bottom, text="Saved Output Folder").grid(row=0, column=0, sticky="w")
        ttk.Entry(bottom, textvariable=self.output_dir_var, state="readonly").grid(
            row=0, column=1, sticky="ew", padx=(8, 0)
        )

    def _browse_audio(self) -> None:
        filename = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[
                ("Audio files", "*.mp3 *.wav *.flac *.ogg *.m4a"),
                ("All files", "*.*"),
            ],
        )
        if filename:
            self.audio_path_var.set(filename)

    def _set_running(self, running: bool) -> None:
        state = "disabled" if running else "normal"
        self.run_button.configure(state=state)

    def _run_inference(self) -> None:
        audio_path = self.audio_path_var.get().strip()
        if not audio_path:
            messagebox.showerror("Missing file", "Choose an audio file first.")
            return

        self._set_running(True)
        self.status_var.set("Running inference...")
        thread = threading.Thread(target=self._run_inference_worker, args=(audio_path,), daemon=True)
        thread.start()

    def _run_inference_worker(self, audio_path: str) -> None:
        try:
            result = run_inference(
                audio_path=Path(audio_path),
                checkpoint_path=DEFAULT_CHECKPOINT,
                device_name="auto",
                smoothing_window=int(self.smoothing_var.get()),
                min_segment_seconds=float(self.min_segment_var.get()),
            )
        except Exception as exc:  # noqa: BLE001
            self.root.after(0, lambda: self._handle_error(str(exc)))
            return

        self.root.after(0, lambda: self._handle_success(result))

    def _handle_success(self, result: dict[str, object]) -> None:
        self._set_running(False)
        self.status_var.set(f"Done on {result['device']}.")
        self.output_dir_var.set(str(result["output_dir"]))
        self._populate_segments(result["segment_df"])

    def _handle_error(self, error_message: str) -> None:
        self._set_running(False)
        self.status_var.set("Inference failed.")
        messagebox.showerror("Inference failed", error_message)

    def _populate_segments(self, segment_df: object) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)

        for row in segment_df.itertuples(index=False):
            self.tree.insert(
                "",
                "end",
                values=(
                    f"{float(row.start_time_seconds):.2f}",
                    f"{float(row.end_time_seconds):.2f}",
                    f"{float(row.duration_seconds):.2f}",
                    str(row.predicted_label),
                ),
            )


def main() -> int:
    root = tk.Tk()
    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")
    app = ChordInferenceApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
