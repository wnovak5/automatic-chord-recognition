from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from infer import (
    DEFAULT_CHECKPOINT,
    DEFAULT_MIN_SEGMENT_SECONDS,
    DEFAULT_SMOOTHING_WINDOW,
    run_inference,
)


APP_TMP_DIR = Path("tmp/streamlit")
UPLOAD_DIR = APP_TMP_DIR / "uploads"
OUTPUT_DIR = APP_TMP_DIR / "outputs"


def ensure_app_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def save_uploaded_audio(uploaded_file: Any) -> Path:
    ensure_app_dirs()
    original_name = Path(uploaded_file.name)
    safe_name = original_name.name.replace(" ", "_")
    target = UPLOAD_DIR / f"{original_name.stem}-{timestamp_slug()}{original_name.suffix.lower()}"
    target.write_bytes(uploaded_file.getbuffer())
    return target


def build_output_dir(audio_path: Path) -> Path:
    return OUTPUT_DIR / audio_path.stem


def format_seconds(seconds: float) -> str:
    total_seconds = int(seconds)
    minutes, remaining_seconds = divmod(total_seconds, 60)
    return f"{minutes}:{remaining_seconds:02d}"


def build_timeline_text(segment_df: pd.DataFrame) -> str:
    if segment_df.empty:
        return "No chord segments produced."

    lines: list[str] = []
    for row in segment_df.itertuples(index=False):
        start = format_seconds(float(row.start_time_seconds))
        end = format_seconds(float(row.end_time_seconds))
        lines.append(f"{start} - {end}: {row.predicted_label}")
    return "\n".join(lines)


def render_segment_table(segment_df: pd.DataFrame) -> None:
    if segment_df.empty:
        st.warning("No chord segments were generated.")
        return

    display_df = segment_df.copy()
    for column in ("start_time_seconds", "end_time_seconds", "duration_seconds"):
        display_df[column] = display_df[column].map(lambda value: round(float(value), 2))
    display_df = display_df.rename(
        columns={
            "start_time_seconds": "Start (s)",
            "end_time_seconds": "End (s)",
            "duration_seconds": "Duration (s)",
            "predicted_label": "Chord",
        }
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def run_app() -> None:
    st.set_page_config(page_title="Chord Recognition", page_icon="🎵", layout="wide")

    st.title("Chord Recognition")
    st.caption("Upload a song, run the pretrained LSTM locally, and inspect the chord timeline.")

    with st.sidebar:
        st.subheader("Inference Settings")
        smoothing_window = st.slider(
            "Smoothing Window",
            min_value=1,
            max_value=31,
            step=2,
            value=DEFAULT_SMOOTHING_WINDOW,
            help="Odd-number majority-vote window over neighboring frames.",
        )
        min_segment_seconds = st.slider(
            "Minimum Segment Length (seconds)",
            min_value=0.0,
            max_value=2.0,
            step=0.05,
            value=float(DEFAULT_MIN_SEGMENT_SECONDS),
            help="Very short segments are merged into adjacent chords.",
        )
        st.markdown(
            "This app uses the committed checkpoint at "
            f"`{DEFAULT_CHECKPOINT}` and runs locally on your machine."
        )

    uploaded_file = st.file_uploader(
        "Audio File",
        type=["mp3", "wav", "flac", "ogg", "m4a"],
        help="Supported formats depend on the local librosa/audio backend.",
    )

    if uploaded_file is not None:
        st.audio(uploaded_file, format=uploaded_file.type or "audio/mpeg")

    if st.button("Run Inference", type="primary", use_container_width=True, disabled=uploaded_file is None):
        if uploaded_file is None:
            st.error("Choose an audio file first.")
            return

        audio_path = save_uploaded_audio(uploaded_file)
        with st.spinner("Running inference..."):
            result = run_inference(
                audio_path=audio_path,
                checkpoint_path=DEFAULT_CHECKPOINT,
                output_dir=build_output_dir(audio_path),
                device_name="auto",
                smoothing_window=smoothing_window,
                min_segment_seconds=min_segment_seconds,
            )
        st.session_state["latest_result"] = result

    result = st.session_state.get("latest_result")
    if result is None:
        st.info("Upload a file and run inference to see chord predictions.")
        return

    segment_df = result["segment_df"]
    frame_df = result["frame_df"]
    raw_frame_df = result["raw_frame_df"]

    summary_columns = st.columns(4)
    summary_columns[0].metric("Device", str(result["device"]).upper())
    summary_columns[1].metric("Segments", int(len(segment_df)))
    summary_columns[2].metric("Smoothed Frames", int(len(frame_df)))
    summary_columns[3].metric("Output Folder", str(result["output_dir"]))

    st.subheader("Chord Timeline")
    st.code(build_timeline_text(segment_df), language="text")

    st.subheader("Merged Chord Segments")
    render_segment_table(segment_df)

    download_columns = st.columns(3)
    download_columns[0].download_button(
        "Download Chord Segments CSV",
        data=segment_df.to_csv(index=False).encode("utf-8"),
        file_name="chord_segments.csv",
        mime="text/csv",
        use_container_width=True,
    )
    download_columns[1].download_button(
        "Download Smoothed Frames CSV",
        data=frame_df.to_csv(index=False).encode("utf-8"),
        file_name="frame_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )
    download_columns[2].download_button(
        "Download Raw Frames CSV",
        data=raw_frame_df.to_csv(index=False).encode("utf-8"),
        file_name="raw_frame_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )

    with st.expander("Frame-Level Predictions"):
        display_frame_df = frame_df.copy()
        for column in ("start_time_seconds", "end_time_seconds"):
            display_frame_df[column] = display_frame_df[column].map(lambda value: round(float(value), 2))
        st.dataframe(display_frame_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    run_app()
