from __future__ import annotations

import base64
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

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


def build_playalong_component(audio_bytes: bytes, audio_mime: str, segment_df: pd.DataFrame) -> str:
    encoded_audio = base64.b64encode(audio_bytes).decode("ascii")
    segment_records = [
        {
            "start": float(row.start_time_seconds),
            "end": float(row.end_time_seconds),
            "duration": float(row.duration_seconds),
            "label": str(row.predicted_label),
        }
        for row in segment_df.itertuples(index=False)
    ]
    segment_json = json.dumps(segment_records)
    audio_src = f"data:{audio_mime};base64,{encoded_audio}"

    return f"""
<div id="playalong-app" style="font-family: Georgia, 'Times New Roman', serif; color: #1d293d;">
  <style>
    #playalong-app {{
      background: linear-gradient(135deg, #f6efe2 0%, #fdf8ef 100%);
      border: 1px solid #d8c9a8;
      border-radius: 18px;
      padding: 18px;
    }}
    #playalong-app .hero {{
      display: grid;
      grid-template-columns: 1.4fr 1fr;
      gap: 16px;
      align-items: stretch;
      margin-top: 14px;
    }}
    #playalong-app .card {{
      background: rgba(255,255,255,0.75);
      border: 1px solid #e6dbc3;
      border-radius: 16px;
      padding: 16px;
      box-shadow: 0 6px 24px rgba(69, 46, 7, 0.08);
    }}
    #playalong-app .eyebrow {{
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-size: 12px;
      color: #7b6640;
      margin-bottom: 8px;
    }}
    #playalong-app .current-chord {{
      font-size: 52px;
      line-height: 1;
      font-weight: 700;
      color: #7a2e1f;
      margin-bottom: 10px;
    }}
    #playalong-app .next-chord {{
      font-size: 28px;
      font-weight: 600;
      color: #214e62;
      margin-bottom: 12px;
    }}
    #playalong-app .meta {{
      font-size: 15px;
      color: #4c5a67;
    }}
    #playalong-app .lane {{
      display: flex;
      gap: 10px;
      overflow-x: auto;
      padding-bottom: 8px;
      margin-top: 14px;
    }}
    #playalong-app .segment {{
      min-width: 140px;
      padding: 12px 14px;
      border-radius: 14px;
      border: 1px solid #d8c9a8;
      background: #fffdfa;
      transition: transform 0.12s ease, background 0.12s ease, border-color 0.12s ease;
    }}
    #playalong-app .segment.active {{
      background: #7a2e1f;
      color: #fff6ea;
      border-color: #7a2e1f;
      transform: translateY(-2px);
    }}
    #playalong-app .segment.upcoming {{
      background: #dff1f5;
      border-color: #78aeb9;
    }}
    #playalong-app .segment-label {{
      font-size: 24px;
      font-weight: 700;
      margin-bottom: 4px;
    }}
    #playalong-app .segment-time {{
      font-size: 13px;
      opacity: 0.85;
    }}
    #playalong-app audio {{
      width: 100%;
      margin-top: 6px;
    }}
    @media (max-width: 700px) {{
      #playalong-app .hero {{
        grid-template-columns: 1fr;
      }}
      #playalong-app .current-chord {{
        font-size: 40px;
      }}
    }}
  </style>

  <div class="eyebrow">Play-Along Prototype</div>
  <audio id="audio" controls preload="metadata" src="{audio_src}"></audio>

  <div class="hero">
    <div class="card">
      <div class="eyebrow">Current Chord</div>
      <div id="currentChord" class="current-chord">Press play</div>
      <div id="nextChord" class="next-chord">Next: -</div>
      <div id="timeMeta" class="meta">Current time: 0:00</div>
    </div>
    <div class="card">
      <div class="eyebrow">What This Does</div>
      <div class="meta">
        This prototype tracks the browser audio time and highlights the current chord segment.
        Use it as a play-along guide rather than a perfectly precise score follower.
      </div>
    </div>
  </div>

  <div id="lane" class="lane"></div>

  <script>
    const segments = {segment_json};
    const audio = document.getElementById("audio");
    const currentChord = document.getElementById("currentChord");
    const nextChord = document.getElementById("nextChord");
    const timeMeta = document.getElementById("timeMeta");
    const lane = document.getElementById("lane");

    function formatSeconds(seconds) {{
      const total = Math.max(0, Math.floor(seconds));
      const minutes = Math.floor(total / 60);
      const rem = total % 60;
      return `${{minutes}}:${{String(rem).padStart(2, "0")}}`;
    }}

    segments.forEach((segment, index) => {{
      const el = document.createElement("div");
      el.className = "segment";
      el.dataset.index = String(index);
      el.innerHTML = `
        <div class="segment-label">${{segment.label}}</div>
        <div class="segment-time">${{formatSeconds(segment.start)}} - ${{formatSeconds(segment.end)}}</div>
      `;
      lane.appendChild(el);
    }});

    function updateView() {{
      const t = audio.currentTime || 0;
      let activeIndex = -1;

      for (let i = 0; i < segments.length; i += 1) {{
        const seg = segments[i];
        if (t >= seg.start && t < seg.end) {{
          activeIndex = i;
          break;
        }}
      }}

      if (activeIndex === -1 && segments.length > 0 && t >= segments[segments.length - 1].end) {{
        activeIndex = segments.length - 1;
      }}

      const active = activeIndex >= 0 ? segments[activeIndex] : null;
      const upcoming = activeIndex >= 0 && activeIndex + 1 < segments.length ? segments[activeIndex + 1] : null;

      currentChord.textContent = active ? active.label : "Press play";
      nextChord.textContent = upcoming ? `Next: ${{upcoming.label}}` : "Next: -";
      if (active) {{
        const remaining = Math.max(0, active.end - t);
        timeMeta.textContent = `Current time: ${{formatSeconds(t)}} | change in ~${{remaining.toFixed(1)}}s`;
      }} else {{
        timeMeta.textContent = `Current time: ${{formatSeconds(t)}}`;
      }}

      [...lane.children].forEach((child, index) => {{
        child.classList.toggle("active", index === activeIndex);
        child.classList.toggle("upcoming", index === activeIndex + 1);
      }});

      if (activeIndex >= 0) {{
        const activeEl = lane.children[activeIndex];
        if (activeEl) {{
          activeEl.scrollIntoView({{behavior: "smooth", block: "nearest", inline: "center"}});
        }}
      }}
    }}

    audio.addEventListener("timeupdate", updateView);
    audio.addEventListener("seeked", updateView);
    audio.addEventListener("play", updateView);
    audio.addEventListener("pause", updateView);
    updateView();
  </script>
</div>
"""


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

    if uploaded_file is not None:
        st.subheader("Play-Along Prototype")
        components.html(
            build_playalong_component(
                audio_bytes=uploaded_file.getvalue(),
                audio_mime=uploaded_file.type or "audio/mpeg",
                segment_df=result["segment_df"],
            ),
            height=520,
            scrolling=False,
        )

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
