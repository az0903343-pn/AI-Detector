from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI Employee Monitoring", layout="wide")

try:
    from monitor.config import Settings
    from monitor.pipeline import MonitorPipeline
    from monitor.zones import grid_zones, zones_to_json
except ModuleNotFoundError as exc:  # deployment guard: folder not pushed
    st.error(f"Python import failed: {exc}")
    st.markdown(
        "The `monitor/` package is missing from the deployed repository. "
        "It must sit next to `app.py` at the repo root, lowercase, with these files:"
    )
    st.code(
        "monitor/__init__.py\n"
        "monitor/config.py\n"
        "monitor/detector.py\n"
        "monitor/zones.py\n"
        "monitor/state.py\n"
        "monitor/alerts.py\n"
        "monitor/activity.py\n"
        "monitor/render.py\n"
        "monitor/pipeline.py",
        language="text",
    )
    st.markdown(
        "Push the folder to GitHub, then reboot the app "
        "(**Manage app** -> **Reboot** in the bottom-right menu)."
    )
    st.code(
        'git add monitor\ngit status --short\n'
        'git commit -m "add monitor package"\ngit push',
        language="bash",
    )
    st.stop()

CONFIG_PATH = Path("config.yaml")


def load_base_settings() -> Settings:
    cfg = Settings.from_file(CONFIG_PATH if CONFIG_PATH.is_file() else None)
    if not cfg.desks.zones:
        cfg.desks.zones = zones_to_json(grid_zones(3, 2))
    return cfg


def deep_merge(base: dict, extra: dict) -> dict:
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def sidebar_settings(base: Settings) -> Settings:
    sb = st.sidebar
    sb.header("Monitor settings")

    if "zones_json" not in st.session_state:
        st.session_state["zones_json"] = json.dumps(base.desks.zones, indent=1)

    cols = sb.columns(2)
    if cols[0].button("6-desk grid", use_container_width=True):
        st.session_state["zones_json"] = json.dumps(zones_to_json(grid_zones(3, 2)))
    if cols[1].button("No zones", use_container_width=True):
        st.session_state["zones_json"] = "[]"
    zones_text = sb.text_area(
        "Desk zones (normalized JSON)",
        key="zones_json",
        height=150,
        help="Polygons in 0-1 coordinates. Each zone becomes one employee ID. "
        "Empty list = track people anywhere, no desk/absent detection.",
    )

    sb.divider()
    cfg = base
    cfg.detector.person_model = sb.selectbox(
        "YOLO model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], index=0
    )
    cfg.detector.conf = sb.slider("Person confidence", 0.10, 0.90, 0.35, 0.05)
    cfg.activity.phone_conf = sb.slider("Phone confidence", 0.10, 0.90, 0.35, 0.05)
    cfg.desks.absent_after_s = sb.slider("Absent after (seconds)", 1.0, 30.0, 5.0, 0.5)
    cfg.alerts.cooldown_s = sb.slider("Alert cooldown (seconds)", 5.0, 300.0, 30.0, 5.0)
    cfg.detector.imgsz = sb.select_slider("Inference size", [640, 768, 960, 1280], value=960)
    cfg.alerts.save_snapshots = sb.checkbox("Save alert snapshots", value=False)

    with sb.expander("Advanced overrides (JSON)"):
        raw = st.text_area("Merged over the settings above", value="", height=90)
    if raw.strip():
        try:
            cfg = Settings.from_dict(deep_merge(cfg.to_dict(), json.loads(raw)))
        except Exception as exc:  # noqa: BLE001 - surface bad input to the user
            sb.error(f"Override error: {exc}")
            cfg = base

    try:
        zones = json.loads(zones_text or "[]")
        cfg.desks.zones = zones if isinstance(zones, list) else []
    except json.JSONDecodeError as exc:
        sb.error(f"Zones JSON error: {exc}")
        cfg.desks.zones = base.desks.zones

    sb.caption(
        "Tips: 1) put one zone on each chair/desk, 2) zone order defines employee IDs, "
        "3) no zones = no ABSENT detection."
    )
    return cfg


def build_pipeline(cfg: Settings) -> MonitorPipeline:
    payload = json.dumps(cfg.to_dict(), sort_keys=True)
    with st.spinner("Loading model weights (first run downloads yolov8n.pt)..."):
        return MonitorPipeline(Settings.from_dict(json.loads(payload)))


def transcode_h264(src: Path, dst: Path) -> bool:
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        proc = subprocess.run(
            [
                exe,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(src),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(dst),
            ],
            capture_output=True,
        )
        return proc.returncode == 0 and dst.is_file() and dst.stat().st_size > 0
    except Exception:  # noqa: BLE001 - ffmpeg is a nice-to-have
        return False


def events_frame(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["Time (s)", "Clock", "Desk", "Type", "Message"])
    df = pd.DataFrame(rows)
    df["t"] = df["t"].map(lambda v: f"{v:.1f}")
    return df.rename(
        columns={
            "t": "Time (s)",
            "clock": "Clock",
            "desk": "Desk",
            "kind": "Type",
            "message": "Message",
        }
    )[["Time (s)", "Clock", "Desk", "Type", "Message"]]


def run_video(uploaded, cfg: Settings, stride: int, max_frames: int) -> None:
    workdir = Path(tempfile.mkdtemp(prefix="monitor_"))
    suffix = Path(uploaded.name).suffix or ".mp4"
    src_path = workdir / f"source{suffix}"
    src_path.write_bytes(uploaded.getbuffer())

    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        st.error("This video could not be opened. Try MP4 (H.264).")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1 or fps > 240:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if width <= 0 or height <= 0:
        st.error("Could not read video dimensions.")
        cap.release()
        return

    raw_path = workdir / "annotated_raw.mp4"
    writer = cv2.VideoWriter(
        str(raw_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1.0, fps / stride),
        (width, height),
    )

    pipe = build_pipeline(cfg)
    pipe.reset()

    progress = st.progress(0.0, text="Processing frames...")
    preview = st.empty()
    history: list[dict] = []
    idx = 0
    processed = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            t = idx / fps
            result = pipe.step(frame, t)
            writer.write(result.annotated)
            history.append(
                {"t": round(t, 2), "working": result.working, "alerts": result.total_alerts}
            )
            processed += 1
            if processed % 10 == 0:
                preview.image(
                    result.annotated, channels="BGR", caption="Live preview"
                )
            if total:
                progress.progress(min(1.0, idx / total), text=f"Frame {idx} / {total}")
            if max_frames and processed >= max_frames:
                break
        idx += 1

    cap.release()
    writer.release()
    if processed == 0:
        preview.empty()
        st.error("No frames were processed.")
        return
    progress.progress(1.0, text="Encoding output video...")

    final_path = workdir / "annotated.mp4"
    if not transcode_h264(raw_path, final_path):
        final_path = raw_path
    preview.empty()
    progress.empty()

    hist = pd.DataFrame(history)
    events = events_frame(pipe.alerts.rows())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Frames processed", processed)
    c2.metric("Peak working", int(hist["working"].max()))
    c3.metric("Average working", round(float(hist["working"].mean()), 1))
    c4.metric("Alerts", len(events))

    st.video(str(final_path))
    st.line_chart(hist.set_index("t")[["working"]], height=200)

    if len(events):
        st.subheader("Alerts")
        st.dataframe(events, use_container_width=True, hide_index=True)
        st.download_button(
            "Download alerts CSV",
            events.to_csv(index=False).encode("utf-8"),
            "alerts.csv",
            "text/csv",
        )
    st.download_button(
        "Download annotated video",
        final_path.read_bytes(),
        "annotated.mp4",
        "video/mp4",
    )

    if pipe.alerts.cfg.save_snapshots:
        snaps = list(Path(cfg.alerts.snapshot_dir).glob("*.jpg"))
        if snaps:
            st.caption(f"{len(snaps)} snapshot(s) saved in {cfg.alerts.snapshot_dir}/")


def run_image(uploaded, cfg: Settings) -> None:
    data = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        st.error("Could not decode this image.")
        return
    pipe = build_pipeline(cfg)
    pipe.reset()
    result = pipe.step(frame, 0.0)
    st.image(result.annotated, channels="BGR")
    if result.statuses:
        st.dataframe(
            pd.DataFrame(
                [
                    {"Desk": s.name, "State": s.state, "Track id": s.track_id}
                    for s in result.statuses
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    st.caption(
        "Single frames cannot confirm phone usage or absence (both need a few seconds "
        "of evidence). Upload a video to see the full state machine."
    )


def main() -> None:
    base = load_base_settings()
    cfg = sidebar_settings(base)

    st.title("AI Employee Monitoring")
    st.caption(
        "Person tracking + WORKING / PHONE_USAGE / ABSENT states + alerts, "
        "rendered like the reference dashboard."
    )

    tab_video, tab_image, tab_help = st.tabs(["Video", "Image", "How it works"])

    with tab_video:
        uploaded = st.file_uploader(
            "Upload an office / CCTV video", type=["mp4", "avi", "mov", "mkv"]
        )
        c1, c2 = st.columns(2)
        stride = c1.slider("Frame stride (2 = twice as fast, skips frames)", 1, 4, 1)
        max_frames = c2.number_input(
            "Max frames to process", min_value=30, max_value=20000, value=1200, step=30
        )
        if st.button("Run monitoring", type="primary", disabled=uploaded is None):
            try:
                run_video(uploaded, cfg, stride, int(max_frames))
            except Exception as exc:  # noqa: BLE001 - show the real error to the user
                st.exception(exc)

    with tab_image:
        image = st.file_uploader("Upload a frame", type=["jpg", "jpeg", "png"])
        if image is not None:
            try:
                run_image(image, cfg)
            except Exception as exc:  # noqa: BLE001 - show the real error to the user
                st.exception(exc)

    with tab_help:
        st.markdown(
            """
**Pipeline**

1. YOLOv8 detects people and tracks them with ByteTrack (stable IDs).
2. COCO `cell phone` detections inside a person's upper body count as phone evidence.
3. Each person's foot point is matched to a desk zone (configurable polygons).
4. A debounced state machine per desk outputs `WORKING`, `PHONE_USAGE` or `ABSENT`
   (empty desk for N seconds).
5. Alerts are logged with cooldowns; the overlay panel mirrors the reference UI.

**Improving accuracy**

- Train the crop classifier on Kaggle (`training/train_activity_classifier.py`)
  and set `activity.classifier_model` to the exported ONNX file.
- Fine-tune YOLO on your office footage (`training/train_yolo.py`) and set
  `detector.person_model` / `detector.phone_model`.

**Privacy**

Process footage you are allowed to process, and inform employees. Keep weights and
videos out of the public repo; only push code.
            """
        )


main()
