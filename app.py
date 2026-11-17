from __future__ import annotations

import html
import importlib.util
import json
import subprocess
import sys
import tempfile
import threading
import time
import types
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI Employee Monitor", layout="wide")

try:
    import av
    from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

    LIVE_AVAILABLE = True
except Exception:  # noqa: BLE001 - live tab is optional
    LIVE_AVAILABLE = False

APP_ROOT = Path(__file__).resolve().parent
_SKIP_DIRS = {".git", "venv", ".venv", "node_modules", "__pycache__"}
_MONITOR_MODULES = [
    "config", "tracker", "zones", "state", "alerts", "detector", "activity", "render",
    "pipeline",
]


def _monitor_parents() -> list[Path]:
    parents = []
    for match in APP_ROOT.rglob("pipeline.py"):
        if any(part in _SKIP_DIRS for part in match.parts):
            continue
        if match.parent.name == "monitor" and (match.parent / "__init__.py").is_file():
            parents.append(match.parent.parent)
    return parents


def _module_files() -> dict[str, Path]:
    found: dict[str, Path] = {}
    for name in _MONITOR_MODULES:
        candidates = sorted(
            (
                path
                for path in APP_ROOT.rglob(f"{name}.py")
                if not any(part in _SKIP_DIRS for part in path.parts)
            ),
            key=lambda path: len(path.parts),
        )
        for path in candidates:
            if name == "config":
                text = path.read_text(encoding="utf-8", errors="ignore")
                if "class Settings" not in text:
                    continue
            found[name] = path
            break
    return found


def _install_flat_monitor() -> bool:
    """Support repos where the folder structure was flattened by the uploader."""
    files = _module_files()
    if len(files) != len(_MONITOR_MODULES):
        return False
    package = types.ModuleType("monitor")
    package.__path__ = [str(APP_ROOT)]
    sys.modules["monitor"] = package
    for name in _MONITOR_MODULES:
        spec = importlib.util.spec_from_file_location(f"monitor.{name}", files[name])
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"monitor.{name}"] = module
        spec.loader.exec_module(module)
    return True


def _import_monitor():
    try:
        from monitor.config import Settings
        from monitor.pipeline import MonitorPipeline
        from monitor.zones import grid_zones, zones_to_json
        return Settings, MonitorPipeline, grid_zones, zones_to_json, None
    except ModuleNotFoundError as exc:
        name = exc.name or ""
        if name != "monitor" and not name.startswith("monitor."):
            return None, None, None, None, exc
        for parent in _monitor_parents():
            if str(parent) in sys.path:
                continue
            sys.path.insert(0, str(parent))
            importlib.invalidate_caches()
            try:
                from monitor.config import Settings
                from monitor.pipeline import MonitorPipeline
                from monitor.zones import grid_zones, zones_to_json
                return Settings, MonitorPipeline, grid_zones, zones_to_json, None
            except ModuleNotFoundError:
                sys.path.remove(str(parent))
        try:
            if _install_flat_monitor():
                from monitor.config import Settings
                from monitor.pipeline import MonitorPipeline
                from monitor.zones import grid_zones, zones_to_json
                return Settings, MonitorPipeline, grid_zones, zones_to_json, None
        except ModuleNotFoundError as flat_exc:
            return None, None, None, None, flat_exc
        return None, None, None, None, exc


Settings, MonitorPipeline, grid_zones, zones_to_json, _import_error = _import_monitor()

if _import_error is not None:
    st.error(f"Python import failed: {_import_error}")
    st.markdown("**Files deployed at the app root:**")
    listing = sorted(
        entry.name + ("/" if entry.is_dir() else "")
        for entry in APP_ROOT.iterdir()
        if entry.name != "__pycache__"
    )
    st.code("\n".join(listing) or "(empty)", language="text")
    found = [str(parent.relative_to(APP_ROOT)) + "/monitor" for parent in _monitor_parents()]
    if found:
        st.warning("Found a monitor package at: " + ", ".join(found))
    detected = _module_files()
    st.markdown(
        "**Monitor modules detected:** "
        + (", ".join(sorted(detected)) if detected else "none")
        + f" ({len(detected)}/{len(_MONITOR_MODULES)})"
    )
    st.markdown(
        "Upload the `monitor/` folder next to `app.py` (all files listed below), "
        "then reboot the app."
    )
    st.code("\n".join(f"monitor/{name}.py" for name in _MONITOR_MODULES), language="text")
    st.stop()

from monitor.state import ABSENT, PHONE, WORKING  # noqa: E402

CONFIG_PATH = Path("config.yaml")
STATE_CLASS = {WORKING: "working", PHONE: "phone", ABSENT: "absent"}

CSS = """
<style>
  .block-container {padding-top: 1.1rem; padding-bottom: 2.5rem; max-width: 1250px;}
  #MainMenu, footer {visibility: hidden;}
  html, body, [class*="css"] {
    font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  }
  .hero {
    background:
      radial-gradient(900px 260px at 12% 0%, rgba(34,197,94,.16), transparent 60%),
      linear-gradient(135deg, #0f172a, #111827 55%, #0b1220);
    border: 1px solid #1f2937; border-radius: 18px;
    padding: 22px 26px; margin-bottom: 16px;
    display: flex; justify-content: space-between; align-items: center;
    gap: 18px; flex-wrap: wrap;
  }
  .hero h1 {font-size: 1.65rem; margin: 0; color: #f8fafc; letter-spacing: -.02em;}
  .hero p {margin: 6px 0 0; color: #94a3b8; font-size: .93rem;}
  .legend {display: flex; gap: 14px; flex-wrap: wrap; color: #cbd5e1; font-size: .82rem;}
  .legend .dot {display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px;}
  .dot.working {background: #22c55e;}
  .dot.phone {background: #f59e0b;}
  .dot.absent {background: #ef4444;}

  .kpi-row {display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 2px 0 16px;}
  .kpi {background: #0f172a; border: 1px solid #1f2937; border-radius: 14px; padding: 14px 16px;}
  .kpi .label {color: #94a3b8; font-size: .74rem; text-transform: uppercase; letter-spacing: .07em;}
  .kpi .value {color: #f8fafc; font-size: 1.75rem; font-weight: 700; line-height: 1.25;}
  .kpi.working {border-left: 3px solid #22c55e;}
  .kpi.phone {border-left: 3px solid #f59e0b;}
  .kpi.absent {border-left: 3px solid #ef4444;}
  .kpi.alerts {border-left: 3px solid #38bdf8;}
  .kpi.neutral {border-left: 3px solid #64748b;}
  .kpi.working .value {color: #4ade80;}
  .kpi.phone .value {color: #fbbf24;}
  .kpi.absent .value {color: #f87171;}

  .section-title {
    font-size: .74rem; letter-spacing: .08em; text-transform: uppercase;
    color: #64748b; margin: 18px 0 8px;
  }
  .desk-grid {display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 10px;}
  .desk {background: #0f172a; border: 1px solid #1f2937; border-radius: 12px; padding: 12px 14px;}
  .desk .top {display: flex; justify-content: space-between; align-items: center; gap: 8px;}
  .desk .id {font-weight: 600; color: #e2e8f0;}
  .desk .meta {color: #64748b; font-size: .76rem; margin-top: 7px;}
  .desk.working {border-color: rgba(34,197,94,.35);}
  .desk.phone {border-color: rgba(245,158,11,.4);}
  .desk.absent {border-color: rgba(239,68,68,.4);}
  .badge {
    font-size: .66rem; font-weight: 700; letter-spacing: .05em;
    padding: 3px 8px; border-radius: 999px; white-space: nowrap;
  }
  .badge.working {background: rgba(34,197,94,.14); color: #4ade80; border: 1px solid rgba(34,197,94,.35);}
  .badge.phone {background: rgba(245,158,11,.14); color: #fbbf24; border: 1px solid rgba(245,158,11,.35);}
  .badge.absent {background: rgba(239,68,68,.14); color: #f87171; border: 1px solid rgba(239,68,68,.35);}
  .badge.neutral {background: rgba(148,163,184,.12); color: #cbd5e1; border: 1px solid rgba(148,163,184,.3);}

  .alert-list {display: flex; flex-direction: column; gap: 8px;}
  .alert-row {
    display: flex; align-items: center; gap: 10px;
    background: #0f172a; border: 1px solid #1f2937; border-radius: 10px;
    padding: 9px 12px; font-size: .84rem; color: #cbd5e1;
  }
  .alert-msg {flex: 1;}
  .alert-time {color: #64748b; font-variant-numeric: tabular-nums;}
  .empty {color: #64748b; font-size: .86rem; padding: 6px 0;}

  [data-testid="stSidebar"] {border-right: 1px solid #1f2937;}
  [data-testid="stSidebar"] h2 {font-size: 1.05rem;}
</style>
"""


def _esc(value) -> str:
    return html.escape(str(value))


def kpi_row(items: list[tuple[str, object, str]]) -> str:
    cards = "".join(
        f'<div class="kpi {cls}"><div class="label">{_esc(label)}</div>'
        f'<div class="value">{_esc(value)}</div></div>'
        for label, value, cls in items
    )
    return f'<div class="kpi-row">{cards}</div>'


def desks_html(statuses) -> str:
    if not statuses:
        return '<div class="empty">Waiting for people...</div>'
    cards = []
    for status in statuses:
        cls = STATE_CLASS.get(status.state, "neutral")
        meta = (
            f"track #{status.track_id}"
            if status.track_id is not None
            else "nobody at this desk"
        )
        cards.append(
            f'<div class="desk {cls}"><div class="top"><span class="id">{_esc(status.name)}</span>'
            f'<span class="badge {cls}">{_esc(status.state)}</span></div>'
            f'<div class="meta">{_esc(meta)}</div></div>'
        )
    return '<div class="desk-grid">' + "".join(cards) + "</div>"


def alerts_html(rows: list[dict]) -> str:
    if not rows:
        return '<div class="empty">No alerts yet.</div>'
    items = []
    for row in list(reversed(rows))[:6]:
        kind = str(row.get("kind", ""))
        cls = "phone" if kind == "phone" else "absent"
        items.append(
            f'<div class="alert-row"><span class="badge {cls}">{_esc(kind.upper())}</span>'
            f'<span class="alert-msg">{_esc(row.get("message", ""))}</span>'
            f'<span class="alert-time">{_esc(row.get("clock", ""))}</span></div>'
        )
    return '<div class="alert-list">' + "".join(items) + "</div>"


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
    sb.markdown("### Settings")
    sb.caption("Changes apply the next time the camera or video starts.")

    if "zones_json" not in st.session_state:
        st.session_state["zones_json"] = json.dumps(base.desks.zones, indent=1)

    cfg = Settings.from_dict(base.to_dict())

    with sb.expander("Desk zones", expanded=False):
        counts = sb.columns(2)
        if counts[0].button("6 desks", use_container_width=True):
            st.session_state["zones_json"] = json.dumps(zones_to_json(grid_zones(3, 2)))
        if counts[1].button("No zones", use_container_width=True):
            st.session_state["zones_json"] = "[]"
        zones_text = sb.text_area(
            "Zones (normalized JSON)",
            key="zones_json",
            height=170,
            help="Each polygon becomes one employee ID. Empty list = track people "
            "anywhere and skip absence detection.",
        )
        sb.caption("Draw custom zones with `scripts/make_zones.py`.")

    with sb.expander("Detection", expanded=True):
        models = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]
        default_model = base.detector.person_model
        index = models.index(default_model) if default_model in models else 0
        cfg.detector.person_model = sb.selectbox("YOLO model", models, index=index)
        cfg.detector.conf = sb.slider(
            "Person confidence", 0.10, 0.90, float(base.detector.conf), 0.05
        )
        cfg.activity.phone_conf = sb.slider(
            "Phone confidence", 0.10, 0.90, float(base.activity.phone_conf), 0.05
        )
        sizes = [480, 640, 768, 960, 1280]
        default_size = min(sizes, key=lambda v: abs(v - base.detector.imgsz))
        cfg.detector.imgsz = sb.select_slider(
            "Inference size", sizes, value=default_size,
            help="640 is the sweet spot on CPU; larger is slower but sees further.",
        )

    with sb.expander("Behaviour", expanded=False):
        cfg.activity.phone_dwell_s = sb.slider(
            "Phone dwell (seconds)", 0.5, 5.0, float(base.activity.phone_dwell_s), 0.5
        )
        cfg.desks.absent_after_s = sb.slider(
            "Absent after (seconds)", 1.0, 30.0, float(base.desks.absent_after_s), 0.5
        )
        cfg.alerts.cooldown_s = sb.slider(
            "Alert cooldown (seconds)", 5.0, 300.0, float(base.alerts.cooldown_s), 5.0
        )
        cfg.alerts.save_snapshots = sb.checkbox(
            "Save alert snapshots", value=bool(base.alerts.save_snapshots)
        )

    with sb.expander("Advanced JSON", expanded=False):
        raw = sb.text_area("Merged over the settings above", value="", height=90)
    if raw.strip():
        try:
            cfg = Settings.from_dict(deep_merge(cfg.to_dict(), json.loads(raw)))
        except Exception as exc:  # noqa: BLE001 - surface bad input to the user
            sb.error(f"Override error: {exc}")

    try:
        zones = json.loads(zones_text or "[]")
        cfg.desks.zones = zones if isinstance(zones, list) else []
    except json.JSONDecodeError as exc:
        sb.error(f"Zones JSON error: {exc}")
        cfg.desks.zones = base.desks.zones

    sb.divider()
    sb.caption(
        "Tips: one zone per chair, zone order = employee IDs, "
        "no zones = no ABSENT detection."
    )
    return cfg


def build_pipeline(cfg: Settings) -> MonitorPipeline:
    payload = json.dumps(cfg.to_dict(), sort_keys=True)
    with st.spinner("Loading model (the first run downloads yolov8n.pt)..."):
        return MonitorPipeline(Settings.from_dict(json.loads(payload)))


if LIVE_AVAILABLE:

    class LiveProcessor(VideoProcessorBase):
        def __init__(self, cfg_dict: dict):
            self.pipeline = MonitorPipeline(
                Settings.from_dict(cfg_dict),
                alert_dir=str(Path(tempfile.gettempdir()) / "monitor_live"),
            )
            self.started = time.time()
            self._lock = threading.Lock()
            self._result = None
            self._error = ""

        def recv(self, frame):
            image = frame.to_ndarray(format="bgr24")
            try:
                result = self.pipeline.step(image, time.time() - self.started)
                with self._lock:
                    self._result = result
                    self._error = ""
                image = result.annotated
            except Exception as exc:  # noqa: BLE001 - keep the stream alive
                with self._lock:
                    self._error = str(exc)
            return av.VideoFrame.from_ndarray(image, format="bgr24")

        def snapshot(self):
            with self._lock:
                rows = self.pipeline.alerts.rows()
                return self._result, self._error, rows

    def _live_factory(cfg_dict: dict):
        def factory():
            return LiveProcessor(cfg_dict)

        return factory

else:
    LiveProcessor = None


def run_live(cfg: Settings) -> None:
    if not LIVE_AVAILABLE:
        st.info(
            "Live camera needs `streamlit-webrtc` and `av`. Run "
            "`pip install -r requirements.txt` and restart the app."
        )
        return

    live_cfg = Settings.from_dict(cfg.to_dict())
    live_cfg.alerts.save_snapshots = False

    st.caption(
        "Press START and allow camera access. On free hosting CPU the "
        "detection runs a few frames per second - use inference size 640 for speed. "
        "Settings changes apply after stopping and starting the camera; "
        "press STOP to switch to Video or Photo."
    )

    ctx = webrtc_streamer(
        key="live-camera",
        video_processor_factory=_live_factory(live_cfg.to_dict()),
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={
            "video": {
                "width": {"ideal": 960},
                "height": {"ideal": 540},
                "frameRate": {"ideal": 15, "max": 30},
            },
            "audio": False,
        },
        async_processing=True,
    )

    if not ctx.state.playing:
        st.info("Camera is off. Press **START** above to begin monitoring.")
        return

    kpi_slot = st.empty()
    desks_slot = st.empty()
    alerts_slot = st.empty()
    error_slot = st.empty()

    while ctx.state.playing:
        processor = getattr(ctx, "video_processor", None)
        if processor is None:
            kpi_slot.info("Connecting to the camera...")
        else:
            result, error, rows = processor.snapshot()
            if error:
                error_slot.warning(f"Frame error: {error}")
            else:
                error_slot.empty()
            if result is None:
                kpi_slot.info("Warming up - the first detection takes a few seconds...")
                time.sleep(0.5)
                continue
            phone = sum(1 for s in result.statuses if s.state == PHONE)
            absent = sum(1 for s in result.statuses if s.state == ABSENT)
            kpi_slot.markdown(
                kpi_row(
                    [
                        ("Working", result.working, "working"),
                        ("On phone", phone, "phone"),
                        ("Absent", absent, "absent"),
                        ("Alerts", result.total_alerts, "alerts"),
                    ]
                ),
                unsafe_allow_html=True,
            )
            desks_slot.markdown(
                '<div class="section-title">Desks</div>' + desks_html(result.statuses),
                unsafe_allow_html=True,
            )
            alerts_slot.markdown(
                '<div class="section-title">Latest alerts</div>' + alerts_html(rows),
                unsafe_allow_html=True,
            )
        time.sleep(0.5)


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


def alerts_frame(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["Clock", "Desk", "Type", "Message"])
    df = pd.DataFrame(rows)
    return df.rename(
        columns={"clock": "Clock", "desk": "Desk", "kind": "Type", "message": "Message"}
    )[["Clock", "Desk", "Type", "Message"]]


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
                preview.image(result.annotated, channels="BGR", caption="Live preview")
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
    events = alerts_frame(pipe.alerts.rows())
    st.markdown(
        kpi_row(
            [
                ("Frames", processed, "neutral"),
                ("Peak working", int(hist["working"].max()), "working"),
                ("Avg working", round(float(hist["working"].mean()), 1), "neutral"),
                ("Alerts", len(events), "alerts"),
            ]
        ),
        unsafe_allow_html=True,
    )

    st.video(str(final_path))
    st.line_chart(hist.set_index("t")[["working"]], height=200)

    if len(events):
        st.markdown('<div class="section-title">Alerts</div>', unsafe_allow_html=True)
        st.dataframe(events, use_container_width=True, hide_index=True)
        st.download_button(
            "Download alerts CSV",
            events.to_csv(index=False).encode("utf-8"),
            "alerts.csv",
            "text/csv",
        )
    st.download_button(
        "Download annotated video", final_path.read_bytes(), "annotated.mp4", "video/mp4"
    )


def run_image(uploaded, cfg: Settings) -> None:
    data = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        st.error("Could not decode this image.")
        return
    pipe = build_pipeline(cfg)
    pipe.reset()
    result = pipe.step(frame, 0.0)
    phone = sum(1 for s in result.statuses if s.state == PHONE)
    absent = sum(1 for s in result.statuses if s.state == ABSENT)
    st.markdown(
        kpi_row(
            [
                ("People", len(result.persons), "neutral"),
                ("On phone", phone, "phone"),
                ("Absent desks", absent, "absent"),
                ("Alerts", result.total_alerts, "alerts"),
            ]
        ),
        unsafe_allow_html=True,
    )
    st.image(result.annotated, channels="BGR")
    st.markdown(
        '<div class="section-title">Desks</div>' + desks_html(result.statuses),
        unsafe_allow_html=True,
    )
    st.caption(
        "A single photo cannot confirm phone usage or absence - both need a few "
        "seconds of evidence. Use the live camera or a video for that."
    )


def header() -> None:
    st.markdown(
        """
<div class="hero">
  <div>
    <h1>AI Employee Monitor</h1>
    <p>Live desk attendance, phone usage and absence from one camera.</p>
  </div>
  <div class="legend">
    <span><span class="dot working"></span>Working</span>
    <span><span class="dot phone"></span>Phone usage</span>
    <span><span class="dot absent"></span>Absent</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    base = load_base_settings()
    cfg = sidebar_settings(base)
    header()

    mode = st.radio(
        "View",
        ["Live camera", "Video", "Photo"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if mode == "Live camera":
        run_live(cfg)

    elif mode == "Video":
        uploaded = st.file_uploader(
            "Upload an office / CCTV video", type=["mp4", "avi", "mov", "mkv"]
        )
        c1, c2 = st.columns(2)
        stride = c1.slider("Frame stride (2 = faster, skips frames)", 1, 4, 1)
        max_frames = c2.number_input(
            "Max frames", min_value=30, max_value=20000, value=1200, step=30
        )
        if st.button(
            "Run monitoring", type="primary", disabled=uploaded is None, use_container_width=True
        ):
            try:
                run_video(uploaded, cfg, stride, int(max_frames))
            except Exception as exc:  # noqa: BLE001 - show the real error to the user
                st.exception(exc)

    else:
        image = st.file_uploader("Upload a frame", type=["jpg", "jpeg", "png"])
        if image is not None:
            try:
                run_image(image, cfg)
            except Exception as exc:  # noqa: BLE001 - show the real error to the user
                st.exception(exc)

    with st.expander("How it works"):
        st.markdown(
            """
1. YOLOv8 detects people and cell phones; a small IoU tracker keeps stable IDs.
2. Phone evidence inside a person's upper body drives `PHONE_USAGE`.
3. Foot points are matched to desk zones, so every desk gets an employee ID.
4. A debounced state machine outputs `WORKING`, `PHONE_USAGE` or `ABSENT`
   (desk empty for N seconds).
5. Alerts respect a per-desk cooldown; videos also produce an alerts CSV.

**Higher accuracy:** train the crop classifier on Kaggle and set
`activity.classifier_model`, or fine-tune YOLO with `training/train_yolo.py`.

**Privacy:** process footage you are allowed to use and keep videos/weights
out of the public repo.
            """
        )


main()
