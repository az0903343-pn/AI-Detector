# AI Employee Monitoring

Fixed-camera desk monitoring: people are detected and tracked, matched to desk zones,
and each desk gets a live state - **WORKING**, **PHONE_USAGE** or **ABSENT** - with
alerts and an overlay that mirrors the reference dashboard (boxes, ID chips, status
panel, "Working: N / Alerts: M").

## Features

- YOLOv8 person detection + ByteTrack IDs (stable identity per person)
- Desk zones (normalized polygons) -> one employee ID per desk, like the reference UI
- Phone-usage detection from COCO `cell phone` boxes inside the person's upper body
- Optional custom activity classifier (ONNX) per person crop - trained on Kaggle
- Debounced state machine so single frames never flip a state
- Alerts with per-desk cooldowns, CSV log and optional snapshot JPEGs
- Streamlit app: upload a video, watch a live preview, download annotated MP4 + alerts CSV
- CLI for files, webcams and RTSP streams
- Kaggle scripts: dataset builder, MobileNetV3 classifier, YOLO fine-tune

## Pipeline

```
frame ──► YOLOv8 + ByteTrack ──► persons (track_id, box)
            │
            ├── phone boxes in upper body ──► phone evidence per person
            ├── person crop ──► activity classifier (ONNX, optional)
            └── foot point (bottom-center) ──► desk assignment (sticky)
                                                  │
                            per-desk state machine (dwell + hysteresis)
                                                  │
                        WORKING / PHONE_USAGE / ABSENT + alerts
                                                  │
                                  overlay renderer ──► MP4 / Streamlit
```

State rules (see `monitor/state.py`):

- **WORKING** - a tracked person is assigned to the desk.
- **PHONE_USAGE** - phone evidence for `phone_dwell_s` over a `phone_window_s` window
  (or the classifier says `phone` above `classifier_threshold`).
- **ABSENT** - no one assigned to the desk for `absent_after_s`; the desk polygon is
  drawn red as `ID n - ABSENT`.
- Alerts fire only on transitions, at most once per `cooldown_s` per desk and alert type.

## Quick start (local)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads `yolov8n.pt` (~6 MB). Upload an office video in the **Video**
tab. If the auto 3x2 desk grid does not match your camera, draw zones first (below).

### CLI

```bash
python scripts/run_video.py --source office.mp4 --out runs/annotated.mp4 --show
python scripts/run_video.py --source 0 --out runs/webcam.mp4          # webcam
python scripts/run_video.py --source rtsp://user:pass@host/stream --out runs/live.mp4
```

Outputs: annotated MP4, `runs/annotated.alerts.csv`, optional snapshots in `alerts_out/`.

### Desk zones

Zones are polygons in normalized (0-1) coordinates; the order defines the IDs.

```bash
python scripts/make_zones.py --source office.mp4 --time 5 --desks 6 --out zones.json
```

Click 3+ points per desk, `ENTER` to close, `q` to save. Copy the printed block into
`config.yaml` under `desks.zones`, or paste the JSON into the Streamlit sidebar.
Without zones the app still tracks people and detects phone usage, but cannot report
`ABSENT`.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (code only - do not commit videos or weights).
2. Go to <https://share.streamlit.io> -> **New app** -> pick the repo/branch,
   main file `app.py`.
3. Python 3.11 is fine; Streamlit installs `requirements.txt` and `packages.txt`
   (system libs) automatically.
4. Streamlit Cloud is CPU-only: keep `yolov8n.pt`, `imgsz` 640-960, and a frame
   stride of 1-2 for demo videos.

## Train on Kaggle

Use `notebooks/kaggle_training.ipynb` (GPU + Internet enabled):

1. Add a Kaggle dataset with your office videos.
2. Set `REPO_URL` (your GitHub repo) and `VIDEOS` paths.
3. Run: sample person crops -> fix labels in `review.csv` -> rebuild folder structure.
4. Train `mobilenet_v3_small` -> `activity_classifier.onnx` + `.json`.
5. Put both files in `models/` and set:

```yaml
activity:
  classifier_model: models/activity_classifier.onnx
```

Manual equivalents:

```bash
python training/prepare_activity_dataset.py --videos data/videos --out data/activity --stride 15
python training/prepare_activity_dataset.py --from-csv data/activity/review.csv --out data/activity
python training/train_activity_classifier.py --data data/activity/crops \
    --out runs/activity --export-onnx models/activity_classifier.onnx
```

Optional YOLO fine-tune for your camera (YOLO-format `data.yaml`):

```bash
python training/train_yolo.py --data data/office/data.yaml --epochs 50 --imgsz 960
# then set detector.person_model: weights/best.pt
```

## Configuration

All runtime knobs live in `config.yaml` (the sidebar can override the common ones):

| Key | Meaning |
| --- | --- |
| `detector.person_model` | COCO `yolov8n/s/m.pt` or your fine-tuned weights |
| `detector.phone_model` | optional custom phone detector (else COCO class 67) |
| `detector.imgsz` / `conf` / `iou` | inference size and thresholds |
| `activity.phone_dwell_s` | phone must persist this long to switch to PHONE_USAGE |
| `activity.phone_hit_ratio` | share of frames in the window that need a phone |
| `activity.classifier_model` | optional ONNX classifier |
| `desks.absent_after_s` | empty-desk time before ABSENT |
| `desks.max_assign_dist` | how far a foot point can sit from a zone |
| `desks.zones` | desk polygons, order = employee ID |
| `alerts.cooldown_s` | per-desk cooldown between alerts |
| `alerts.save_snapshots` | write JPEG snapshots of alert frames |

## Project layout

```
app.py                      Streamlit app (video / image / help tabs)
config.yaml                 runtime settings + example 6-desk layout
monitor/
  config.py                 typed settings
  detector.py               YOLO person tracking + phone association
  zones.py                  polygons, sticky desk assignment, grid helper
  state.py                  debounced per-desk state machine
  alerts.py                 alert cooldown + CSV rows + snapshots
  activity.py               optional ONNX crop classifier
  render.py                 boxes, chips, status panel
  pipeline.py               frame -> FrameResult
scripts/
  run_video.py              CLI for files / webcams / RTSP
  make_zones.py             interactive zone picker
training/
  prepare_activity_dataset.py   videos -> labelled crops (+ review CSV)
  train_activity_classifier.py  MobileNetV3 -> ONNX
  train_yolo.py                 YOLOv8 fine-tune wrapper
notebooks/kaggle_training.ipynb Kaggle workflow
tests/                      unit tests for state, zones, config
```

## Tests

```bash
python -m unittest discover -s tests -v
```

## Troubleshooting

**`ModuleNotFoundError` / `from monitor.config import Settings` on Streamlit Cloud**

The Python package folder did not reach GitHub. The repo root must look like this:

```
app.py
config.yaml
requirements.txt
packages.txt
monitor/
  __init__.py
  config.py
  detector.py
  zones.py
  state.py
  alerts.py
  activity.py
  render.py
  pipeline.py
```

Fix: upload/push the `monitor/` folder (GitHub web: **Add file -> Upload files**,
drag the folder), then **Manage app -> Reboot** in Streamlit Cloud. If files were
uploaded loose (e.g. `config.py` at the repo root), delete them and upload the folder
with its name so imports resolve.

## Limitations

- A single fixed camera per zone; overlapping desks need tighter polygons.
- COCO `cell phone` works best when the phone is visible to the camera; a custom
  phone model or the trained classifier fixes most misses.
- Phone/absence needs a few seconds of evidence by design; single images cannot
  produce those states.
- CPU inference is slow - use `yolov8n`, `imgsz` 640-960, or a GPU.

## Privacy

This kind of monitoring is regulated. Process only footage you are allowed to use,
inform employees, keep videos and weights out of the public repo, and prefer local
processing. The app stores nothing unless you enable snapshots.
