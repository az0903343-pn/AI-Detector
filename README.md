# AI Employee Monitor

Fixed-camera desk monitoring: people are detected and tracked, matched to desk
zones, and every desk gets a live state - **WORKING**, **PHONE_USAGE** or
**ABSENT** - with an alert log and a clean dashboard (live camera, video upload,
single photo). The annotated video mirrors the classic CCTV overlay: corner
boxes, ID chips, status panel and "Working / Alerts" counters.

## Features

- Live browser camera (WebRTC) with realtime overlay and status cards
- YOLOv8 person + cell-phone detection; no tracker compilation needed
  (lightweight IoU tracker, so no `lap` / ByteTrack install issues)
- Desk zones (normalized polygons) -> one employee ID per desk
- Debounced state machine: phone usage and absence need a few seconds of evidence
- Alerts with per-desk cooldowns, CSV export, optional snapshot JPEGs
- Video tab: annotated MP4 + alerts CSV + working-count chart
- Optional activity classifier (ONNX) trained on Kaggle
- Streamlit Community Cloud ready; repo can even be uploaded flat

## Pipeline

```
frame ──► YOLOv8 (person + phone) ──► IoU tracker (stable track IDs)
            │
            ├── phone boxes in upper body ──► phone evidence per person
            ├── person crop ──► activity classifier (ONNX, optional)
            └── foot point (bottom-center) ──► desk assignment (sticky)
                                                  │
                            per-desk state machine (dwell + hysteresis)
                                                  │
                        WORKING / PHONE_USAGE / ABSENT + alerts
                                                  │
                                  overlay renderer ──► browser / MP4
```

State rules (see `monitor/state.py`):

- **WORKING** - a tracked person is assigned to the desk.
- **PHONE_USAGE** - phone evidence for `phone_dwell_s` over a `phone_window_s`
  window (or the classifier says `phone` above `classifier_threshold`).
- **ABSENT** - no one assigned to the desk for `absent_after_s`; the desk is
  drawn red as `ID n - ABSENT`.
- Alerts fire only on transitions, at most once per `cooldown_s` per desk/type.

## Quick start (local)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the **Live camera** mode and press START (allow camera access). The first
run downloads `yolov8n.pt` (~6 MB). For speed on CPU keep inference size at 640.

Other modes: **Video** (upload MP4, download annotated video + alerts CSV) and
**Photo** (single frame; no phone/absence states from one image).

### CLI

```bash
python scripts/run_video.py --source office.mp4 --out runs/annotated.mp4 --show
python scripts/run_video.py --source 0 --out runs/webcam.mp4          # local webcam
python scripts/run_video.py --source rtsp://user:pass@host/stream --out runs/live.mp4
```

Outputs: annotated MP4, `runs/annotated.alerts.csv`, optional snapshots in `alerts_out/`.

### Desk zones

Zones are polygons in normalized (0-1) coordinates; order defines IDs.

```bash
python scripts/make_zones.py --source office.mp4 --time 5 --desks 6 --out zones.json
```

Click 3+ points per desk, `ENTER` closes a polygon, `q` saves. Paste the JSON into
the sidebar or `config.yaml` under `desks.zones`. Without zones the app tracks
people and phone usage but cannot report `ABSENT`.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub with the `monitor/` folder intact (GitHub web upload
   supports dragging a folder - keep the folder, not loose files).
2. <https://share.streamlit.io> -> **New app** -> repo/branch, main file `app.py`.
3. Use **Python 3.11 or 3.12** (Advanced settings). Newer 3.13/3.14 versions may
   lack wheels for `av`/`aiortc` and the live camera will not install.
4. Streamlit installs `requirements.txt` + `packages.txt` (system libs).
5. CPU-only: `yolov8n.pt`, inference 640, stride 1-2. Live camera runs a few
   frames per second on free hosting.

If the live video stays black behind a strict network, the browser could not
establish WebRTC: run the app locally for the camera, or add a TURN server in
`rtc_configuration` inside `run_live()`.

## Train on Kaggle

Use `notebooks/kaggle_training.ipynb` (GPU + Internet):

1. Attach a Kaggle dataset with your office videos.
2. Set `REPO_URL` and `VIDEOS`.
3. Sample person crops -> fix labels in `review.csv` -> rebuild folders.
4. Train `mobilenet_v3_small` -> `activity_classifier.onnx` + `.json`.
5. Commit both into `models/` and set:

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

Optional YOLO fine-tune (YOLO-format `data.yaml`):

```bash
python training/train_yolo.py --data data/office/data.yaml --epochs 50 --imgsz 960
# then set detector.person_model: weights/best.pt
```

## Configuration

`config.yaml` holds the defaults; the sidebar overrides the common ones.

| Key | Meaning |
| --- | --- |
| `detector.person_model` | COCO `yolov8n/s/m.pt` or your fine-tuned weights |
| `detector.phone_model` | optional custom phone detector (else COCO class 67) |
| `detector.imgsz` / `conf` / `iou` | inference size and thresholds |
| `activity.phone_dwell_s` | phone must persist this long for PHONE_USAGE |
| `activity.phone_hit_ratio` | share of window frames that need a phone |
| `activity.classifier_model` | optional ONNX classifier |
| `desks.absent_after_s` | empty-desk time before ABSENT |
| `desks.max_assign_dist` | how far a foot point can sit from a zone |
| `desks.zones` | desk polygons, order = employee ID |
| `alerts.cooldown_s` | per-desk cooldown between alerts |
| `alerts.save_snapshots` | write JPEG snapshots of alert frames |

## Project layout

```
app.py                      Streamlit dashboard (live / video / photo)
config.yaml                 runtime settings + example 6-desk layout
monitor/
  config.py                 typed settings
  tracker.py                dependency-free IoU tracker
  detector.py               YOLO detection + phone association
  zones.py                  polygons, sticky desk assignment, grid helper
  state.py                  debounced per-desk state machine
  alerts.py                 alert cooldown + CSV rows + snapshots
  activity.py               optional ONNX crop classifier
  render.py                 boxes, chips, status panel
  pipeline.py               frame -> FrameResult
scripts/                    run_video.py, make_zones.py
training/                   dataset builder, classifier, YOLO fine-tune
notebooks/kaggle_training.ipynb
tests/                      tracker, state, zones, config
```

## Tests

```bash
python -m unittest discover -s tests -v
```

## Troubleshooting

**`ModuleNotFoundError: No module named 'monitor'`**

The package folder did not reach GitHub. Upload the `monitor/` folder (with
`__init__.py` and all modules) next to `app.py`, then **Manage app -> Reboot**.
The app also supports flattened uploads and lists any missing modules on screen.

**`ModuleNotFoundError: No module named 'lap'`**

That came from Ultralytics ByteTrack. The current code uses the built-in IoU
tracker, so `lap` is not needed - upload the latest `monitor/detector.py` +
`monitor/tracker.py` and reboot.

**Live camera not installing / black video**

Use Python 3.11 or 3.12 in app settings. Behind strict networks WebRTC may need a
TURN server; the Video tab always works without a camera.

## Limitations

- One fixed camera; overlapping desks need tighter polygons.
- COCO `cell phone` works best when the phone is visible; a custom phone model or
  the trained classifier fixes most misses.
- Phone/absence needs a few seconds of evidence by design.
- CPU inference is slow - `yolov8n`, imgsz 640-960, or a GPU.

## Privacy

This kind of monitoring is regulated. Process only footage you are allowed to
use, inform employees, keep videos and weights out of the public repo, and prefer
local processing. The app stores nothing unless you enable snapshots.
