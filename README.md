# Aircraft turnaround monitor

Real-time aircraft turnaround monitoring from a live airport video feed (YouTube live stream) or a recorded video.
A custom YOLOv8 model detects the aircraft and four "connected" equipment classes (aerobridge, catering truck,
cargo loader, pushback tug). ByteTrack plus a temporal layer (hysteresis, grace periods, coasting aircraft
identities, manual ROI, camera-view guard) turns noisy per-frame detections into stable service events, stored in
SQLite and shown in a Streamlit dashboard with turnaround analytics.

The complete functional specification (architecture, data contracts, every function, extension guide) is in
[`docs/PIPELINE_SPEC.md`](docs/PIPELINE_SPEC.md).

## Install

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
# GPU (RTX 3050): install a CUDA build of torch first, see https://pytorch.org/get-started/locally/
pip install -r requirements.txt
python scripts/check_environment.py                       # verifies imports, CUDA, FFmpeg backend, config
```

## Run

1. Copy your trained weights to `weights/best.pt` (or set `model.weights_path`).
2. Check that `classes:` in `configs/default.yaml` matches your model's class names:
   `python scripts/check_environment.py --load-model`
3. Validate the setup on a recorded video first (`source.kind: file`, `source.uri: data/your_clip.mp4`):
   ```bash
   python scripts/run_pipeline.py --source data/your_clip.mp4 --kind file
   ```
4. Go live on YouTube:
   ```bash
   python scripts/run_pipeline.py --kind youtube --source "https://www.youtube.com/watch?v=XXXX"
   ```
5. Dashboard (second terminal): `python scripts/run_dashboard.py` and open http://localhost:8501

## Define the ROI (fixed camera)

```bash
python scripts/select_roi.py --kind youtube --source "https://www.youtube.com/watch?v=XXXX"
```
Click the polygon around the stand, press Enter and paste the printed YAML into `configs/default.yaml`.
Pick streams with a static camera. If the view changes (PTZ / camera switch) the scene guard pauses the analysis
until the original view returns.

## Moving to another PC

Copy the folder (with the weights), create a venv, `pip install -r requirements.txt`, run
`python scripts/check_environment.py --load-model`. All settings live in the YAML; relative paths resolve against the
project root (override with `TURNAROUND_HOME`). Without a GPU the pipeline falls back to CPU automatically.

## Tests and documentation

```bash
python -m pytest                              # unit + integration + dashboard smoke tests
python tools/generate_spec.py --strict        # regenerate docs/PIPELINE_SPEC.md (fails on undocumented code)
```

## Notes

Use this for research and demos only: respect YouTube's terms of service and do not redistribute stream footage.
