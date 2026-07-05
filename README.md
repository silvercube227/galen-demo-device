# galen-demo-device

Demo firmware for a benchtop imaging detector. The pipeline preprocesses incoming
frames, applies quality filters, and runs a detection gate before results are
accepted.

## Layout

```
detector/
  preprocess.py   # denoising, optional pre-detection smoothing, contrast normalization
  thresholds.py   # acceptance gate (signal, noise, intensity bounds)
  pipeline.py     # wires preprocessing + gate together
config/
  device.yaml     # tunable device parameters
main.py           # CLI for single-frame evaluation
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py path/to/frame.png
```

Exit code `0` means the frame passed the detection gate; `1` means it was rejected.

## Configuration

Edit `config/device.yaml` to adjust blur settings, filter mode, and gate thresholds
without changing application code.
