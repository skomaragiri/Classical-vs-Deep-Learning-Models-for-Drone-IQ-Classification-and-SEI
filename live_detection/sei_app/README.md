# SEI App — the missing pieces for your pipeline

This repo only contains what your architecture is **missing**. It does not
duplicate `pylibos_watch_dir.py`, `pylibos_run_file.py`,
`pylibos_common.py`, `collate_filter.py`, or `resample_cli.grc` — those
stay as-is.

## Mapping to your architecture diagram

```
Stage 1 — Capture                       your hardware (Air-T)
Stage 2 — Watcher                       pylibos_watch_dir.py + pylibos_run_file.py
                                        (emits OmniSIG-annotated SigMF pairs)

Stage 3 — Compute Channel Params        compute_channel_params.py   ← NEW (this repo)

Stage 4a — Channelize to target rate    resample_cli.py              ← yours (from .grc)
Stage 4b — SEI classification           infer.py                     ← NEW (this repo)
Stage 4c — Orchestrate 3 + 4a + 4b      gnuradio_watcher.py          ← NEW (this repo)
Stage 4d — Delete SigMF files           handled by gnuradio_watcher.py
                                          with --delete-originals

Offline — Train the SEI model           train.py                     ← NEW (this repo)
```

Everything else (`sigmf_utils.py`, `features.py`, `sei_model.py`) is
internal support code used by the three stage scripts above.

## Files in this repo

| File | Role |
|---|---|
| `compute_channel_params.py` | **Stage 3.** Reads an OmniSIG-annotated `.sigmf-meta` and emits `{bw, offset, cf, samp_rate, label, …}` per annotation. Dual-use: importable library + CLI (`--pretty` JSON or JSONL). |
| `infer.py` | **Stage 4b — the SEI App.** Two entry points: classify a raw-complex64 file produced by `resample_cli.py` (`--channelized-file`), or standalone classify an annotated SigMF pair (`--annotated-datafile`). |
| `gnuradio_watcher.py` | **Stage 4 orchestrator.** Watches the directory `pylibos_watch_dir.py` writes to, chains Stage 3 → `resample_cli.py` subprocess → SEI App → delete/move, optionally publishes over ZMQ. |
| `train.py` | **Offline training.** Walks a directory of OmniSIG-annotated captures of your drone, channelizes in-process, fits autoencoder + One-Class SVM, calibrates threshold, saves model. |
| `sei_model.py` | Model code: 1-D CNN autoencoder (tensor layout `(B,N,2)` matching `pylibos_common.numpy_to_tensor`) + One-Class SVM + hybrid scoring + save/load. |
| `features.py` | Engineered impairment features: CFO, IQ imbalance α/φ, circularity, DC offset, envelope moments, phase noise, spectral asymmetry, 24-bin PSD shape. 37-dim vector. |
| `sigmf_utils.py` | Training helpers that supplement `pylibos_common` (not a replacement): `annotation_iq`, `channelize_inmem`, `iq_window_to_bn2`, `slice_windows`, `read_raw_complex64`. |
| `requirements.txt` | Python deps beyond what pylibos already needs. |

All the new code tries `from pylibos_common import …` first and only
falls back to inline helpers when the module isn't importable, so you can
drop this directory next to your existing code and it will reuse
`get_samples_mmap`, `numpy_to_tensor`, `get_meta`, `write_json_to_file`,
and `dsprint`.

## How the pipeline fits together at run time

```
Air-T + OmniSIG
      │
      ▼  (raw + .sigmf-meta pairs)
pylibos_watch_dir.py --watchdir /sdr/raw  --output-path /sdr/omnisig_out
      │
      ▼  (OmniSIG-annotated pairs)
gnuradio_watcher.py --watchdir /sdr/omnisig_out  --output-path /sdr/sei_out  \
                    --resample-cli ./resample_cli.py  --model ./sei_model   \
                    --target-rate 10e6  --delete-originals
      │
      │  for each annotation:
      │    ① compute_channel_params.compute_from_meta()
      │    ② annotation_iq() → temp cf32_le file
      │    ③ subprocess: resample_cli.py --samp-rate --bw --offset-freq --target-rate
      │    ④ infer.classify_baseband() on the channelized output
      │    ⑤ write sei:verdict / sei:hybrid_score / … into the annotation
      │    ⑥ move enriched .sigmf-meta to --output-path   (or delete if --delete-originals)
      ▼
(optional ZMQ PUB at :9200 with the enriched meta as JSON)
```

The Stage 4a call is literally your `.grc`-generated script — this
orchestrator just hands it the right `--samp-rate` / `--bw` /
`--offset-freq` / `--target-rate` per annotation via subprocess and reads
the raw-complex64 output back.

## Computing channel params (answers the original question)

`compute_channel_params.py` is what you asked about. For each OmniSIG
annotation it does:

```
BW     = core:freq_upper_edge - core:freq_lower_edge
CF     = core:freq_lower_edge + BW/2
Offset = CF - captures[0]['core:frequency']       (recorder center freq)
```

and emits that, plus `samp_rate`, `label`, `snr_estimate`,
`sample_start`, `sample_count`. As a CLI:

```bash
# JSONL (one annotation per line) — pipe into a shell loop
python compute_channel_params.py --metafile capture.sigmf-meta

# or as a pretty JSON list
python compute_channel_params.py --metafile capture.sigmf-meta --pretty
```

As a library:

```python
from compute_channel_params import compute_from_meta
from pylibos_common import get_meta
params = compute_from_meta(get_meta("capture.sigmf-meta"))
for p in params:
    print(p["label"], p["bw"], p["offset"])
```

## Quick start

```bash
pip install -r requirements.txt

# 1. Record 5+ sessions of YOUR drone across different days / locations
#    and run your existing OmniSIG flow so you get annotated .sigmf-meta
#    files under e.g. /data/our_drone/.

# 2. Train the one-class SEI model
python train.py --data-dir /data/our_drone  --out-dir ./sei_model \
                --label-filter drone_c2     --epochs 40

# 3. Validate standalone on one capture
python infer.py --annotated-datafile /captures/test.sigmf-data \
                --model ./sei_model

# 4. Production: orchestrator behind pylibos_watch_dir.py
python gnuradio_watcher.py \
    --watchdir     /sdr/omnisig_out           \
    --output-path  /sdr/sei_out               \
    --resample-cli ./resample_cli.py          \
    --model        ./sei_model                \
    --target-rate  10e6                       \
    --delete-originals                        \
    --enable-zmq --port 9200
```

`infer.py` also supports Mode 1 — classifying a raw-complex64 file that
`resample_cli.py` has already produced, for cases where you want to drive
the pipeline by hand:

```bash
python resample_cli.py --file-input  soi.sigmf-data \
                       --file-output soi_chan.raw   \
                       --samp-rate 30.72e6 --bw 10e6 --offset-freq 2e6 \
                       --target-rate 10e6

python infer.py --channelized-file soi_chan.raw --target-rate 10e6 \
                --model ./sei_model
```

## One-drone training — is it enough?

Yes, mechanically. This is one-class classification / novelty detection:
the autoencoder + OC-SVM learn the joint distribution of YOUR drone's
impairments, and everything else (including the same make/model with
different PA / mixer silicon) scores as anomalous.

The trap: if you train on one recording session, the model also learns
that day's RF environment. Tomorrow over different ground, your own
drone will be called foreign. Required discipline:

* **5+ sessions minimum** across different days, locations, altitudes,
  battery states, and temperatures.
* Hold out a whole session for validation, not random windows from a
  session the model saw — random splits leak.
* Scatter-plot `sei:ae_score` vs `sei:ocsvm_score` over a test session:
  your drone should cluster tightly; unseen captures should scatter.
* Record indoors and outdoors and compare verdicts. If they flip you're
  learning channel, not hardware.

`train.py` augments with random CFO / phase / AWGN, but augmentation
can't substitute for environmental diversity.

## Output (fields added to each annotation)

```json
{
  "core:sample_start": 123456,
  "core:freq_lower_edge": 2.406e9,
  "core:freq_upper_edge": 2.416e9,
  "core:label": "drone_c2",
  "deepsig:confidence": 0.93,
  "deepsig:snr_estimate": 22.1,
  "sei:verdict": "own",
  "sei:hybrid_score": -0.42,
  "sei:ae_score": -0.31,
  "sei:ocsvm_score": -0.58,
  "sei:threshold": 1.64,
  "sei:n_windows": 8,
  "sei:snr_db": 22.1
}
```

`sei:verdict` is one of `own`, `foreign`, `unknown` (SNR below gate /
empty / read error), or `skipped` (label filter). A top-level
`md['global']['sei:model_threshold']` is also added for traceability.

## Tuning knobs

* `--min-snr-db` (default 15): below this, annotations are labeled
  `unknown` instead of scored. Uses `deepsig:snr_estimate` from OmniSIG
  when present, falls back to an in-band/out-of-band power estimate.
* `--false-reject-target` (default 0.05, training time): allowable
  fraction of own-drone bursts the calibrated threshold misses. Lower is
  stricter.
* `--target-rate` (default 10e6): matches the default in your
  `resample_cli.grc`. All three pieces agree.
* `--window-len` (default 1024): autoencoder input window.
* `--max-windows-per-soi` (default 16 infer / 8 train): cap on windows
  scored per annotation to keep latency bounded.
* `--oversample` (default 4×, training-time channelization): bandwidth
  kept around the SOI. Keeps the spectral regrowth and image-frequency
  leakage that carry the IQ-imbalance / PA-nonlinearity fingerprint.

## ZMQ integration

`gnuradio_watcher.py --enable-zmq --port 9200` publishes each enriched
meta dict as JSON on `tcp://*:9200`. Example subscriber:

```python
import zmq
ctx = zmq.Context(); s = ctx.socket(zmq.SUB)
s.connect("tcp://localhost:9200"); s.setsockopt_string(zmq.SUBSCRIBE, "")
while True:
    md = s.recv_json()
    for a in md.get("annotations", []):
        if a.get("sei:verdict") == "foreign":
            print("FOREIGN:", a["core:label"], a["core:freq_lower_edge"])
```

This is a different port from `pylibos_watch_dir.py`'s default (9100), so
the two publishers can run side by side.

## Known limitations

* The blind IQ-imbalance estimator in `features.py` assumes roughly
  circular baseband. For strongly non-circular modulations (BPSK, strong
  AM) it biases; swap for data-aided estimation if that matters on your
  waveform.
* `gnuradio_watcher.py` uses the same `on_created` + 1-second-sleep
  pattern as `pylibos_watch_dir.py` to resolve the meta/data write race;
  same caveats apply on fast recorders.
* No TensorRT optimization. For Air-T Tegra latency, convert
  `autoencoder.pt` with `torch2trt` or TensorRT directly — the tensor
  layout `(B, N, 2)` already matches your `numpy_to_tensor`.
