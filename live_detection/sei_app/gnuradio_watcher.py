#!/usr/bin/python3
"""Stage 4 orchestrator: Compute Channel Params + in-process channelize + SEI App + Delete.

This is the piece missing from your architecture diagram between
pylibos_watch_dir.py (Stage 2) and whatever consumes final SEI verdicts.

It watches the directory that pylibos_watch_dir.py drops annotated SigMF
pairs into (--output-path of that stage), and for every new pair:

    1. Computes per-annotation channel params   (compute_channel_params.py)
    2. Runs the pre-ML gate pipeline             (gates.py)
    3. Channelizes the burst in-process          (sigmf_utils.channelize_inmem)
    4. Runs the SEI App on the channelized output (infer.classify_baseband)
    5. Writes sei:* fields back into the .sigmf-meta
    6. Moves the enriched pair to --output-path   (or deletes, per flag)
    7. Optionally publishes the enriched meta over ZMQ PUB

NOTE: Earlier versions shelled out to resample_cli.py (a GNURadio flowgraph).
That's been replaced with the same in-process channelize_inmem call that
train.py and infer.py use. Benefits: no subprocess launch overhead, no temp
files, no dependency on the resample_cli.grc -> .py regeneration step, and
the channelization path is now identical to training.

Model modes:

    * Single-model:  --model <dir>   [+ optional --label-filter]
    * Dual-model:    --model-ul <dir> --model-dl <dir>
                     [+ --label-ul / --label-dl to override default label strings]

    With --fuse-capture, a single capture-level verdict is written to
    md['global']['sei:capture_verdict'] = "foreign" if any annotation in
    the capture was foreign, else "own" if any own, else "unknown".

Usage (single-model):
    python gnuradio_watcher.py \
        --watchdir    /sdr/omnisig_out \
        --output-path /sdr/sei_out \
        --model       ./sei_model \
        --target-rate 10e6 \
        --profiles    ./emitter_profiles.json \
        --delete-originals \
        --enable-zmq --port 9200

Usage (dual-model + fusion):
    python gnuradio_watcher.py \
        --watchdir     /sdr/omnisig_out \
        --output-path  /sdr/sei_out \
        --model-ul     ./sei_model_uplink \
        --model-dl     ./sei_model_downlink \
        --label-ul n11_pro_UL --label-dl n11_pro_DL \
        --target-rate 10e6 --fuse-capture \
        --profiles ./emitter_profiles.json \
        --delete-originals --enable-zmq --port 9200
"""
import argparse
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from queue import Queue, Empty

import numpy as np
import torch
import zmq
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from sigmf_utils import dsprint, annotation_iq, channelize_inmem
from compute_channel_params import compute_from_meta
from infer import classify_baseband
from sei_model import SEIModel
from gates import (
    ACCEPT, REJECT, ABSTAIN,
    build_standard_pipeline, load_profiles, get_profile,
)

try:
    from dslogger import Logger
    _have_dslogger = True
except Exception:
    import logging
    _have_dslogger = False

try:
    from pylibos_common import get_meta, write_json_to_file
except Exception:
    def get_meta(filename):
        return json.load(open(filename))
    def write_json_to_file(filename, obj):
        with open(filename, 'w') as f:
            json.dump(obj, f, indent=2)


# --------------------------------------------------------------------------- #
# Args
# --------------------------------------------------------------------------- #

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--watchdir', required=True,
                   help="Directory to watch (pylibos_watch_dir.py's --output-path)")
    p.add_argument('--output-path', required=True,
                   help="Where to move enriched SigMF pairs after SEI runs")

    # Model selection --
    #   (a) --model <dir>                           : single model, scores all matching labels
    #   (b) --model-ul + --model-dl                 : dual-link routing by label
    p.add_argument('--model', default=None,
                   help="Trained SEI model dir (single-model mode)")
    p.add_argument('--model-ul', default=None,
                   help="UL SEI model dir (dual-model mode; requires --model-dl)")
    p.add_argument('--model-dl', default=None,
                   help="DL SEI model dir (dual-model mode; requires --model-ul)")
    p.add_argument('--label-ul', default="n11_pro_UL",
                   help="OmniSIG core:label that maps to the UL model (default n11_pro_UL)")
    p.add_argument('--label-dl', default="n11_pro_DL",
                   help="OmniSIG core:label that maps to the DL model (default n11_pro_DL)")
    p.add_argument('--model-map', nargs='+', default=None,
                   metavar='LABEL:MODEL_DIR',
                   help="Multi-drone routing: one or more LABEL:MODEL_DIR pairs. "
                        "e.g. nafyre_n11_pro:./sei_model_dl ruko_f11_mini:./sei_model_dl. "
                        "A model dir is only loaded once even if multiple labels share it.")

    p.add_argument('--target-rate', type=float, default=10e6,
                   help="Output sample rate of the in-process channelizer (default 10e6, "
                        "matches the training-time target rate)")
    p.add_argument('--oversample', type=float, default=4.0,
                   help="Channelizer oversample factor relative to annotation bw (default 4.0)")
    p.add_argument('--label-filter', default=None,
                   help="Single-model mode only: score only annotations with this core:label")
    p.add_argument('--fuse-capture', action='store_true',
                   help="Add md['global']['sei:capture_verdict'] with a per-capture any-foreign fusion")
    p.add_argument('--min-snr-db', type=float, default=15.0)
    p.add_argument('--max-windows', type=int, default=16)
    p.add_argument('--delete-originals', action='store_true',
                   help="Delete the original SigMF pair after processing (matches the "
                        "'Delete SigMF files' box in your diagram)")

    # Pre-ML gate pipeline (mirrors infer.py / train.py)
    p.add_argument('--profiles', default=None,
                   help="emitter_profiles.json. Enables freq/bw/duration gates when set.")
    p.add_argument('--min-power-dbfs', type=float, default=None,
                   help="Raw-IQ power floor (dBFS). If None, power_gate abstains.")
    p.add_argument('--no-wideband-snr', action='store_true',
                   help="Disable the wideband (in-band vs out-of-band) SNR gate.")
    p.add_argument('--allow-blind-snr', action='store_true',
                   help="Add blind SNR as a last-resort gate (unreliable on baseband).")

    p.add_argument('--enable-zmq', action='store_true')
    p.add_argument('--port', default='9200')
    p.add_argument('--log-level', default='INFO',
                   choices=['debug','info','warning','error','critical',
                            'DEBUG','INFO','WARNING','ERROR','CRITICAL'])
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--write-log', action='store_true')
    p.add_argument('--device', default=None)
    return p.parse_args()


def build_logger(args):
    if _have_dslogger:
        return Logger(args.verbose, args.log_level.upper(), args.write_log).get_logger()
    import logging
    logger = logging.getLogger("sei_pipeline")
    logger.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(h)
    return logger


# --------------------------------------------------------------------------- #
# Stage 4 core: channelize one annotation in-process, gate, then SEI
# --------------------------------------------------------------------------- #

def process_annotation(dslog, datafile, md, anno, channel_params, args, sei_set,
                        device, profiles):
    """Channelize one SOI in-process, run gates + SEI model, return sei:* fields."""
    p = channel_params
    label = p.get("label", "")

    # Route annotation to the right SEI model
    if "__single__" in sei_set:
        sei = sei_set["__single__"]
        if args.label_filter and label != args.label_filter:
            return {"sei:verdict": "skipped", "sei:reason": "label filter",
                    "sei:label": label}
        sei_link = None
    else:
        sei = sei_set.get(label)
        if sei is None:
            return {"sei:verdict": "skipped",
                    "sei:reason": f"no model for label '{label}'",
                    "sei:label": label}
        sei_link = label

    # Pull the raw (pre-channelization) IQ once -- the gate pipeline wants the
    # wideband burst for freq_band / wideband_snr / power gates.
    raw = annotation_iq(datafile, md, anno)
    if len(raw) == 0:
        out = {"sei:verdict": "unknown", "sei:reason": "empty annotation",
               "sei:label": label}
        if sei_link is not None:
            out["sei:link"] = sei_link
        return out

    # In-process channelize. Replaces the old resample_cli.py subprocess call --
    # same math, no fork/exec, no temp files, matches train.py and infer.py.
    baseband, new_sr = channelize_inmem(raw, p["samp_rate"], p["offset"],
                                        p["bw"], args.oversample)

    # Pre-ML gate pipeline. Any REJECT -> emit 'foreign' with the gate reason
    # and skip the ML. Any ACCEPT (or all ABSTAIN) -> fall through to SEI.
    profile = get_profile(profiles, label)
    pipeline = build_standard_pipeline(
        p, wideband_iq=raw, baseband_iq=baseband,
        profile=profile, min_snr_db=args.min_snr_db,
        min_power_dbfs=args.min_power_dbfs,
        use_wideband_snr=(not args.no_wideband_snr),
        use_blind_snr_fallback=args.allow_blind_snr,
    )
    outcome = pipeline.run()
    if outcome.verdict == REJECT:
        out = {"sei:verdict": "foreign",
               "sei:reason": "pre-ML gate: " + "; ".join(outcome.reasons),
               "sei:gate_summary": outcome.summary(),
               "sei:label": label}
        if sei_link is not None:
            out["sei:link"] = sei_link
        return out

    # Run the hybrid AE + OCSVM classifier on the channelized baseband.
    result = classify_baseband(baseband, new_sr, sei,
                               min_snr_db=args.min_snr_db,
                               max_windows=args.max_windows,
                               device=device, snr_meta=p.get("snr_estimate"))
    result["sei:label"] = label
    result["sei:gate_summary"] = outcome.summary()
    if outcome.verdict == ABSTAIN:
        result["sei:gate_abstained"] = True
    if sei_link is not None:
        result["sei:link"] = sei_link
    return result


# --------------------------------------------------------------------------- #
# Per-file pipeline
# --------------------------------------------------------------------------- #

def run_pipeline_on_pair(dslog, datafile, args, sei_set, device, zmq_socket, profiles):
    metafile = datafile.replace("sigmf-data", "sigmf-meta")
    if not (os.path.isfile(datafile) and os.path.isfile(metafile)):
        dslog.error(f"Missing pair: {datafile} / {metafile}")
        return

    md = get_meta(metafile)
    params = compute_from_meta(md)
    dslog.info(f"[Pipeline] {os.path.basename(datafile)}: {len(params)} annotation(s)")

    counts = {"own": 0, "foreign": 0, "unknown": 0, "skipped": 0}

    enriched = []
    for anno, p in zip(md.get('annotations', []), params):
        sei_fields = process_annotation(dslog, datafile, md, anno, p, args,
                                          sei_set, device, profiles)
        new_anno = dict(anno); new_anno.update(sei_fields)
        enriched.append(new_anno)
        v = sei_fields.get("sei:verdict", "unknown")
        counts[v] = counts.get(v, 0) + 1

    md['annotations'] = enriched
    g = md.setdefault('global', {})
    if "__single__" in sei_set:
        g['sei:model_threshold'] = float(sei_set["__single__"].threshold)
    else:
        g['sei:model_thresholds'] = {lab: float(m.threshold) for lab, m in sei_set.items()}

    if args.fuse_capture:
        vs = [a.get("sei:verdict") for a in enriched]
        if "foreign" in vs:
            g['sei:capture_verdict'] = "foreign"
        elif "own" in vs:
            g['sei:capture_verdict'] = "own"
        else:
            g['sei:capture_verdict'] = "unknown"
        dslog.info(f"[Pipeline] capture verdict: {g['sei:capture_verdict']}")

    dslog.info(f"[Pipeline] verdicts: {counts}")

    # Relocate or delete per --delete-originals
    if args.delete_originals:
        dslog.info(f"[Pipeline] Deleting originals: {datafile}, {metafile}")
        try:
            os.remove(datafile); os.remove(metafile)
        except OSError as e:
            dslog.error(f"delete failed: {e}")
    else:
        os.makedirs(args.output_path, exist_ok=True)
        new_data = os.path.join(args.output_path, os.path.basename(datafile))
        new_meta = os.path.join(args.output_path, os.path.basename(metafile))
        dslog.info(f"[Pipeline] Moving pair -> {args.output_path}")
        shutil.move(datafile, new_data)
        write_json_to_file(new_meta, md)
        try: os.remove(metafile)
        except OSError: pass

    if zmq_socket is not None:
        try:
            zmq_socket.send_json(md); time.sleep(0.1)
        except Exception as e:
            dslog.error(f"ZMQ publish failed: {e}")


# --------------------------------------------------------------------------- #
# Watcher (same pending-pair pattern as pylibos_watch_dir.py)
# --------------------------------------------------------------------------- #

class SigMFFileHandler(FileSystemEventHandler):
    def __init__(self, dslog, job_queue):
        super().__init__()
        self.dslog = dslog
        self.job_queue = job_queue
        self.pending = {}

    def on_created(self, event):
        if event.is_directory:
            return
        fp = Path(event.src_path)
        if fp.suffix not in (".sigmf-meta", ".sigmf-data"):
            return
        prefix = fp.with_suffix('')
        slot = self.pending.setdefault(prefix, {"meta": None, "data": None})
        slot["meta" if fp.suffix == ".sigmf-meta" else "data"] = fp
        if slot["meta"] is not None and slot["data"] is not None:
            self.dslog.info(f"[Watcher] pair complete: {fp.stem}")
            time.sleep(1)  # race-condition guard (matches pylibos_watch_dir)
            self.job_queue.put(str(slot["data"]))
            del self.pending[prefix]


def watcher_loop(dslog, queue, stop_event, args):
    h = SigMFFileHandler(dslog, queue)
    obs = Observer(); obs.schedule(h, args.watchdir, recursive=False); obs.start()
    dslog.info(f"[Watcher] watching {args.watchdir}")
    try:
        while not stop_event.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    obs.stop(); obs.join()


def worker_loop(dslog, queue, stop_event, args, sei_set, device, zmq_socket, profiles):
    dslog.info("[Worker] ready")
    while not stop_event.is_set():
        try:
            datafile = queue.get(timeout=0.5)
        except Empty:
            continue
        try:
            run_pipeline_on_pair(dslog, datafile, args, sei_set, device, zmq_socket, profiles)
        except Exception as e:
            dslog.error(f"[Worker] pipeline error for {datafile}: {e}")
        queue.task_done()
    dslog.info("[Worker] stopped")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    args = get_args()
    dslog = build_logger(args)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve model selection
    using_map  = bool(args.model_map)
    dual       = bool(args.model_ul or args.model_dl)

    n_modes = sum([using_map, dual, bool(args.model)])
    if n_modes == 0:
        dslog.error("Need --model, --model-ul/--model-dl, or --model-map")
        sys.exit(2)
    if n_modes > 1:
        dslog.error("--model, --model-ul/--model-dl, and --model-map are mutually exclusive")
        sys.exit(2)
    if dual and not (args.model_ul and args.model_dl):
        dslog.error("Dual-model mode requires BOTH --model-ul and --model-dl")
        sys.exit(2)

    sei_set = {}
    if using_map:
        # Multi-drone mode: any number of LABEL:MODEL_DIR pairs.
        # Models are cached by directory path so a shared model (e.g. one SEI
        # model covering all drone DL links) is only loaded once.
        _model_cache = {}
        for entry in args.model_map:
            if ":" not in entry:
                dslog.error(f"--model-map entry must be LABEL:MODEL_DIR, got: {entry!r}")
                sys.exit(2)
            label, model_dir = entry.split(":", 1)
            if model_dir not in _model_cache:
                dslog.info(f"Loading model from {model_dir}")
                _model_cache[model_dir] = SEIModel.load(model_dir)
                m = _model_cache[model_dir]
                dslog.info(f"  threshold={m.threshold:.3f}  input_len={m.config.input_length}")
            sei_set[label] = _model_cache[model_dir]
            dslog.info(f"  Mapped label '{label}' -> {model_dir}")
        dslog.info(f"Multi-drone routing active: {list(sei_set.keys())}")

    elif dual:
        dslog.info(f"Loading UL model from {args.model_ul}  (label='{args.label_ul}')")
        sei_ul = SEIModel.load(args.model_ul)
        dslog.info(f"  UL threshold={sei_ul.threshold:.3f}  input_len={sei_ul.config.input_length}")
        dslog.info(f"Loading DL model from {args.model_dl}  (label='{args.label_dl}')")
        sei_dl = SEIModel.load(args.model_dl)
        dslog.info(f"  DL threshold={sei_dl.threshold:.3f}  input_len={sei_dl.config.input_length}")
        sei_set = {args.label_ul: sei_ul, args.label_dl: sei_dl}

    else:
        dslog.info(f"Loading SEI model from {args.model}")
        sei = SEIModel.load(args.model)
        dslog.info(f"  threshold={sei.threshold:.3f}  input_len={sei.config.input_length}")
        sei_set = {"__single__": sei}

    # Load emitter profiles for the gate pipeline (abstains if --profiles omitted).
    profiles = load_profiles(args.profiles)
    if args.profiles:
        dslog.info(f"Loaded profiles from {args.profiles}: "
                   f"{[k for k in profiles if not k.startswith('_')]}")

    zmq_socket = None
    if args.enable_zmq:
        ctx = zmq.Context(); zmq_socket = ctx.socket(zmq.PUB)
        zmq_socket.bind(f"tcp://*:{args.port}")
        dslog.info(f"ZMQ PUB on tcp://*:{args.port}")

    os.makedirs(args.output_path, exist_ok=True)
    queue = Queue()
    stop_event = threading.Event()

    wt = threading.Thread(target=watcher_loop, args=(dslog, queue, stop_event, args),
                          daemon=True, name="watcher")
    bt = threading.Thread(target=worker_loop,
                          args=(dslog, queue, stop_event, args, sei_set, device,
                                zmq_socket, profiles),
                          daemon=True, name="worker")
    wt.start(); bt.start()

    dslog.info("[Main] Ctrl+C to stop")
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        dslog.info("[Main] shutting down")
    stop_event.set(); wt.join(); bt.join()


if __name__ == "__main__":
    main()
