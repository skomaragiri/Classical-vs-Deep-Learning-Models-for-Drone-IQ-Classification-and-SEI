#!/usr/bin/python3
"""Stage 4b - The SEI App.

Consumes the channelized output of resample_cli.py (raw complex64, headerless)
and emits a verdict: 'own' / 'foreign' / 'unknown'.

Two modes:

  1. Normal mode (pipeline): resample_cli.py has already channelized one SOI
     to a raw .sigmf-data file at --target-rate. The SEI app classifies it.

         python infer.py \
             --channelized-file /tmp/soi_0.sigmf-data \
             --target-rate 10e6 \
             --model ./sei_model

  2. Standalone mode: skip resample_cli and channelize in-process directly
     from an annotated SigMF pair. Useful for dev/testing.

         python infer.py \
             --annotated-datafile capture.sigmf-data \
             --model ./sei_model
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

from sigmf_utils import (
    dsprint, read_raw_complex64, annotation_iq,
    channelize_inmem, iq_window_to_bn2, slice_windows,
)
from compute_channel_params import compute_from_meta
from features import extract_features, extract_features_batch, estimate_snr_db
from sei_model import SEIModel
from gates import (
    ACCEPT, REJECT, ABSTAIN,
    build_standard_pipeline, load_profiles, get_profile,
)

try:
    from pylibos_common import get_meta, write_json_to_file
except Exception:
    def get_meta(filename):
        return json.load(open(filename))
    def write_json_to_file(filename, obj):
        with open(filename, 'w') as f:
            json.dump(obj, f, indent=2)


# --------------------------------------------------------------------------- #
# Core classifier: baseband IQ -> verdict dict
# --------------------------------------------------------------------------- #

def classify_baseband(iq, sample_rate, sei, min_snr_db=15.0, max_windows=16, device="cpu",
                      snr_meta=None):
    """Classify one channelized baseband burst. Returns a sei:* dict.

    If `snr_meta` is provided (e.g. OmniSIG's capture_details:SNRdB), it is used
    as the authoritative SNR. The blind estimator is only used as a fallback, since
    it is unreliable on channelized baseband (no quiet region for noise floor).
    """
    if len(iq) < sei.config.input_length // 2:
        return {"sei:verdict": "unknown", "sei:reason": "too few samples"}

    if snr_meta is not None:
        snr = float(snr_meta)
        snr_source = "meta"
    else:
        snr = float(estimate_snr_db(iq))
        snr_source = "blind"

    if snr < min_snr_db:
        return {"sei:verdict": "unknown",
                "sei:reason": f"snr {snr:.1f} < {min_snr_db} ({snr_source})",
                "sei:snr_db": snr,
                "sei:snr_source": snr_source}

    windows = list(slice_windows(iq, sei.config.input_length,
                                 stride=sei.config.input_length, max_windows=max_windows))
    if not windows:
        return {"sei:verdict": "unknown", "sei:reason": "no windows"}

    iq_bn2 = np.concatenate([iq_window_to_bn2(w, sei.config.input_length) for w in windows], axis=0)

    # Batch feature extraction on GPU (all windows in one vectorised pass)
    feats = extract_features_batch(windows, sample_rate, device=device)

    # Pinned-memory transfer: faster H2D on Orin NX unified memory
    iq_tensor = torch.from_numpy(iq_bn2)
    try:
        iq_tensor = iq_tensor.pin_memory()
    except RuntimeError:
        pass  # pin_memory unavailable in some contexts; fall back silently

    scored = sei.score(iq_tensor, feats, device=device)

    hybrid = scored["hybrid_score"]
    med = float(np.median(hybrid))
    return {
        "sei:verdict": "foreign" if med > sei.threshold else "own",
        "sei:hybrid_score": med,
        "sei:ae_score": float(np.median(scored["ae_z"])),
        "sei:ocsvm_score": float(np.median(scored["ocsvm_z"])),
        "sei:threshold": float(sei.threshold),
        "sei:n_windows": int(len(windows)),
        "sei:snr_db": float(snr),
        "sei:snr_source": snr_source,
    }


def classify_channelized_file(path, target_rate, sei, **kw):
    """Mode 1: classify a resample_cli.py output file (raw complex64)."""
    iq = read_raw_complex64(path)
    return classify_baseband(iq, target_rate, sei, **kw)


def classify_annotated_sigmf(datafile, metafile, sei, min_snr_db=15.0,
                             label_filter=None, oversample=4.0, device="cpu",
                             profiles_path=None, min_power_dbfs=None,
                             use_wideband_snr=True, use_blind_snr_fallback=False):
    """Mode 2: channelize in-process per annotation and classify each.

    Flow per annotation:
        1. Run the gate pipeline (freq/bw, duration, meta SNR, raw power,
           wideband SNR). If any gate REJECTS, emit 'foreign' with the reason
           and skip the ML -- this is the fast pre-filter the user asked for.
        2. Otherwise, run the hybrid AE + OCSVM classifier.
    """
    md = get_meta(metafile)
    params = compute_from_meta(md)
    profiles = load_profiles(profiles_path)
    results = []
    for anno, p in zip(md.get('annotations', []), params):
        if label_filter and p["label"] != label_filter:
            continue
        raw = annotation_iq(datafile, md, anno)
        if len(raw) == 0:
            results.append({"annotation": p, "sei": {"sei:verdict": "unknown",
                                                      "sei:reason": "empty"}})
            continue
        baseband, new_sr = channelize_inmem(raw, p["samp_rate"], p["offset"], p["bw"], oversample)

        # Pre-ML gate pipeline.
        profile = get_profile(profiles, p.get("label"))
        pipeline = build_standard_pipeline(
            p, wideband_iq=raw, baseband_iq=baseband,
            profile=profile, min_snr_db=min_snr_db, min_power_dbfs=min_power_dbfs,
            use_wideband_snr=use_wideband_snr,
            use_blind_snr_fallback=use_blind_snr_fallback,
        )
        outcome = pipeline.run()
        if outcome.verdict == REJECT:
            results.append({"annotation": p,
                            "sei": {"sei:verdict": "foreign",
                                    "sei:reason": "pre-ML gate: " + "; ".join(outcome.reasons),
                                    "sei:gate_summary": outcome.summary()}})
            continue

        verdict = classify_baseband(baseband, new_sr, sei, min_snr_db=min_snr_db,
                                     device=device, snr_meta=p.get("snr_estimate"))
        verdict["sei:gate_summary"] = outcome.summary()
        if outcome.verdict == ABSTAIN:
            verdict["sei:gate_abstained"] = True
        results.append({"annotation": p, "sei": verdict})
    return md, results


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def get_args():
    p = argparse.ArgumentParser(description="Stage 4b - Run SEI App")
    p.add_argument('--model', required=True, help="Trained SEI model directory")

    # Mode 1: classify resample_cli output
    p.add_argument('--channelized-file', default=None,
                   help="Raw complex64 file produced by resample_cli.py")
    p.add_argument('--target-rate', type=float, default=10e6,
                   help="Sample rate of --channelized-file (default 10e6)")

    # Mode 2: classify an annotated SigMF directly (standalone)
    p.add_argument('--annotated-datafile', default=None,
                   help="Path to .sigmf-data with OmniSIG annotations in its sidecar")
    p.add_argument('--label-filter', default=None)
    p.add_argument('--oversample', type=float, default=4.0)

    p.add_argument('--min-snr-db', type=float, default=15.0)
    p.add_argument('--max-windows', type=int, default=16)
    p.add_argument('--device', default=None)
    p.add_argument('--json-out', default=None, help="Write verdict(s) to this path instead of stdout")

    # Gate pipeline (applies to annotated mode; channelized-file mode has no
    # wideband reference so it still uses the simpler SNR-only path.)
    p.add_argument('--profiles', default=None,
                   help="emitter_profiles.json. Enables freq/bw/duration gates.")
    p.add_argument('--min-power-dbfs', type=float, default=None,
                   help="Raw-IQ power floor (dBFS). If None, power_gate abstains.")
    p.add_argument('--no-wideband-snr', action='store_true',
                   help="Disable the wideband (in-band vs out-of-band) SNR gate.")
    p.add_argument('--allow-blind-snr', action='store_true',
                   help="Add blind SNR as a last-resort gate (unreliable).")
    return p.parse_args()


def main():
    args = get_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dsprint("INFO", f"Loading SEI model from {args.model}")
    sei = SEIModel.load(args.model)

    if args.channelized_file:
        verdict = classify_channelized_file(
            args.channelized_file, args.target_rate, sei,
            min_snr_db=args.min_snr_db, max_windows=args.max_windows, device=device,
        )
        out = {"channelized_file": args.channelized_file, "result": verdict}
    elif args.annotated_datafile:
        # Accept either .sigmf-data or .sigmf-meta; derive the sibling path
        p = args.annotated_datafile
        if p.endswith(".sigmf-meta"):
            metafile = p
            datafile = p[:-len(".sigmf-meta")] + ".sigmf-data"
        elif p.endswith(".sigmf-data"):
            datafile = p
            metafile = p[:-len(".sigmf-data")] + ".sigmf-meta"
        else:
            datafile = p
            metafile = p.replace("sigmf-data", "sigmf-meta")
        if not os.path.isfile(datafile):
            print(f"Error: data file not found: {datafile}", file=sys.stderr)
            sys.exit(2)
        if not os.path.isfile(metafile):
            print(f"Error: meta file not found: {metafile}", file=sys.stderr)
            sys.exit(2)
        md, results = classify_annotated_sigmf(
            datafile, metafile, sei,
            min_snr_db=args.min_snr_db, label_filter=args.label_filter,
            oversample=args.oversample, device=device,
            profiles_path=args.profiles, min_power_dbfs=args.min_power_dbfs,
            use_wideband_snr=(not args.no_wideband_snr),
            use_blind_snr_fallback=args.allow_blind_snr,
        )
        out = {"datafile": datafile, "results": results}
    else:
        print("Error: need --channelized-file or --annotated-datafile", file=sys.stderr)
        sys.exit(2)

    if args.json_out:
        write_json_to_file(args.json_out, out)
        dsprint("INFO", f"Wrote {args.json_out}")
    else:
        json.dump(out, sys.stdout, indent=2)
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
