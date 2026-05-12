#!/usr/bin/python3
r"""Stage 3 — Compute Channel Params.

Takes an OmniSIG-annotated .sigmf-meta and, for every annotation, computes:

    BW     = core:freq_upper_edge - core:freq_lower_edge
    CF     = core:freq_lower_edge + BW/2
    Offset = CF - captures[0]['core:frequency']       (recorder center freq)

These are exactly the three variables consumed by resample_cli.py:
    --bw            <- BW
    --offset-freq   <- Offset
    --target-rate   <- a chosen post-channelize rate (default 10e6, same as the .grc)

Dual-use: importable library + CLI.

CLI output is one JSON object per annotation, one per line (JSONL) so a shell
loop can feed them straight into resample_cli.py:

    python compute_channel_params.py --metafile capture.sigmf-meta | \
        jq -r '"--bw \(.bw) --offset-freq \(.offset)"'
"""
import argparse
import json
import sys


def compute_from_annotation(anno, recorded_cf):
    """Return a dict of channel params for one SigMF annotation."""
    lf = float(anno['core:freq_lower_edge'])
    uf = float(anno['core:freq_upper_edge'])
    bw = uf - lf
    cf = lf + bw / 2.0
    offset = cf - float(recorded_cf)
    return {
        "sample_start": int(anno['core:sample_start']),
        "sample_count": int(anno.get('core:sample_count', 0)),
        "lower_freq": lf,
        "upper_freq": uf,
        "bw": bw,
        "cf": cf,
        "offset": offset,
        "label": anno.get('core:label', ''),
        # Authoritative SNR: prefer capture_details:SNRdB (OmniSIG populates this
        # when the 'capture_details' extension is configured), fall back to
        # deepsig:snr_estimate. IMPORTANT: use explicit None check, not `or`,
        # because a legitimate 0 dB reading would otherwise be discarded as falsy.
        "snr_estimate": (anno['capture_details:SNRdB']
                          if 'capture_details:SNRdB' in anno
                          else anno.get('deepsig:snr_estimate')),
        "confidence": anno.get('deepsig:confidence'),
    }


def compute_from_meta(md):
    """Compute channel params for every annotation in a SigMF meta dict."""
    recorded_cf = float(md['captures'][0]['core:frequency'])
    sample_rate = int(md['global']['core:sample_rate'])
    params = [compute_from_annotation(a, recorded_cf) for a in md.get('annotations', [])]
    # Attach the recorder sample rate for convenience so downstream (resample_cli)
    # has everything it needs from one dict.
    for p in params:
        p["samp_rate"] = sample_rate
        p["recorded_cf"] = recorded_cf
    return params


def compute_from_file(metafile):
    with open(metafile, 'r') as f:
        md = json.load(f)
    return compute_from_meta(md)


def main():
    ap = argparse.ArgumentParser(description="Stage 3 - Compute Channel Params")
    ap.add_argument('--metafile', required=True, help="Path to OmniSIG-annotated .sigmf-meta")
    ap.add_argument('--target-rate', type=float, default=10e6,
                    help="Desired output sample rate after channelization (default 10e6)")
    ap.add_argument('--label-filter', default=None,
                    help="Only emit annotations with this core:label")
    ap.add_argument('--min-snr-db', type=float, default=None,
                    help="Drop annotations whose deepsig:snr_estimate is below this")
    ap.add_argument('--pretty', action='store_true', help="Pretty-print JSON list instead of JSONL")
    args = ap.parse_args()

    params = compute_from_file(args.metafile)
    out = []
    for p in params:
        if args.label_filter and p["label"] != args.label_filter:
            continue
        if args.min_snr_db is not None and p["snr_estimate"] is not None \
                and p["snr_estimate"] < args.min_snr_db:
            continue
        p["target_rate"] = float(args.target_rate)
        out.append(p)

    if args.pretty:
        json.dump(out, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        for p in out:
            sys.stdout.write(json.dumps(p) + "\n")


if __name__ == "__main__":
    main()
