#!/usr/bin/python3
"""CUDA stream parallelism patch for gnuradio_watcher.py — dual-model mode.

In dual-model mode (--model-ul + --model-dl), UL and DL annotations in a
single capture are independent: they come from different links, use different
SEI models, and produce separate sei:* fields. The original implementation
processes them sequentially in a for-loop. This patch runs UL and DL
inference concurrently on separate CUDA streams using ThreadPoolExecutor.

Because PyTorch CUDA streams are thread-local, each thread gets its own
stream automatically when it calls torch.cuda.current_stream(). We make
this explicit with torch.cuda.stream() context managers.

APPLY THIS PATCH
----------------
Replace the `run_pipeline_on_pair` function in gnuradio_watcher.py with the
version below. Also add the two imports at the top of the file:

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

HOW IT WORKS
------------
1. For each capture, annotations are partitioned into UL and DL lists.
2. A ThreadPoolExecutor submits one task per link (UL, DL).
3. Each task runs on its own CUDA stream via torch.cuda.stream().
4. torch.cuda.synchronize() joins both streams before writing metadata.
5. Single-model mode is unchanged (no executor overhead).

THREAD SAFETY NOTE
------------------
- classify_baseband() and extract_features_batch() are stateless: no shared
  mutable state, only model.forward() calls.
- SEIModel._ae_error() uses @torch.no_grad() and separate CUDA streams, so
  concurrent calls are safe.
- The OC-SVM (sklearn) decision_function() is GIL-bound but CPU-only and fast.
  The GIL serialises it; the GPU work overlaps freely.
- gnuradio_watcher.py's file I/O (annotation_iq, write_json_to_file) is
  per-annotation and has no shared state.
"""

# ============================================================
# ADD THESE TWO IMPORTS TO THE TOP OF gnuradio_watcher.py
# ============================================================
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import threading

# ============================================================
# REPLACE run_pipeline_on_pair WITH THIS VERSION
# ============================================================

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import torch


def run_pipeline_on_pair_cuda_streams(dslog, datafile, args, sei_set, device,
                                      zmq_socket, profiles):
    """Per-file pipeline with concurrent UL/DL inference via CUDA streams.

    Signature identical to the original run_pipeline_on_pair so it is a
    drop-in replacement. Single-model mode falls through to sequential logic.
    """
    import os, json, shutil
    from pathlib import Path
    from sigmf_utils import dsprint, annotation_iq, channelize_inmem
    from compute_channel_params import compute_from_meta
    from infer import classify_baseband
    from gates import (ACCEPT, REJECT, ABSTAIN,
                       build_standard_pipeline, load_profiles, get_profile)

    try:
        from pylibos_common import get_meta, write_json_to_file
    except Exception:
        def get_meta(filename): return json.load(open(filename))
        def write_json_to_file(filename, obj):
            with open(filename, 'w') as f: json.dump(obj, f, indent=2)

    metafile = datafile.replace("sigmf-data", "sigmf-meta")
    if not (os.path.isfile(datafile) and os.path.isfile(metafile)):
        dslog.error(f"Missing pair: {datafile} / {metafile}")
        return

    md = get_meta(metafile)
    params = compute_from_meta(md)
    dslog.info(f"[Pipeline] {os.path.basename(datafile)}: {len(params)} annotation(s)")

    dual_mode = "__single__" not in sei_set
    counts = {"own": 0, "foreign": 0, "unknown": 0, "skipped": 0}

    # ------------------------------------------------------------------
    # Helper: process one annotation (same logic as original, plus stream)
    # ------------------------------------------------------------------
    def _process_one(anno, p, cuda_stream=None):
        """Run channelize + gate + SEI for a single annotation.

        If cuda_stream is not None, wraps the GPU inference in that stream.
        """
        label = p.get("label", "")

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

        raw = annotation_iq(datafile, md, anno)
        if len(raw) == 0:
            out = {"sei:verdict": "unknown", "sei:reason": "empty annotation",
                   "sei:label": label}
            if sei_link is not None: out["sei:link"] = sei_link
            return out

        baseband, new_sr = channelize_inmem(raw, p["samp_rate"], p["offset"],
                                            p["bw"], args.oversample)

        profile  = get_profile(profiles, label)
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
            if sei_link is not None: out["sei:link"] = sei_link
            return out

        # GPU inference — wrapped in caller-supplied CUDA stream if provided
        if cuda_stream is not None and device == "cuda":
            with torch.cuda.stream(cuda_stream):
                result = classify_baseband(baseband, new_sr, sei,
                                           min_snr_db=args.min_snr_db,
                                           max_windows=args.max_windows,
                                           device=device,
                                           snr_meta=p.get("snr_estimate"))
        else:
            result = classify_baseband(baseband, new_sr, sei,
                                       min_snr_db=args.min_snr_db,
                                       max_windows=args.max_windows,
                                       device=device,
                                       snr_meta=p.get("snr_estimate"))

        result["sei:label"]        = label
        result["sei:gate_summary"] = outcome.summary()
        if outcome.verdict == ABSTAIN:
            result["sei:gate_abstained"] = True
        if sei_link is not None:
            result["sei:link"] = sei_link
        return result

    # ------------------------------------------------------------------
    # Dual-model: partition annotations by link, run UL + DL concurrently
    # ------------------------------------------------------------------
    if dual_mode and device == "cuda":
        # Pre-allocate two streams for the lifetime of this capture
        stream_ul = torch.cuda.Stream()
        stream_dl = torch.cuda.Stream()

        # Map label -> stream
        stream_map = {
            args.label_ul: stream_ul,
            args.label_dl: stream_dl,
        }

        anno_param_pairs = list(zip(md.get('annotations', []), params))

        # Submit all annotations; each gets the right stream for its link
        futures_to_idx = {}
        with ThreadPoolExecutor(max_workers=2) as pool:
            for idx, (anno, p) in enumerate(anno_param_pairs):
                label = p.get("label", "")
                stream = stream_map.get(label)   # None -> sequential fallback
                future = pool.submit(_process_one, anno, p, stream)
                futures_to_idx[future] = idx

            # Collect in submission order
            results_by_idx = {}
            for future in as_completed(futures_to_idx):
                idx = futures_to_idx[future]
                try:
                    results_by_idx[idx] = future.result()
                except Exception as exc:
                    dslog.error(f"Annotation {idx} raised: {exc}")
                    results_by_idx[idx] = {"sei:verdict": "unknown",
                                           "sei:reason": f"exception: {exc}"}

        # Wait for both streams before touching metadata
        torch.cuda.synchronize()

        enriched = []
        for idx, (anno, p) in enumerate(anno_param_pairs):
            sei_fields = results_by_idx.get(idx, {"sei:verdict": "unknown"})
            new_anno = dict(anno)
            new_anno.update(sei_fields)
            enriched.append(new_anno)
            v = sei_fields.get("sei:verdict", "unknown")
            counts[v] = counts.get(v, 0) + 1

    else:
        # Single-model or CPU: original sequential path
        enriched = []
        for anno, p in zip(md.get('annotations', []), params):
            sei_fields = _process_one(anno, p, cuda_stream=None)
            new_anno = dict(anno)
            new_anno.update(sei_fields)
            enriched.append(new_anno)
            v = sei_fields.get("sei:verdict", "unknown")
            counts[v] = counts.get(v, 0) + 1

    md['annotations'] = enriched

    # ------------------------------------------------------------------
    # Capture-level verdict fusion (--fuse-capture flag)
    # ------------------------------------------------------------------
    if args.fuse_capture:
        verdicts = [a.get("sei:verdict") for a in enriched]
        if "foreign" in verdicts:
            capture_verdict = "foreign"
        elif "own" in verdicts:
            capture_verdict = "own"
        else:
            capture_verdict = "unknown"
        md['global']['sei:capture_verdict'] = capture_verdict
        dslog.info(f"[Fusion] capture_verdict={capture_verdict}")

    dslog.info(f"[Counts] own={counts['own']} foreign={counts['foreign']} "
               f"unknown={counts['unknown']} skipped={counts['skipped']}")

    # ------------------------------------------------------------------
    # Write enriched metadata and move/delete originals
    # ------------------------------------------------------------------
    out_dir = Path(args.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_meta = str(out_dir / os.path.basename(metafile))
    out_data = str(out_dir / os.path.basename(datafile))

    write_json_to_file(out_meta, md)

    if args.delete_originals:
        os.remove(datafile)
        os.remove(metafile)
    else:
        shutil.copy2(datafile, out_data)

    if zmq_socket is not None:
        try:
            zmq_socket.send_json(md, flags=zmq.NOBLOCK)
        except Exception:
            pass

    dslog.info(f"[Done] {os.path.basename(metafile)} -> {out_meta}")


# ============================================================
# ALSO PATCH: model loading in main() — use load_trt() if available
# ============================================================
# Replace:
#     sei_ul = SEIModel.load(args.model_ul)
#     sei_dl = SEIModel.load(args.model_dl)
# With:
#     _load = getattr(SEIModel, 'load_trt', SEIModel.load)
#     sei_ul = _load(args.model_ul)
#     sei_dl = _load(args.model_dl)
#
# This uses TensorRT engines if present, PyTorch otherwise.
# ============================================================
