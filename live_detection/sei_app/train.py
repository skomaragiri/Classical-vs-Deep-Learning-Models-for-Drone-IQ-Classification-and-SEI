#!/usr/bin/python3
"""Offline training for the one-class SEI model.

Reads a directory of OmniSIG-annotated SigMF captures of YOUR drone, walks
every annotation, channelizes each one in-memory (fast; avoids subprocess-
per-annotation), extracts features, and fits:
    1. CNN autoencoder on raw baseband IQ
    2. One-Class SVM on engineered impairment features
    3. A calibrated hybrid anomaly threshold on held-out your-drone data

Usage:
    python train.py \
        --data-dir  /data/our_drone_sigmf \
        --out-dir   ./sei_model \
        --label-filter drone_c2 \
        --epochs 40

Data requirement: 5+ recording sessions on different days / locations.
Training on one session teaches the RF environment, not the emitter.
"""
import argparse
import glob
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sigmf_utils import (
    dsprint, annotation_iq, channelize_inmem,
    iq_window_to_bn2, slice_windows,
)
from compute_channel_params import compute_from_meta
from features import extract_features, estimate_snr_db, N_FEATURES
from sei_model import SEIConfig, SEIModel, augment_bn2
from gates import (
    ACCEPT, REJECT, ABSTAIN,
    build_standard_pipeline, load_profiles, get_profile,
)

try:
    from pylibos_common import get_meta
except Exception:
    import json
    def get_meta(filename):
        try:
            return json.loads(open(filename).read())
        except json.decoder.JSONDecodeError:
            return None


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True, help="Directory of annotated SigMF captures of your drone")
    p.add_argument('--out-dir', default='./sei_model')
    p.add_argument('--label-filter', default=None, help="Only use annotations with this core:label")
    p.add_argument('--min-snr-db', type=float, default=15.0,
                   help="SNR floor fed to meta_snr_gate / wideband_snr_gate. "
                        "Set to a very low value (e.g. -100) to disable SNR-based rejection.")
    p.add_argument('--allow-blind-snr', action='store_true',
                   help="Add the percentile-ratio blind SNR gate as a last-resort "
                        "fallback. Unreliable on channelized baseband; off by default.")
    p.add_argument('--profiles', default=None,
                   help="Path to emitter_profiles.json. If omitted, freq/bw/duration "
                        "gates abstain and only SNR/power gates run.")
    p.add_argument('--min-power-dbfs', type=float, default=None,
                   help="Raw-IQ power floor (dBFS). If None, power_gate abstains.")
    p.add_argument('--no-wideband-snr', action='store_true',
                   help="Disable the wideband_snr_gate (in-band vs out-of-band PSD).")
    p.add_argument('--admit-abstain', action='store_true', default=True,
                   help="If all gates abstain, still admit the sample. Default True "
                        "(matches trust-the-chamber when no gate has data).")
    p.add_argument('--strict', dest='admit_abstain', action='store_false',
                   help="Strict mode: reject samples where every gate abstains.")
    p.add_argument('--window-len', type=int, default=1024)
    p.add_argument('--max-windows-per-soi', type=int, default=8)
    p.add_argument('--oversample', type=float, default=4.0)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--ocsvm-nu', type=float, default=0.05)
    p.add_argument('--calib-frac', type=float, default=0.15)
    p.add_argument('--false-reject-target', type=float, default=0.05)
    p.add_argument('--device', default=None)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def build_dataset(data_dir, window_len, min_snr_db, max_windows, label_filter, oversample,
                  allow_blind_snr=False, profiles_path=None, min_power_dbfs=None,
                  use_wideband_snr=True, admit_abstain=True):
    """Walk OmniSIG captures and return (iq_windows, features).

    Admission is decided by the modular gate pipeline in gates.py:

        freq_band + duration + meta_snr + power + wideband_snr (+ blind_snr)

    Each gate returns ACCEPT / REJECT / ABSTAIN. The pipeline folds:
      - any REJECT  -> drop the sample
      - any ACCEPT  -> keep the sample
      - all ABSTAIN -> keep if `admit_abstain` (default), drop in --strict mode
    """
    meta_files = sorted(glob.glob(str(Path(data_dir) / "**" / "*.sigmf-meta"), recursive=True))
    if not meta_files:
        raise FileNotFoundError(f"No .sigmf-meta under {data_dir}")
    dsprint("INFO", f"Scanning {len(meta_files)} captures")

    profiles = load_profiles(profiles_path)
    if profiles_path:
        dsprint("INFO", f"Loaded profiles from {profiles_path}: "
                        f"{[k for k in profiles if not k.startswith('_')]}")

    iq_windows, feats = [], []
    gate_stats = {"accept": 0, "reject": 0, "abstain_admit": 0, "abstain_drop": 0,
                  "skip_label": 0, "skip_empty": 0}
    reject_reasons = {}   # gate name -> count

    for mf in meta_files:
        md = get_meta(mf)
        if md is None:
            continue
        datafile = mf.replace(".sigmf-meta", ".sigmf-data")
        if not os.path.isfile(datafile):
            continue

        params = compute_from_meta(md)
        used = 0
        for anno, p in zip(md.get('annotations', []), params):
            if label_filter and p["label"] != label_filter:
                gate_stats["skip_label"] += 1
                continue

            raw = annotation_iq(datafile, md, anno)
            if len(raw) == 0:
                gate_stats["skip_empty"] += 1
                continue

            # Channelize before blind SNR, but AFTER we've already given the
            # pre-channelization wideband IQ to freq_band / wideband_snr / power.
            baseband, new_sr = channelize_inmem(raw, p["samp_rate"], p["offset"],
                                                p["bw"], oversample)

            profile = get_profile(profiles, p.get("label"))
            pipeline = build_standard_pipeline(
                p,
                wideband_iq=raw,
                baseband_iq=baseband,
                profile=profile,
                min_snr_db=min_snr_db,
                min_power_dbfs=min_power_dbfs,
                use_wideband_snr=use_wideband_snr,
                use_blind_snr_fallback=allow_blind_snr,
            )
            outcome = pipeline.run()

            if outcome.verdict == REJECT:
                gate_stats["reject"] += 1
                for r in outcome.results:
                    if r.verdict == REJECT:
                        reject_reasons[r.name] = reject_reasons.get(r.name, 0) + 1
                continue
            if outcome.verdict == ABSTAIN:
                if not admit_abstain:
                    gate_stats["abstain_drop"] += 1
                    continue
                gate_stats["abstain_admit"] += 1
            else:
                gate_stats["accept"] += 1

            for w in slice_windows(baseband, window_len, window_len // 2, max_windows):
                iq_windows.append(iq_window_to_bn2(w, window_len))
                feats.append(extract_features(w, new_sr))
                used += 1
        dsprint("INFO", f"  {os.path.basename(mf)}: {used} windows")

    dsprint("INFO",
            f"Gate pipeline: accept={gate_stats['accept']} "
            f"abstain_admit={gate_stats['abstain_admit']} "
            f"abstain_drop={gate_stats['abstain_drop']} "
            f"reject={gate_stats['reject']} "
            f"skip_label={gate_stats['skip_label']} "
            f"skip_empty={gate_stats['skip_empty']}")
    if reject_reasons:
        dsprint("INFO", "Reject breakdown: " +
                ", ".join(f"{k}={v}" for k, v in sorted(reject_reasons.items(),
                                                        key=lambda kv: -kv[1])))

    if not iq_windows:
        raise RuntimeError("No usable windows. Check gate config / label filter / annotations.")
    return np.concatenate(iq_windows, axis=0), np.asarray(feats, dtype=np.float32)


class IQWindowDataset(Dataset):
    def __init__(self, iq, augment=True):
        self.iq = iq
        self.augment = augment

    def __len__(self):
        return len(self.iq)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.iq[idx]).unsqueeze(0)
        if self.augment:
            x = augment_bn2(x)
        return x.squeeze(0)


def train_autoencoder(model, train_iq, epochs, batch_size, lr, device):
    model.to(device).train()
    loader = DataLoader(IQWindowDataset(train_iq, augment=True),
                        batch_size=batch_size, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    for e in range(epochs):
        total, n = 0.0, 0
        for batch in loader:
            batch = batch.to(device)
            loss = F.mse_loss(model(batch), batch)
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.detach()) * batch.size(0); n += batch.size(0)
        sched.step()
        dsprint("INFO", f"  epoch {e+1:3d}/{epochs}  loss={total/max(n,1):.6f}")


def main():
    args = get_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.time()
    X_iq, X_feat = build_dataset(
        args.data_dir, args.window_len, args.min_snr_db,
        args.max_windows_per_soi, args.label_filter, args.oversample,
        allow_blind_snr=args.allow_blind_snr,
        profiles_path=args.profiles,
        min_power_dbfs=args.min_power_dbfs,
        use_wideband_snr=(not args.no_wideband_snr),
        admit_abstain=args.admit_abstain,
    )
    assert X_feat.shape[1] == N_FEATURES
    dsprint("INFO", f"Total windows: {len(X_iq)}  feat dim: {X_feat.shape[1]}")

    idx = np.random.permutation(len(X_iq))
    n_cal = max(32, int(args.calib_frac * len(X_iq)))
    cal_idx, train_idx = idx[:n_cal], idx[n_cal:]

    cfg = SEIConfig(input_length=args.window_len, ocsvm_nu=args.ocsvm_nu,
                    false_reject_target=args.false_reject_target)
    sei = SEIModel(cfg)
    sei._infer_device = device  # ensure _ae_error uses the correct device during calibration

    dsprint("INFO", "Training autoencoder...")
    train_autoencoder(sei.autoencoder, X_iq[train_idx], args.epochs, args.batch_size, args.lr, device)

    dsprint("INFO", "Fitting One-Class SVM...")
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import OneClassSVM
    sei.scaler = StandardScaler().fit(X_feat[train_idx])
    sei.ocsvm = OneClassSVM(kernel="rbf", nu=cfg.ocsvm_nu, gamma=cfg.ocsvm_gamma).fit(
        sei.scaler.transform(X_feat[train_idx])
    )

    dsprint("INFO", "Calibrating anomaly threshold...")
    cal_iq = torch.from_numpy(X_iq[cal_idx])
    ae = sei._ae_error(cal_iq, device=device)
    oc = sei._ocsvm_score(X_feat[cal_idx])
    sei.ae_mean, sei.ae_std = float(np.mean(ae)), float(np.std(ae) + 1e-9)
    sei.ocs_mean, sei.ocs_std = float(np.mean(oc)), float(np.std(oc) + 1e-9)
    hybrid = np.maximum((ae - sei.ae_mean) / sei.ae_std, (oc - sei.ocs_mean) / sei.ocs_std)
    sei.threshold = float(np.quantile(hybrid, 1 - cfg.false_reject_target))
    dsprint("INFO", f"  threshold={sei.threshold:.3f}  FRR target={cfg.false_reject_target:.2f}")

    sei.save(args.out_dir)
    dsprint("INFO", f"Saved to {Path(args.out_dir).resolve()}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
