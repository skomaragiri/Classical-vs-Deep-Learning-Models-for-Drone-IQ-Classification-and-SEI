#!/usr/bin/python3
"""One-class SEI model.

CNN autoencoder on raw IQ + One-Class SVM on engineered features, combined
into a single hybrid anomaly score with a calibrated threshold.

Input tensor convention (matches pylibos_common.numpy_to_tensor):
    (batch, N, 2)  with last dim = [I, Q]

The autoencoder internally transposes to (batch, 2, N) for 1-D conv layers.
"""
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# 1-D CNN autoencoder
# --------------------------------------------------------------------------- #

class IQAutoencoder(nn.Module):
    """Autoencoder on (batch, N, 2) IQ tensors."""

    def __init__(self, input_length=1024):
        super().__init__()
        self.input_length = input_length

        self.encoder = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128), nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32), nn.GELU(),
            nn.ConvTranspose1d(32, 2, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        # x: (B, N, 2) -> (B, 2, N)
        x = x.transpose(1, 2)
        z = self.encoder(x)
        out = self.decoder(z)
        # back to (B, N, 2)
        return out.transpose(1, 2)


# --------------------------------------------------------------------------- #
# Augmentation for training-time robustness
# --------------------------------------------------------------------------- #

def augment_bn2(tensor_bn2, max_cfo_norm=1e-3, snr_db_range=(20.0, 40.0), random_phase=True):
    """Augment a (B, N, 2) float tensor with CFO, phase rotation, and AWGN.

    Operates on numpy internally for clarity; returns a new torch tensor.
    """
    arr = tensor_bn2.detach().cpu().numpy().copy()
    B, N, _ = arr.shape
    for b in range(B):
        I = arr[b, :, 0].astype(np.complex64) + 1j * arr[b, :, 1]
        if max_cfo_norm > 0:
            cfo = np.random.uniform(-max_cfo_norm, max_cfo_norm)
            I = I * np.exp(2j * np.pi * cfo * np.arange(N)).astype(np.complex64)
        if random_phase:
            I = I * np.exp(1j * np.random.uniform(-np.pi, np.pi)).astype(np.complex64)
        if snr_db_range is not None:
            snr_db = np.random.uniform(*snr_db_range)
            sig_pow = float(np.mean(np.abs(I) ** 2))
            noise_pow = sig_pow / (10 ** (snr_db / 10.0))
            noise = (np.random.randn(N) + 1j * np.random.randn(N)).astype(np.complex64)
            noise *= np.sqrt(noise_pow / 2.0)
            I = I + noise
        arr[b, :, 0] = I.real.astype(np.float32)
        arr[b, :, 1] = I.imag.astype(np.float32)
    return torch.from_numpy(arr)


# --------------------------------------------------------------------------- #
# Hybrid one-class SEI model
# --------------------------------------------------------------------------- #

@dataclass
class SEIConfig:
    input_length: int = 1024
    ocsvm_nu: float = 0.05
    ocsvm_gamma: str = "scale"
    false_reject_target: float = 0.05


class SEIModel:
    """Autoencoder + OC-SVM + scaler + calibrated threshold."""

    def __init__(self, config=None):
        self.config = config or SEIConfig()
        self.autoencoder = IQAutoencoder(self.config.input_length)
        self.ocsvm = None
        self.scaler = None
        self.ae_mean = 0.0
        self.ae_std = 1.0
        self.ocs_mean = 0.0
        self.ocs_std = 1.0
        self.threshold = 0.0

    # ---- individual scoring paths ---- #
    @torch.no_grad()
    def _ae_error(self, iq_bn2, device=None):
        # device arg kept for API compatibility; actual device set at load time.
        dev = getattr(self, '_infer_device', 'cpu')
        x = iq_bn2.to(dev, non_blocking=True)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        recon = self.autoencoder(x)
        err = ((recon - x) ** 2).mean(dim=[1, 2])
        return err.cpu().numpy()

    def _ocsvm_score(self, features):
        if self.ocsvm is None or self.scaler is None:
            raise RuntimeError("OC-SVM not fitted")
        if features.ndim == 1:
            features = features[None, :]
        Xs = self.scaler.transform(features)
        # decision_function: positive = inlier, flip so higher = more anomalous
        return -self.ocsvm.decision_function(Xs)

    # ---- hybrid scoring ---- #
    def score(self, iq_bn2, features, device="cpu"):
        ae = self._ae_error(iq_bn2, device=device)
        oc = self._ocsvm_score(features)
        ae_z = (ae - self.ae_mean) / (self.ae_std + 1e-9)
        oc_z = (oc - self.ocs_mean) / (self.ocs_std + 1e-9)
        hybrid = np.maximum(ae_z, oc_z)
        verdict = np.where(hybrid > self.threshold, "foreign", "own")
        return {
            "ae_error": ae,
            "ae_z": ae_z,
            "ocsvm_score": oc,
            "ocsvm_z": oc_z,
            "hybrid_score": hybrid,
            "verdict": verdict,
            "threshold": self.threshold,
        }

    # ---- persistence ---- #
    def save(self, out_dir):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.autoencoder.state_dict(), out_dir / "autoencoder.pt")
        import joblib
        joblib.dump(self.ocsvm, out_dir / "ocsvm.joblib")
        joblib.dump(self.scaler, out_dir / "scaler.joblib")
        meta = {
            "config": asdict(self.config),
            "ae_mean": self.ae_mean, "ae_std": self.ae_std,
            "ocs_mean": self.ocs_mean, "ocs_std": self.ocs_std,
            "threshold": self.threshold,
        }
        with open(out_dir / "model_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, model_dir):
        model_dir = Path(model_dir)
        with open(model_dir / "model_meta.json") as f:
            meta = json.load(f)
        cfg = SEIConfig(**meta["config"])
        obj = cls(cfg)
        obj.autoencoder.load_state_dict(
            torch.load(model_dir / "autoencoder.pt", map_location="cpu")
        )
        import joblib
        obj.ocsvm = joblib.load(model_dir / "ocsvm.joblib")
        obj.scaler = joblib.load(model_dir / "scaler.joblib")
        obj.ae_mean = meta["ae_mean"]; obj.ae_std = meta["ae_std"]
        obj.ocs_mean = meta["ocs_mean"]; obj.ocs_std = meta["ocs_std"]
        obj.threshold = meta["threshold"]
        # Pin autoencoder to GPU once at load time; stays there permanently.
        obj._infer_device = "cuda" if torch.cuda.is_available() else "cpu"
        obj.autoencoder = obj.autoencoder.to(obj._infer_device)
        obj.autoencoder.eval()
        return obj
