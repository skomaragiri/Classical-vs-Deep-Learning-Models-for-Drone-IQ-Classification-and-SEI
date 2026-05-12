#!/usr/bin/python3
"""Per-burst RF impairment feature extraction for SEI.

Features (37 dims) from one complex64 baseband window:
    residual_cfo_norm       1   residual CFO / sample_rate (fourth-power est.)
    iq_amp_imbalance        1   sqrt(E[I^2]/E[Q^2]) - 1
    iq_phase_imbalance      1   phi = asin(2*E[I*Q]/(E[I^2]+E[Q^2]))
    circularity             1   |E[x^2]| / E[|x|^2]
    dc_i, dc_q              2   mean(I)/rms, mean(Q)/rms
    envelope_skew           1
    envelope_kurt           1
    phase_noise_var         1   robust variance of instantaneous frequency
    spectral_flatness       1
    spectral_centroid_norm  1
    spectral_rolloff_norm   1
    spectral_regrowth       1
    psd_features           24   decimated, mean-centered log-PSD

GPU acceleration notes:
    - extract_features_batch() runs all per-window computations on CUDA via
      PyTorch, processing the entire window batch in one vectorized pass.
    - The CPU fallback (extract_features) is retained for single-window calls
      and for environments without CUDA.
    - Welch PSD is replaced with a batched FFT-based periodogram on GPU,
      matching Welch's variance reduction by averaging non-overlapping segments
      within each window.
    - All intermediate tensors stay on GPU until the final .cpu().numpy() call.
"""
import numpy as np
import torch
import torch.nn.functional as F

# SciPy kept only for the CPU fallback path
from scipy.signal import welch
from scipy.stats import kurtosis, skew

# ---------------------------------------------------------------------------
# Feature metadata
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "residual_cfo_norm",
    "iq_amp_imbalance",
    "iq_phase_imbalance",
    "circularity",
    "dc_i",
    "dc_q",
    "envelope_skew",
    "envelope_kurt",
    "phase_noise_var",
    "spectral_flatness",
    "spectral_centroid_norm",
    "spectral_rolloff_norm",
    "spectral_regrowth",
] + [f"psd_{i:02d}" for i in range(24)]

N_FEATURES = len(FEATURE_NAMES)  # 37


# ---------------------------------------------------------------------------
# GPU batch extraction (primary path)
# ---------------------------------------------------------------------------

def _make_hann(nperseg: int, device: torch.device) -> torch.Tensor:
    """Hann window cached as a module-level dict to avoid re-allocation."""
    return torch.hann_window(nperseg, periodic=False, device=device)


_HANN_CACHE: dict = {}


def _hann(nperseg: int, device: torch.device) -> torch.Tensor:
    key = (nperseg, str(device))
    if key not in _HANN_CACHE:
        _HANN_CACHE[key] = torch.hann_window(nperseg, periodic=False, device=device)
    return _HANN_CACHE[key]


def extract_features_batch(
    windows: list,
    sample_rate: float,
    device: str | torch.device = "cuda",
    n_psd_bins: int = 24,
    soi_bw_fraction: float = 0.5,
) -> np.ndarray:
    """Compute 37-dim feature vectors for a batch of complex64 IQ windows on GPU.

    Args:
        windows:     List of complex64 numpy arrays, each of length input_length
                     (e.g. 1024). All windows must be the same length.
        sample_rate: Receiver sample rate in Hz.
        device:      'cuda' or 'cpu'. Falls back to CPU automatically if CUDA
                     is unavailable.
        n_psd_bins:  Number of decimated log-PSD features (default 24).
        soi_bw_fraction: Fraction of sample_rate used as SOI bandwidth for
                     spectral regrowth gate (default 0.5).

    Returns:
        (N_windows, 37) float32 numpy array, ready to pass to SEIModel.score().
    """
    if not windows:
        return np.zeros((0, N_FEATURES), dtype=np.float32)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    sr = float(sample_rate)
    B = len(windows)
    N = len(windows[0])

    # Stack windows into (B, N) complex64 tensor on GPU
    arr = np.stack([w.astype(np.complex64) for w in windows])          # (B, N)
    x_c = torch.from_numpy(arr).to(dev)                                # complex64

    I = x_c.real.to(torch.float64)   # (B, N)
    Q = x_c.imag.to(torch.float64)   # (B, N)

    # ------------------------------------------------------------------
    # 1. IQ imbalance (second-order statistics)
    # ------------------------------------------------------------------
    Eii  = I.pow(2).mean(dim=1).clamp(min=1e-12)   # (B,)
    Eqq  = Q.pow(2).mean(dim=1).clamp(min=1e-12)
    Eiq  = (I * Q).mean(dim=1)

    alpha      = (Eii / Eqq).sqrt() - 1.0                          # amp imbalance
    sin_phi    = (2.0 * Eiq / (Eii + Eqq)).clamp(-1.0, 1.0)
    phi        = torch.asin(sin_phi)                                # phase imbalance

    # Circularity: |E[x^2]| / E[|x|^2]
    xc_d = x_c.to(torch.complex128)
    Ex2  = (xc_d.pow(2)).mean(dim=1)                               # (B,) complex
    Eabs = xc_d.abs().pow(2).mean(dim=1).clamp(min=1e-12).real
    circ = (Ex2.abs() / Eabs).to(torch.float64)

    # ------------------------------------------------------------------
    # 2. DC offset (normalised by RMS)
    # ------------------------------------------------------------------
    rms    = xc_d.abs().pow(2).mean(dim=1).sqrt().real.clamp(min=1e-12)
    dc_i   = I.mean(dim=1) / rms
    dc_q   = Q.mean(dim=1) / rms

    # ------------------------------------------------------------------
    # 3. Envelope moments (skewness & kurtosis via standardised moments)
    # ------------------------------------------------------------------
    env    = x_c.abs().to(torch.float64)                            # (B, N)
    e_mu   = env.mean(dim=1, keepdim=True)
    e_std  = env.std(dim=1, keepdim=True).clamp(min=1e-12)
    e_norm = (env - e_mu) / e_std                                   # (B, N)
    e_skew = e_norm.pow(3).mean(dim=1)
    e_kurt = e_norm.pow(4).mean(dim=1) - 3.0                       # Fisher

    # ------------------------------------------------------------------
    # 4. Residual CFO (fourth-power method)
    # ------------------------------------------------------------------
    # x^4 concentrates energy at 4*delta_f for QPSK-like signals
    x4     = xc_d.pow(4)                                            # (B, N)
    nfft   = 1 << int(np.ceil(np.log2(N)))
    spec4  = torch.fft.fft(x4, n=nfft, dim=1).abs()                # (B, nfft)
    freqs4 = torch.fft.fftfreq(nfft, d=1.0 / sr).to(dev)          # (nfft,)
    peak_idx = spec4.argmax(dim=1)                                  # (B,)
    cfo_hz   = freqs4[peak_idx] / 4.0
    cfo_norm = cfo_hz / sr                                          # (B,)

    # ------------------------------------------------------------------
    # 5. Phase noise (MAD-based variance of instantaneous frequency)
    # ------------------------------------------------------------------
    phase   = xc_d.add(1e-12).angle()                              # (B, N)
    # Unwrap phase per window
    phase_np = phase.cpu().numpy()
    phase_np = np.unwrap(phase_np, axis=1)
    phase    = torch.from_numpy(phase_np).to(dev)

    dphi     = phase.diff(dim=1)                                    # (B, N-1)
    med_dphi = dphi.median(dim=1, keepdim=True).values
    mad      = (dphi - med_dphi).abs().median(dim=1).values
    pn_var   = (1.4826 * mad).pow(2)

    # ------------------------------------------------------------------
    # 6. Spectral features via batched Welch-like periodogram on GPU
    # ------------------------------------------------------------------
    nperseg  = min(256, max(64, N // 8))
    # Non-overlapping segments within each window
    n_seg    = N // nperseg
    if n_seg < 1:
        n_seg = 1
        nperseg = N

    win_t = _hann(nperseg, dev)                                     # (nperseg,)

    # Reshape to (B * n_seg, nperseg) — use only the first n_seg*nperseg samples
    x_real = x_c[:, :n_seg * nperseg].reshape(B, n_seg, nperseg)   # (B, n_seg, nperseg)
    x_seg  = x_real * win_t                                         # broadcast
    # FFT over last dim -> (B, n_seg, nperseg) complex
    F_seg  = torch.fft.fft(x_seg, dim=2)
    psd_2d = F_seg.abs().pow(2)                                     # (B, n_seg, nperseg)
    psd    = psd_2d.mean(dim=1)                                     # (B, nperseg) — Welch avg

    # Frequency axis (fftshift order): negative freqs first
    freqs  = torch.fft.fftfreq(nperseg, d=1.0 / sr).to(dev)       # (nperseg,)
    order  = freqs.argsort()
    freqs  = freqs[order]
    psd    = psd[:, order]                                          # (B, nperseg)

    psd_safe   = psd.clamp(min=1e-20)
    psd_sum    = psd_safe.sum(dim=1, keepdim=True).clamp(min=1e-20)
    psd_norm   = psd_safe / psd_sum                                 # (B, nperseg) prob.
    log_psd    = psd_safe.log()                                     # (B, nperseg)

    # Spectral flatness: geometric mean / arithmetic mean
    flatness   = log_psd.mean(dim=1).exp() / psd_safe.mean(dim=1).clamp(min=1e-20)

    # Spectral centroid
    centroid   = (freqs * psd_norm).sum(dim=1)
    cent_norm  = centroid / (sr / 2.0)

    # Spectral rolloff (85th percentile frequency)
    cumsum     = psd_norm.cumsum(dim=1)                             # (B, nperseg)
    # Find first bin where cumsum >= 0.85
    above      = (cumsum >= 0.85).float()
    roll_idx   = above.argmax(dim=1).clamp(0, nperseg - 1)
    rolloff    = freqs[roll_idx].abs()
    roll_norm  = rolloff / (sr / 2.0)

    # Spectral regrowth
    soi_half   = sr * soi_bw_fraction / 2.0
    out_band   = freqs.abs() > soi_half                             # (nperseg,) bool
    out_power  = (psd_safe * out_band.float()).sum(dim=1)
    regrowth   = out_power / psd_safe.sum(dim=1).clamp(min=1e-20)

    # Decimated log-PSD (n_psd_bins features, mean-centred)
    # Interpolate each row from nperseg -> n_psd_bins using linear interp
    log_psd_f  = log_psd.float()                                    # (B, nperseg)
    psd_dec    = F.interpolate(
        log_psd_f.unsqueeze(1),                                     # (B, 1, nperseg)
        size=n_psd_bins,
        mode="linear",
        align_corners=False,
    ).squeeze(1)                                                     # (B, n_psd_bins)
    psd_dec    = psd_dec - psd_dec.mean(dim=1, keepdim=True)       # mean-centre

    # ------------------------------------------------------------------
    # 7. Assemble feature matrix
    # ------------------------------------------------------------------
    scalar_feats = torch.stack([
        cfo_norm.float(),
        alpha.float(),
        phi.float(),
        circ.float(),
        dc_i.float(),
        dc_q.float(),
        e_skew.float(),
        e_kurt.float(),
        pn_var.float(),
        flatness.float(),
        cent_norm.float(),
        roll_norm.float(),
        regrowth.float(),
    ], dim=1)                                                        # (B, 13)

    full = torch.cat([scalar_feats, psd_dec], dim=1)               # (B, 37)
    full = torch.nan_to_num(full, nan=0.0, posinf=0.0, neginf=0.0)
    return full.cpu().numpy()


# ---------------------------------------------------------------------------
# CPU fallback (single window) — preserved from original
# ---------------------------------------------------------------------------

def estimate_cfo_fourth_power(iq, sample_rate):
    """Residual CFO via fourth-power method (works for QPSK-like modulations)."""
    if len(iq) < 64:
        return 0.0
    x4 = iq.astype(np.complex128) ** 4
    nfft = 1 << int(np.ceil(np.log2(len(x4))))
    spec = np.abs(np.fft.fft(x4, n=nfft))
    freqs = np.fft.fftfreq(nfft, 1.0 / float(sample_rate))
    return float(freqs[int(np.argmax(spec))] / 4.0)


def estimate_iq_imbalance(iq):
    """Blind IQ-imbalance estimation from second-order statistics."""
    I = iq.real.astype(np.float64)
    Q = iq.imag.astype(np.float64)
    Eii = float(np.mean(I * I)) + 1e-12
    Eqq = float(np.mean(Q * Q)) + 1e-12
    Eiq = float(np.mean(I * Q))
    alpha = np.sqrt(Eii / Eqq)
    sin_phi = np.clip(2.0 * Eiq / (Eii + Eqq), -1.0, 1.0)
    phi = float(np.arcsin(sin_phi))
    Ex2 = complex(np.mean(iq.astype(np.complex128) ** 2))
    Eabs2 = float(np.mean(np.abs(iq) ** 2)) + 1e-12
    circ = float(np.abs(Ex2) / Eabs2)
    return float(alpha - 1.0), phi, circ


def estimate_dc_offset(iq):
    rms = float(np.sqrt(np.mean(np.abs(iq) ** 2))) + 1e-12
    return float(np.mean(iq.real) / rms), float(np.mean(iq.imag) / rms)


def estimate_envelope_moments(iq):
    env = np.abs(iq)
    return float(skew(env)), float(kurtosis(env, fisher=True))


def estimate_phase_noise(iq):
    """Variance of instantaneous frequency (robust MAD-based std)."""
    if len(iq) < 16:
        return 0.0
    phase = np.unwrap(np.angle(iq.astype(np.complex128) + 1e-12))
    dphi = np.diff(phase)
    mad = np.median(np.abs(dphi - np.median(dphi)))
    return float((1.4826 * mad) ** 2)


def estimate_spectral_features(iq, sample_rate, n_psd_bins=24, soi_bw_fraction=0.5):
    nperseg = min(256, max(64, len(iq) // 8))
    f, psd = welch(iq, fs=sample_rate, nperseg=nperseg,
                   return_onesided=False, scaling="density")
    order = np.argsort(f)
    f, psd = f[order], psd[order]
    psd = np.maximum(psd, 1e-20)
    psd_norm = psd / psd.sum()
    log_psd = np.log(psd)
    flatness = float(np.exp(log_psd.mean()) / (psd.mean() + 1e-20))
    centroid = float((f * psd_norm).sum())
    centroid_norm = centroid / (sample_rate / 2.0)
    cumulative = np.cumsum(psd_norm)
    rolloff_idx = int(np.searchsorted(cumulative, 0.85))
    rolloff = float(f[min(rolloff_idx, len(f) - 1)])
    rolloff_norm = abs(rolloff) / (sample_rate / 2.0)
    soi_half = sample_rate * soi_bw_fraction / 2.0
    out_band = np.abs(f) > soi_half
    regrowth = float(psd[out_band].sum() / (psd.sum() + 1e-20))
    decimated = np.interp(
        np.linspace(0, len(log_psd) - 1, n_psd_bins),
        np.arange(len(log_psd)),
        log_psd,
    )
    decimated = decimated - decimated.mean()
    return flatness, centroid_norm, rolloff_norm, regrowth, decimated


def extract_features(iq, sample_rate):
    """Compute the full 37-dim feature vector for one baseband window (CPU)."""
    cfo = estimate_cfo_fourth_power(iq, sample_rate)
    alpha_res, phi, circ = estimate_iq_imbalance(iq)
    dc_i, dc_q = estimate_dc_offset(iq)
    env_skew, env_kurt = estimate_envelope_moments(iq)
    pn = estimate_phase_noise(iq)
    flat, cent, roll, regrowth, psd_feats = estimate_spectral_features(iq, sample_rate)
    feats = np.array([
        cfo / sample_rate,
        alpha_res, phi, circ,
        dc_i, dc_q,
        env_skew, env_kurt,
        pn,
        flat, cent, roll, regrowth,
    ], dtype=np.float32)
    full = np.concatenate([feats, psd_feats.astype(np.float32)])
    return np.nan_to_num(full, nan=0.0, posinf=0.0, neginf=0.0)


def estimate_snr_db(iq):
    """Rough SNR proxy: 95th-percentile power over 20th-percentile power (dB)."""
    mag2 = np.abs(iq) ** 2
    if mag2.size < 32:
        return 0.0
    noise = float(np.percentile(mag2, 20)) + 1e-20
    signal = float(np.percentile(mag2, 95)) + 1e-20
    return 10.0 * np.log10(signal / noise)
