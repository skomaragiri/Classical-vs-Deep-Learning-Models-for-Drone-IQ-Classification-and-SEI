#!/usr/bin/python3
"""Internal helpers used by the SEI trainer and SEI app.

Scope: NOT a replacement for pylibos_common.py. These are the few extras the
SEI code needs beyond what your pylibos_* modules already provide:

  * annotation_iq()       read just one annotation's raw IQ from a SigMF pair
  * channelize_inmem()    in-process baseband shift + resample (used for
                          TRAINING, where subprocess-per-annotation is too
                          slow; production inference uses resample_cli.py)
  * iq_window_to_bn2()    pack a complex64 window into the (1,N,2) float32
                          layout your pylibos_common.numpy_to_tensor produces
  * slice_windows()       slide fixed-length windows across a baseband burst
  * read_raw_complex64()  read a headerless complex64 file (resample_cli
                          output has no .sigmf-meta)

Relies on pylibos_common.get_samples_mmap when available for the SigMF read.
"""
from fractions import Fraction
import numpy as np
import os
import mmap
from scipy.signal import resample_poly

try:
    from pylibos_common import get_samples_mmap, dsprint
except Exception:
    def dsprint(level, msg):
        print(f"[{level}] - {msg}")

    def get_samples_mmap(filename, datatype, offset, sample_count):
        if datatype == "cf32_le":
            start_byte = offset * 8
            num_bytes = sample_count * 8
            with open(filename, "rb") as f:
                with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
                    mm.seek(start_byte)
                    return np.frombuffer(mm.read(num_bytes), dtype=np.complex64).astype("complex64")
        elif datatype == "ci16_le":
            start_byte = offset * 4
            num_bytes = sample_count * 4
            with open(filename, "rb") as f:
                with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
                    mm.seek(start_byte)
                    raw = np.frombuffer(mm.read(num_bytes), dtype=np.int16)
                    return (raw[0::2] / 32767.0 + 1j * raw[1::2] / 32767.0).astype("complex64")
        raise ValueError(f"Unsupported datatype: {datatype}")


def annotation_iq(datafile, md, anno):
    """Read the raw (un-channelized) IQ samples that cover one annotation."""
    meta_dt = md['global']['core:datatype']
    s = int(anno['core:sample_start'])
    n = int(anno['core:sample_count'])
    if n <= 0:
        return np.zeros(0, dtype=np.complex64)
    return get_samples_mmap(datafile, meta_dt, s, n)


def channelize_inmem(iq, sample_rate, offset, target_bw, oversample=4.0):
    """Frequency-shift by -offset then resample to target_bw * oversample.

    Only used for TRAINING-time dataset construction. Production inference
    goes through resample_cli.py (invoked by sei_pipeline.py).

    Oversampling keeps the image-frequency leakage and spectral regrowth that
    carry IQ-imbalance / PA-nonlinearity fingerprints.
    """
    n = len(iq)
    if n == 0:
        return iq, sample_rate
    t = np.arange(n, dtype=np.float64) / float(sample_rate)
    shifted = (iq * np.exp(-2j * np.pi * offset * t)).astype(np.complex64)

    target_sr = target_bw * oversample
    frac = Fraction(target_sr / float(sample_rate)).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator
    if up == 0 or down == 0 or up == down:
        return shifted, sample_rate
    out = resample_poly(shifted, up, down).astype(np.complex64)
    return out, float(sample_rate) * up / down


def iq_window_to_bn2(iq, target_len):
    """Pack a complex window into a (1, target_len, 2) float32 array.

    Matches pylibos_common.numpy_to_tensor's layout: last dim is [I, Q].
    RMS-normalizes so the model learns shape, not absolute power.
    """
    if len(iq) >= target_len:
        start = (len(iq) - target_len) // 2
        iq = iq[start:start + target_len]
    else:
        iq = np.pad(iq, (0, target_len - len(iq)))
    rms = float(np.sqrt(np.mean(np.abs(iq) ** 2) + 1e-12))
    iq = iq / rms
    out = np.empty((1, target_len, 2), dtype=np.float32)
    out[0, :, 0] = iq.real.astype(np.float32)
    out[0, :, 1] = iq.imag.astype(np.float32)
    return out


def slice_windows(baseband, window_len, stride=None, max_windows=16):
    """Yield fixed-length complex windows from a baseband burst."""
    stride = stride or window_len
    n = len(baseband)
    if n < window_len // 2:
        return
    if n < window_len:
        yield np.pad(baseband, (0, window_len - n))
        return
    count = 0
    for i in range(0, n - window_len + 1, stride):
        yield baseband[i:i + window_len]
        count += 1
        if count >= max_windows:
            return


def read_raw_complex64(path):
    """Read a headerless complex64 file (the format resample_cli.py writes)."""
    size = os.path.getsize(path)
    n = size // 8  # 8 bytes per complex64 sample
    with open(path, "rb") as f:
        with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
            return np.frombuffer(mm.read(n * 8), dtype=np.complex64).astype(np.complex64)
