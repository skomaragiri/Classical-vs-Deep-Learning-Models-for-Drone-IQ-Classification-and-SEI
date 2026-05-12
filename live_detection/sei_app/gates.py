#!/usr/bin/python3
"""Modular discriminator gates for SEI.

Each gate is a small, independent check that returns one of three verdicts:

    ACCEPT   the gate is satisfied
    REJECT   the gate positively disqualifies the sample
    ABSTAIN  the gate could not make a determination (missing data)

The GatePipeline composes a list of gates and folds the outcomes:

    - any REJECT    -> pipeline REJECTs
    - else any ACCEPT -> pipeline ACCEPTs
    - else         -> pipeline ABSTAINs

This design lets callers wire together whichever discriminators are available
in their context. If one signal is missing (no emitter profile, no meta SNR,
too-few samples for wideband SNR), that gate abstains and the rest carry the
verdict. No single gate is required.

Gates currently provided:

    freq_band_gate        cf / bw vs emitter profile
    duration_gate         sample_count / samp_rate vs protocol bounds
    meta_snr_gate         annotation-supplied SNR (capture_details or deepsig)
    power_gate            raw-IQ power floor (|iq|^2 in dBFS)
    wideband_snr_gate     in-band vs out-of-band spectral power
    blind_snr_gate        percentile-ratio SNR on channelized baseband (last resort)

Profiles live in emitter_profiles.json, keyed by core:label. All fields are
optional; a missing field makes the relevant gate abstain.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional

import numpy as np

ACCEPT = "accept"
REJECT = "reject"
ABSTAIN = "abstain"


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #

@dataclass
class GateResult:
    verdict: str
    name: str
    reason: str = ""
    value: Optional[float] = None   # measured quantity if applicable

    def tag(self) -> str:
        if self.value is None:
            return f"{self.name}:{self.verdict}"
        return f"{self.name}:{self.verdict}({self.value:.2f})"


@dataclass
class PipelineOutcome:
    verdict: str
    results: List[GateResult] = field(default_factory=list)

    @property
    def reasons(self) -> List[str]:
        if self.verdict == REJECT:
            return [f"{r.name}: {r.reason}" for r in self.results if r.verdict == REJECT]
        if self.verdict == ABSTAIN:
            return ["all gates abstained"]
        return []

    def summary(self) -> str:
        return " | ".join(r.tag() for r in self.results) or "(no gates)"


# --------------------------------------------------------------------------- #
# Individual gates. Every one is pure and independent.
# --------------------------------------------------------------------------- #

def freq_band_gate(p: dict, expected_cf=None, expected_bw=None,
                   cf_tol_hz: float = 2e6, bw_tol_hz: float = 5e6) -> GateResult:
    """Compare annotation cf/bw against an emitter profile."""
    have_cf = expected_cf is not None
    have_bw = expected_bw is not None
    if not have_cf and not have_bw:
        return GateResult(ABSTAIN, "freq_band", "no profile")
    if have_cf:
        cf_err = abs(p["cf"] - float(expected_cf))
        if cf_err > cf_tol_hz:
            return GateResult(REJECT, "freq_band",
                              f"cf {p['cf']/1e6:.2f} MHz outside "
                              f"{float(expected_cf)/1e6:.2f}±{cf_tol_hz/1e6:.2f}",
                              value=p["cf"])
    if have_bw:
        bw_err = abs(p["bw"] - float(expected_bw))
        if bw_err > bw_tol_hz:
            return GateResult(REJECT, "freq_band",
                              f"bw {p['bw']/1e6:.2f} MHz outside "
                              f"{float(expected_bw)/1e6:.2f}±{bw_tol_hz/1e6:.2f}",
                              value=p["bw"])
    return GateResult(ACCEPT, "freq_band",
                      f"cf={p['cf']/1e6:.2f}MHz bw={p['bw']/1e6:.2f}MHz",
                      value=p["cf"])


def duration_gate(p: dict, min_s: Optional[float] = None,
                  max_s: Optional[float] = None) -> GateResult:
    sc = int(p.get("sample_count", 0))
    sr = int(p.get("samp_rate", 0))
    if sr <= 0 or sc <= 0:
        return GateResult(ABSTAIN, "duration", "missing sample info")
    if min_s is None and max_s is None:
        return GateResult(ABSTAIN, "duration", "no bounds configured")
    dur = sc / float(sr)
    if min_s is not None and dur < float(min_s):
        return GateResult(REJECT, "duration",
                          f"dur {dur*1e3:.2f} ms < min {float(min_s)*1e3:.2f}",
                          value=dur)
    if max_s is not None and dur > float(max_s):
        return GateResult(REJECT, "duration",
                          f"dur {dur*1e3:.2f} ms > max {float(max_s)*1e3:.2f}",
                          value=dur)
    return GateResult(ACCEPT, "duration", f"dur {dur*1e3:.2f} ms", value=dur)


def meta_snr_gate(p: dict, min_snr_db: Optional[float] = None,
                  treat_zero_as_missing: bool = True) -> GateResult:
    """Authoritative SNR from capture_details:SNRdB or deepsig:snr_estimate.

    OmniSIG often ships annotations with an unpopulated deepsig:snr_estimate
    that is literally the int 0 (not null). `treat_zero_as_missing` folds that
    case into ABSTAIN so the pipeline falls through to other gates instead of
    rejecting a real signal on a sentinel reading.
    """
    if min_snr_db is None:
        return GateResult(ABSTAIN, "meta_snr", "disabled")
    snr = p.get("snr_estimate")
    if snr is None:
        return GateResult(ABSTAIN, "meta_snr", "no meta snr")
    if treat_zero_as_missing and float(snr) == 0.0:
        return GateResult(ABSTAIN, "meta_snr", "snr=0 (likely unpopulated)")
    if float(snr) < float(min_snr_db):
        return GateResult(REJECT, "meta_snr",
                          f"snr {float(snr):.1f} dB < min {float(min_snr_db):.1f}",
                          value=float(snr))
    return GateResult(ACCEPT, "meta_snr", f"snr {float(snr):.1f} dB", value=float(snr))


def power_gate(iq: Optional[np.ndarray],
               min_power_dbfs: Optional[float] = None) -> GateResult:
    """Raw-IQ power floor in dBFS (full-scale = 1.0 complex magnitude)."""
    if min_power_dbfs is None:
        return GateResult(ABSTAIN, "power", "disabled")
    if iq is None or len(iq) == 0:
        return GateResult(ABSTAIN, "power", "empty iq")
    p_lin = float(np.mean(np.abs(iq) ** 2))
    if p_lin <= 0:
        return GateResult(REJECT, "power", "zero power", value=-200.0)
    p_db = 10.0 * np.log10(p_lin + 1e-30)
    if p_db < float(min_power_dbfs):
        return GateResult(REJECT, "power",
                          f"power {p_db:.1f} dBFS < min {float(min_power_dbfs):.1f}",
                          value=p_db)
    return GateResult(ACCEPT, "power", f"power {p_db:.1f} dBFS", value=p_db)


def wideband_snr_gate(wideband_iq: Optional[np.ndarray], samp_rate: float,
                      recorded_cf: float, lower_edge: float, upper_edge: float,
                      min_snr_db: Optional[float] = None,
                      guard_hz: Optional[float] = None,
                      nperseg: int = 4096) -> GateResult:
    """Estimate SNR from in-band vs out-of-band spectral power.

    Uses the PRE-channelization IQ so there's a real noise reference outside
    the signal's frequency range. This replaces the blind percentile-ratio
    estimator, which fails on channelized baseband.
    """
    if min_snr_db is None:
        return GateResult(ABSTAIN, "wideband_snr", "disabled")
    if wideband_iq is None or len(wideband_iq) < nperseg:
        return GateResult(ABSTAIN, "wideband_snr",
                          f"need >={nperseg} samples, have {0 if wideband_iq is None else len(wideband_iq)}")
    try:
        from scipy.signal import welch
    except Exception:
        return GateResult(ABSTAIN, "wideband_snr", "scipy unavailable")

    f, psd = welch(wideband_iq, fs=float(samp_rate),
                   nperseg=min(nperseg, len(wideband_iq)),
                   return_onesided=False, scaling="density")
    order = np.argsort(f)
    f, psd = f[order], psd[order]
    f_abs = f + float(recorded_cf)

    bw = float(upper_edge - lower_edge)
    if guard_hz is None:
        guard_hz = max(0.25 * bw, 1e6)

    sig_mask = (f_abs >= float(lower_edge)) & (f_abs <= float(upper_edge))
    noise_mask = ((f_abs < float(lower_edge) - guard_hz) |
                  (f_abs > float(upper_edge) + guard_hz))
    if not sig_mask.any() or not noise_mask.any():
        return GateResult(ABSTAIN, "wideband_snr",
                          "signal/noise bands don't fit in capture bandwidth")
    sig_p = float(np.mean(psd[sig_mask]))
    noise_p = float(np.mean(psd[noise_mask]))
    if sig_p <= 0 or noise_p <= 0:
        return GateResult(ABSTAIN, "wideband_snr", "degenerate power")
    snr = 10.0 * np.log10(sig_p / noise_p)
    if snr < float(min_snr_db):
        return GateResult(REJECT, "wideband_snr",
                          f"snr {snr:.1f} dB < min {float(min_snr_db):.1f}",
                          value=snr)
    return GateResult(ACCEPT, "wideband_snr", f"snr {snr:.1f} dB", value=snr)


def blind_snr_gate(baseband_iq: Optional[np.ndarray],
                   min_snr_db: Optional[float] = None) -> GateResult:
    """Last-resort percentile-ratio SNR on channelized baseband.

    Unreliable when the whole burst is above the noise floor (no quiet bins),
    but useful as a final fallback when nothing else is available.
    """
    if min_snr_db is None:
        return GateResult(ABSTAIN, "blind_snr", "disabled")
    if baseband_iq is None or len(baseband_iq) < 32:
        return GateResult(ABSTAIN, "blind_snr", "empty or too-short iq")
    try:
        from features import estimate_snr_db
    except Exception:
        from .features import estimate_snr_db  # pragma: no cover
    snr = float(estimate_snr_db(baseband_iq))
    if snr < float(min_snr_db):
        return GateResult(REJECT, "blind_snr",
                          f"snr {snr:.1f} dB < min {float(min_snr_db):.1f}",
                          value=snr)
    return GateResult(ACCEPT, "blind_snr", f"snr {snr:.1f} dB", value=snr)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

class GatePipeline:
    """Run a list of zero-arg gate callables and fold the outcome.

    Example:
        pipeline = GatePipeline([
            lambda: freq_band_gate(p, **profile.get("freq_band", {})),
            lambda: meta_snr_gate(p, min_snr_db=15.0),
            lambda: wideband_snr_gate(raw, sr, cf, lo, hi, min_snr_db=15.0),
            lambda: power_gate(raw, min_power_dbfs=-80.0),
            lambda: duration_gate(p, min_s=1e-5, max_s=1e-2),
        ])
        outcome = pipeline.run()
        if outcome.verdict == REJECT: ...
    """
    def __init__(self, gates: Iterable[Callable[[], GateResult]]):
        self.gates = list(gates)

    def run(self) -> PipelineOutcome:
        results: List[GateResult] = []
        for g in self.gates:
            try:
                results.append(g())
            except Exception as e:   # don't let a broken gate poison the rest
                results.append(GateResult(ABSTAIN, getattr(g, "__name__", "gate"),
                                          f"error: {e!r}"))
        any_reject = any(r.verdict == REJECT for r in results)
        any_accept = any(r.verdict == ACCEPT for r in results)
        if any_reject:
            return PipelineOutcome(REJECT, results)
        if any_accept:
            return PipelineOutcome(ACCEPT, results)
        return PipelineOutcome(ABSTAIN, results)


# --------------------------------------------------------------------------- #
# Emitter-profile loader. Profiles are a plain dict, keyed by core:label.
# --------------------------------------------------------------------------- #

def load_profiles(path: Optional[str]) -> dict:
    """Load the emitter-profile JSON. Returns {} on missing path."""
    if not path or not os.path.isfile(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def get_profile(profiles: dict, label: Optional[str]) -> dict:
    """Look up a per-label profile; return an empty dict if missing."""
    if not label:
        return {}
    return dict(profiles.get(label, {}))


# --------------------------------------------------------------------------- #
# Convenience builder: wire up the standard set of gates from a profile +
# runtime args. Any field missing from the profile just means the gate abstains.
# --------------------------------------------------------------------------- #

def build_standard_pipeline(p: dict,
                            wideband_iq: Optional[np.ndarray],
                            baseband_iq: Optional[np.ndarray],
                            profile: Optional[dict] = None,
                            min_snr_db: Optional[float] = None,
                            min_power_dbfs: Optional[float] = None,
                            use_wideband_snr: bool = True,
                            use_blind_snr_fallback: bool = False) -> GatePipeline:
    """Build the default gate pipeline.

    `profile` may contain any of:
        expected_cf, expected_bw, cf_tol_hz, bw_tol_hz, min_duration_s,
        max_duration_s, min_snr_db (overrides arg), min_power_dbfs (overrides arg)
    """
    profile = profile or {}
    # Profile values override function args (so per-emitter tightness wins).
    eff_min_snr = profile.get("min_snr_db", min_snr_db)
    eff_min_pwr = profile.get("min_power_dbfs", min_power_dbfs)

    gates: List[Callable[[], GateResult]] = [
        lambda: freq_band_gate(
            p,
            expected_cf=profile.get("expected_cf"),
            expected_bw=profile.get("expected_bw"),
            cf_tol_hz=float(profile.get("cf_tol_hz", 2e6)),
            bw_tol_hz=float(profile.get("bw_tol_hz", 5e6)),
        ),
        lambda: duration_gate(
            p,
            min_s=profile.get("min_duration_s"),
            max_s=profile.get("max_duration_s"),
        ),
        lambda: meta_snr_gate(p, min_snr_db=eff_min_snr),
        lambda: power_gate(wideband_iq, min_power_dbfs=eff_min_pwr),
    ]
    if use_wideband_snr and "recorded_cf" in p and "lower_freq" in p and "upper_freq" in p:
        gates.append(
            lambda: wideband_snr_gate(
                wideband_iq, p.get("samp_rate", 0), p["recorded_cf"],
                p["lower_freq"], p["upper_freq"], min_snr_db=eff_min_snr,
            )
        )
    if use_blind_snr_fallback:
        gates.append(lambda: blind_snr_gate(baseband_iq, min_snr_db=eff_min_snr))
    return GatePipeline(gates)
