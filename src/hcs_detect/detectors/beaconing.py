#
# This work was authored by Two Six Labs, LLC dba Two Six Technologies and team
# in support of Defense Advanced Research Projects Agency under Agreement
# <CONTRACT NUMBER>.
#
# Use, duplication, or disclosure is subject to the restrictions as stated in
# Agreement HR001125CE021 between the Government and the Performer.
#
# Copyright 2023-2026 Two Six Technologies. All rights reserved.
#

"""Beaconing and periodicity detectors on request events.

A dead-drop sender polls on a timer, so its inter-request intervals cluster tightly
around one value; cover traffic driven by people or by downloads does not. We expose
three complementary statistics:

``cv_iat``      the coefficient of variation of the inter-arrivals (low means regular);
``mode_frac``   the share of intervals within plus or minus 10% of the median, a regularity
                measure that a few long gaps cannot wash out;
``spec_peak``   the dominant spectral line's share of total power in the 1 s binned event series.
"""

from __future__ import annotations

import numpy as np

from . import SessionContext, register

MIN_EVENTS = 6


def periodicity(times: np.ndarray, *, tolerance: float = 0.10, max_bins: int = 4096) -> tuple[float, float, float]:
    """Return ``(cv, mode_frac, spectral_peak_ratio)`` for sorted event times; NaNs if too few."""
    t = np.sort(np.asarray(times, dtype=float))
    if t.size < MIN_EVENTS:
        return (np.nan, np.nan, np.nan)
    d = np.diff(t)
    d = d[d > 0]
    if d.size < MIN_EVENTS - 1:
        return (np.nan, np.nan, np.nan)
    cv = float(d.std() / d.mean())
    med = float(np.median(d))
    mode_frac = float(np.mean(np.abs(d - med) <= tolerance * med))
    span = t[-1] - t[0]
    bins = int(min(max(span, 10.0), max_bins))
    hist, _ = np.histogram(t, bins=bins)
    hist = hist - hist.mean()
    power = np.abs(np.fft.rfft(hist)) ** 2
    power = power[1:]  # drop DC
    spr = float(power.max() / power.sum()) if power.sum() > 0 else np.nan
    return (cv, mode_frac, spr)


def _events(ctx: SessionContext) -> np.ndarray:
    return ctx.request_events()


@register("req_events", "beaconing", "high", rank=False)
def req_events(ctx: SessionContext) -> float:
    """Number of request events the periodicity statistics were computed on."""
    return float(_events(ctx).size)


@register("cv_iat", "beaconing", "low")
def cv_iat(ctx: SessionContext) -> float:
    """Coefficient of variation of inter-request intervals (low = timer-like)."""
    return periodicity(_events(ctx))[0]


@register("mode_frac", "beaconing", "high")
def mode_frac(ctx: SessionContext) -> float:
    """Fraction of inter-request intervals within plus or minus 10% of the median."""
    return periodicity(_events(ctx))[1]


@register("spec_peak", "beaconing", "high")
def spec_peak(ctx: SessionContext) -> float:
    """Dominant spectral peak's share of (non-DC) power in the 1 s event series."""
    return periodicity(_events(ctx))[2]
