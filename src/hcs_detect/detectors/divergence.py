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

"""Distributional-divergence detectors: how far a session sits from the benign users of the
same service. That is to say, we model the cover, and measure departure from it.

``kl_len`` and ``ks_len``   the packet-size distribution against the pooled benign baseline;
``kl_iat``                  the inter-arrival distribution (log-binned) against the same.

KL is reported as KL(session || benign) with additive smoothing, so that empty reference
bins cannot blow up. It scores the whole histogram, which is why it was the single most
broadly useful detector in the spot-check: it caught the Skyhook receiver, whose
keep-alive connections defeat both connection counting and beaconing, at 3.87 bits
against a benign maximum of 0.009.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp

from . import SessionContext, register

LEN_BINS = np.array([0, 60, 80, 120, 200, 400, 600, 800, 1000, 1200, 1400, 1500, 1600, 65536])
IAT_BINS = np.logspace(-4, 3, 22)


def kl_divergence(p_counts: np.ndarray, q_counts: np.ndarray, eps: float = 1e-3) -> float:
    """KL(P || Q) in bits from two count histograms, with additive smoothing."""
    p = np.asarray(p_counts, dtype=float) + eps
    q = np.asarray(q_counts, dtype=float) + eps
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log2(p / q)))


def len_hist(lengths: np.ndarray) -> np.ndarray:
    return np.histogram(lengths, bins=LEN_BINS)[0]


def iat_hist(ts_sorted: np.ndarray) -> np.ndarray:
    d = np.diff(ts_sorted)
    return np.histogram(d[d > 0], bins=IAT_BINS)[0]


@register("kl_len", "divergence", "high")
def kl_len(ctx: SessionContext) -> float:
    """KL divergence (bits) of the session's packet-size histogram from the benign baseline."""
    if ctx.baseline is None:
        return float("nan")
    return kl_divergence(len_hist(ctx.len), ctx.baseline.len_hist)


@register("ks_len", "divergence", "high")
def ks_len(ctx: SessionContext) -> float:
    """Kolmogorov-Smirnov statistic of packet sizes against the benign size pool."""
    if ctx.baseline is None or ctx.baseline.len_pool.size == 0:
        return float("nan")
    return float(ks_2samp(ctx.len, ctx.baseline.len_pool).statistic)


@register("kl_iat", "divergence", "high")
def kl_iat(ctx: SessionContext) -> float:
    """KL divergence (bits) of the log-binned inter-arrival histogram from the benign baseline."""
    if ctx.baseline is None:
        return float("nan")
    return kl_divergence(iat_hist(ctx.ts), ctx.baseline.iat_hist)
