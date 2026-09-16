#
# This work was authored by Two Six Labs, LLC dba Two Six Technologies and team
# in support of Defense Advanced Research Projects Agency under Agreement
# HR001125CE021.
#
# Use, duplication, or disclosure is subject to the restrictions as stated in
# Agreement HR001125CE021 between the Government and the Performer.
#
# Copyright 2023-2026 Two Six Technologies. All rights reserved.
#

"""Fully-encrypted-protocol (FEP) detectors on the first bytes of each flow.

obfs4 and its relatives have no plaintext handshake; from byte zero the stream is
indistinguishable from random. TLS, HTTP, IRC, and SSH, by contrast, announce themselves
in the first handful of bytes. We therefore run the test per flow on the first data
segment and average over the session's flows. This is a like-for-like window, and it
lets a single-flow tunnel be compared fairly with a poller that opens two thousand
connections.

``fep_entropy``   the mean Shannon entropy (bits per byte) of the first N bytes;
``fep_popcount``  the mean number of set bits per byte, which is 4.0 for uniform random data
                  and lower for structured protocols (the classic GFW heuristic);
``fep_magic``     the fraction of flows whose first bytes carry known protocol magic (TLS
                  ``16 03``, or printable ASCII); a value of 0.0 is the FEP tell.

``payload_entropy`` and ``payload_kl_uniform`` are whole-prefix statistics over the pooled
client-to-server bytes. They are size-biased (a 660-byte sample cannot reach 8 bits), and
we keep them for context rather than ranking.
"""

from __future__ import annotations

import numpy as np

from . import SessionContext, register
from .divergence import kl_divergence


def shannon_entropy(data: bytes) -> float:
    """Shannon entropy in bits per byte (NaN for empty input)."""
    if not data:
        return float("nan")
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256).astype(float)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def popcount_per_byte(data: bytes) -> float:
    if not data:
        return float("nan")
    return float(np.unpackbits(np.frombuffer(data, dtype=np.uint8)).mean() * 8)


def has_known_magic(segment: bytes) -> bool:
    """True if the segment opens like a known protocol: TLS record header or printable ASCII."""
    if len(segment) < 2:
        return False
    if segment[:2] == b"\x16\x03":  # TLS handshake record, any version
        return True
    head = segment[:8]
    return len(head) >= 4 and all(32 <= b < 127 for b in head)


def _segments(ctx: SessionContext) -> list[bytes]:
    return ctx.first_segments


@register("fep_flows", "fep", "high", rank=False)
def fep_flows(ctx: SessionContext) -> float:
    """Number of first segments the FEP statistics were averaged over."""
    return float(len(_segments(ctx)))


@register("fep_entropy", "fep", "high")
def fep_entropy(ctx: SessionContext) -> float:
    """Mean Shannon entropy of each flow's first bytes."""
    segs = _segments(ctx)
    return float(np.mean([shannon_entropy(s) for s in segs])) if segs else float("nan")


@register("fep_popcount", "fep", "high")
def fep_popcount(ctx: SessionContext) -> float:
    """Mean set bits per byte over each flow's first bytes (about 4.0 for random data)."""
    segs = _segments(ctx)
    return float(np.mean([popcount_per_byte(s) for s in segs])) if segs else float("nan")


@register("fep_magic", "fep", "low")
def fep_magic(ctx: SessionContext) -> float:
    """Fraction of flows whose first bytes carry recognizable protocol magic."""
    segs = _segments(ctx)
    return float(np.mean([has_known_magic(s) for s in segs])) if segs else float("nan")


@register("payload_entropy", "fep", "high", rank=False)
def payload_entropy(ctx: SessionContext) -> float:
    """Entropy of the pooled client-to-server prefix bytes (context only; size-biased)."""
    return shannon_entropy(ctx.payload) if len(ctx.payload) >= 512 else float("nan")


@register("payload_kl_uniform", "fep", "low", rank=False)
def payload_kl_uniform(ctx: SessionContext) -> float:
    """KL of the pooled prefix byte distribution from uniform (context only; size-biased)."""
    if len(ctx.payload) < 512:
        return float("nan")
    counts = np.bincount(np.frombuffer(ctx.payload, dtype=np.uint8), minlength=256)
    return kl_divergence(counts, np.ones(256), eps=0.5)
