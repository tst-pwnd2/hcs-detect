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

"""Capture facts worth knowing before trusting any statistic.

Two lessons from the spot-check are codified here. First, files named after different
interfaces may be time-contiguous chunks of one capture (size-rotated) rather than
parallel vantage points, so we check the chunk boundaries. Second, a router capture may
see each forwarded packet twice, on ingress and on egress. The clean test is the TTL
distribution: one uniform value means single-point observation, and a mix of n and n-1
means duplicates. Note that the naive heuristic (same 5-tuple within a few milliseconds)
is fooled by back-to-back ACKs and bulk segments, and we report it only as a warning.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .ingest import cache_path
from .logs import get_logger

log = get_logger(__name__)


@dataclass
class ChunkFacts:
    chunk: str
    packets: int
    start: float
    end: float
    duration_s: float


@dataclass
class CaptureFacts:
    chunks: list[ChunkFacts]
    contiguous: bool
    max_gap_s: float
    total_packets: int
    span_s: float
    ttl_top: dict[int, int]
    single_ttl_share: float
    likely_single_vantage: bool
    naive_dup_share: float

    def render(self) -> str:
        lines = ["=== capture facts ==="]
        for c in self.chunks:
            lines.append(f"  {c.chunk:8s} {c.packets:>10,d} pkts  {c.start:.0f} to {c.end:.0f}  ({c.duration_s:.0f}s)")
        lines.append(
            f"  chunks time-contiguous: {self.contiguous} (max gap {self.max_gap_s:.3f}s) "
            f"-> treat as one timeline: {self.contiguous}"
        )
        lines.append(f"  total {self.total_packets:,} pkts over {self.span_s:.0f}s")
        lines.append(f"  TTL distribution (top): {self.ttl_top}   dominant-TTL share {self.single_ttl_share:.3f}")
        lines.append(f"  single-vantage (no in+out duplication): {self.likely_single_vantage}")
        lines.append(
            f"  naive same-5-tuple-within-5ms share: {self.naive_dup_share:.3f}  "
            "(inflated by ACK bursts; NOT a duplicate measure)"
        )
        return "\n".join(lines)


def capture_facts(extract_dir: Path, chunks: Sequence[str]) -> CaptureFacts:
    cf: list[ChunkFacts] = []
    frames = []
    for c in chunks:
        df = pd.read_pickle(cache_path(extract_dir, c))
        cf.append(
            ChunkFacts(c, len(df), float(df["ts"].min()), float(df["ts"].max()), float(df["ts"].max() - df["ts"].min()))
        )
        frames.append(df[["ts", "src", "dst", "sport", "dport", "len", "ttl"]])
    cf.sort(key=lambda x: x.start)
    gaps = [cf[i + 1].start - cf[i].end for i in range(len(cf) - 1)]
    max_gap = max(gaps) if gaps else 0.0
    all_ = pd.concat(frames, ignore_index=True)
    ttl_counts = all_["ttl"].value_counts()
    top = {int(k): int(v) for k, v in ttl_counts.head(5).items()}
    share = float(ttl_counts.iloc[0] / len(all_)) if len(all_) else 0.0
    d = all_.sort_values("ts", kind="stable")
    key = d[["src", "dst", "sport", "dport", "len"]].to_numpy()
    same = (key[1:] == key[:-1]).all(axis=1)
    dt = np.diff(d["ts"].to_numpy())
    naive = float(np.mean(same & (dt < 0.005))) if len(d) > 1 else 0.0
    facts = CaptureFacts(
        chunks=cf,
        contiguous=bool(max_gap < 1.0),
        max_gap_s=float(max_gap),
        total_packets=len(all_),
        span_s=float(cf[-1].end - cf[0].start) if cf else 0.0,
        ttl_top=top,
        single_ttl_share=share,
        likely_single_vantage=bool(share > 0.98),
        naive_dup_share=naive,
    )
    log.info(
        "capture_facts",
        chunks=len(cf),
        contiguous=facts.contiguous,
        max_gap_s=round(facts.max_gap_s, 3),
        total_packets=facts.total_packets,
        span_s=round(facts.span_s, 1),
        dominant_ttl=next(iter(top), None),
        dominant_ttl_share=round(share, 4),
        single_vantage=facts.likely_single_vantage,
        naive_dup_share=round(naive, 4),
    )
    if not facts.contiguous:
        log.warning(
            "chunks_not_contiguous", max_gap_s=round(facts.max_gap_s, 3), effect="gaps will inflate inter-arrivals"
        )
    if not facts.likely_single_vantage:
        log.warning("multi_vantage_suspected", ttl_top=top, effect="packets may be counted twice; dedup before scoring")
    return facts
