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

"""Client-to-server payload prefixes and first data segments, keyed by (host, service, port).

We take two views of the same ``*.payload.psv`` tables. ``prefixes`` holds the concatenated
first N bytes of every flow in a session (capped), for whole-session byte statistics.
``first_segments`` holds the first ``first_segment_bytes`` of each flow's first data segment
(relative ``tcp.seq == 1``); this is the like-for-like window the FEP test uses, so that a
single-flow tunnel and a poller with two thousand connections are compared on equal footing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .logs import get_logger
from .netutil import ip_to_u32

log = get_logger(__name__)

SessionKey = tuple[int, int, int]  # (host_u32, svc_u32, port)


@dataclass
class PayloadIndex:
    prefixes: dict[SessionKey, bytes] = field(default_factory=dict)
    first_segments: dict[SessionKey, list[bytes]] = field(default_factory=dict)

    @classmethod
    def load(cls, extract_dir: Path, *, first_segment_bytes: int = 64, prefix_cap: int = 262_144) -> PayloadIndex:
        idx = cls()
        files = 0
        for psv in sorted(extract_dir.glob("*.payload.psv")):
            if psv.stat().st_size == 0:
                continue
            files += 1
            for chunk in pd.read_csv(psv, sep="|", dtype=str, chunksize=200_000):
                chunk = chunk[chunk["tcp.payload"].notna()]
                seq = pd.to_numeric(chunk["tcp.seq"], errors="coerce")
                hexes = chunk["tcp.payload"].str.replace(":", "", regex=False)
                for s, d, dp, sq, hx in zip(
                    chunk["ip.src"], chunk["ip.dst"], chunk["tcp.dstport"], seq, hexes, strict=True
                ):
                    if len(hx) % 2:
                        continue
                    key: SessionKey = (ip_to_u32(s), ip_to_u32(d), int(dp))
                    cur = idx.prefixes.get(key, b"")
                    if len(cur) < prefix_cap:
                        idx.prefixes[key] = cur + bytes.fromhex(hx)
                    if sq == 1 and len(hx) >= 16:
                        idx.first_segments.setdefault(key, []).append(bytes.fromhex(hx[: first_segment_bytes * 2]))
        log.info(
            "payload_indexed",
            files=files,
            sessions=len(idx.prefixes),
            prefix_bytes=sum(len(v) for v in idx.prefixes.values()),
            first_segments=sum(len(v) for v in idx.first_segments.values()),
        )
        if files == 0:
            log.warning("payload_missing", extract_dir=extract_dir, effect="fep detectors will be NaN")
        return idx
