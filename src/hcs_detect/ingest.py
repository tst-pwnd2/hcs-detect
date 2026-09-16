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

"""Turn tshark feature tables into a compact, all-numeric packet table.

Memory is the constraint. A 500 MB pcap yields about 5M rows, and a naive pandas load with
string IPs and int64 everywhere runs to several GB. Here IPs are ``uint32``, ports and
lengths ``uint16``, and flags ``bool``, about 30 bytes per packet, so three chunks (some
14M packets) fit comfortably in a few hundred MB. We reduce chunks one at a time and cache
each as a pickle; :func:`load_packets` concatenates the caches.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .logs import get_logger
from .netutil import dotted_quads_to_u32

log = get_logger(__name__)

PACKET_COLUMNS = ("ts", "src", "dst", "proto", "sport", "dport", "len", "plen", "ttl", "syn")


def _ips_to_u32(col: pd.Series) -> np.ndarray:
    cat = col.astype("category")
    levels = dotted_quads_to_u32(cat.cat.categories.astype(str))
    codes = cat.cat.codes.to_numpy()
    out = levels[np.where(codes < 0, 0, codes)]
    out[codes < 0] = 0
    return out


def reduce_features(psv: Path) -> pd.DataFrame:
    """Read one ``*.features.psv`` and return the typed packet table (IP packets only)."""
    df = pd.read_csv(
        psv, sep="|", low_memory=False, dtype={"ip.src": "category", "ip.dst": "category", "tcp.flags": "string"}
    )
    df = df[df["ip.src"].notna() & df["ip.dst"].notna()]
    num = lambda c: pd.to_numeric(df[c], errors="coerce")  # noqa: E731
    flags = df["tcp.flags"].fillna("")
    out = pd.DataFrame(
        {
            "ts": df["frame.time_epoch"].astype("float64").to_numpy(),
            "src": _ips_to_u32(df["ip.src"]),
            "dst": _ips_to_u32(df["ip.dst"]),
            "proto": num("ip.proto").fillna(0).astype("int8").to_numpy(),
            "sport": num("tcp.srcport").fillna(num("udp.srcport")).fillna(0).astype("uint16").to_numpy(),
            "dport": num("tcp.dstport").fillna(num("udp.dstport")).fillna(0).astype("uint16").to_numpy(),
            "len": num("frame.len").fillna(0).clip(upper=65535).astype("uint16").to_numpy(),
            "plen": num("tcp.len").fillna(num("udp.length")).fillna(0).clip(upper=65535).astype("uint16").to_numpy(),
            "ttl": num("ip.ttl").fillna(0).astype("uint8").to_numpy(),
            # pure SYN (no ACK): a new client-to-server connection attempt
            "syn": flags.str.fullmatch(r"0x0*002").fillna(False).to_numpy(dtype=bool),
        }
    )
    return out.sort_values("ts", kind="stable").reset_index(drop=True)


def cache_path(extract_dir: Path, chunk: str) -> Path:
    return extract_dir / f"{chunk}.packets.pkl"


def features_path(extract_dir: Path, chunk: str) -> Path:
    matches = sorted(extract_dir.glob(f"*{chunk}*.features.psv"))
    if not matches:
        raise FileNotFoundError(f"no *{chunk}*.features.psv in {extract_dir}")
    return matches[0]


def reduce_chunk(extract_dir: Path, chunk: str, *, force: bool = False) -> Path:
    """Reduce one chunk to its packet cache (skipped if the cache is fresh)."""
    out = cache_path(extract_dir, chunk)
    src = features_path(extract_dir, chunk)
    if out.exists() and not force and out.stat().st_mtime >= src.stat().st_mtime:
        log.debug("reduce_chunk_cached", chunk=chunk, cache=out.name)
        return out
    with log.timed("reduce_chunk", chunk=chunk, source=src.name) as t:
        df = reduce_features(src)
        df.to_pickle(out, protocol=5)
        t["rows"] = len(df)
        t["cache"] = out.name
    return out


def load_packets(extract_dir: Path, chunks: Sequence[str]) -> pd.DataFrame:
    """Concatenate the packet caches of the given chunks into one timeline."""
    frames = []
    for c in chunks:
        p = cache_path(extract_dir, c)
        if not p.exists():
            raise FileNotFoundError(f"packet cache missing for chunk '{c}': {p}")
        frames.append(pd.read_pickle(p))
    packets = pd.concat(frames, ignore_index=True)
    span = float(packets["ts"].max() - packets["ts"].min()) if len(packets) else 0.0
    log.info("packets_loaded", chunks=list(chunks), packets=len(packets), span_s=round(span, 1))
    return packets
