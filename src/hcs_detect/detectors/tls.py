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

"""TLS ClientHello fingerprinting (JA3), built from tshark's handshake components.

Recent tshark builds expose the components but leave ``tls.handshake.ja3`` empty, so we
assemble the fingerprint here per the JA3 definition: ``version,ciphers,extensions,
groups,ec_point_formats`` with GREASE values removed, then MD5 hashed. Note that this is
a cross-session consistency check rather than a per-session detector. The question it
answers is whether a covert client presents the same fingerprint as the benign clients
of the same service, and it is evaluable only when the capture includes benign session
establishment.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

GREASE = {0x0A0A + 0x1010 * i for i in range(16)}


def _ints(field: str) -> list[int]:
    """Parse tshark's ``-``-aggregated values; tokens are ``0x`` hex or plain decimal."""
    out = []
    for tok in str(field or "").split("-"):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok, 0))
        except ValueError:
            try:
                out.append(int(tok, 10))
            except ValueError:
                continue
    return out


@dataclass(frozen=True)
class Hello:
    ja3: str
    ja3_full: str
    shape: str  # human-readable "v771 15cs 12ext"


def ja3_from_components(version: str, ciphers: str, extensions: str, groups: str, ec_formats: str) -> Hello:
    ver = _ints(version)[:1]
    cs = [c for c in _ints(ciphers) if c not in GREASE]
    ex = [e for e in _ints(extensions) if e not in GREASE]
    gr = [g for g in _ints(groups) if g not in GREASE]
    pf = _ints(ec_formats)
    full = ",".join(
        [
            str(ver[0]) if ver else "",
            "-".join(map(str, cs)),
            "-".join(map(str, ex)),
            "-".join(map(str, gr)),
            "-".join(map(str, pf)),
        ]
    )
    return Hello(hashlib.md5(full.encode()).hexdigest(), full, f"v{ver[0] if ver else '?'} {len(cs)}cs {len(ex)}ext")


def load_hellos(extract_dir: Path) -> pd.DataFrame:
    """All ClientHellos in the extract dir with JA3 attached; empty frame if none."""
    files = [p for p in sorted(extract_dir.glob("*.hello.psv")) if p.stat().st_size > 0]
    if not files:
        return pd.DataFrame(columns=["ts", "src", "dst", "port", "ja3", "ja3_full", "shape", "ja4"])
    df = pd.concat([pd.read_csv(f, sep="|", dtype=str) for f in files], ignore_index=True).fillna("")
    hellos = [
        ja3_from_components(v, c, e, g, p)
        for v, c, e, g, p in zip(  # noqa: B905 (columns come from one frame)
            df["tls.handshake.version"],
            df["tls.handshake.ciphersuite"],
            df["tls.handshake.extension.type"],
            df["tls.handshake.extensions_supported_group"],
            df["tls.handshake.extensions_ec_point_format"],
        )
    ]
    return pd.DataFrame(
        {
            "ts": pd.to_numeric(df["frame.time_epoch"], errors="coerce"),
            "src": df["ip.src"],
            "dst": df["ip.dst"],
            "port": pd.to_numeric(df["tcp.dstport"], errors="coerce").fillna(0).astype(int),
            "ja3": [h.ja3 for h in hellos],
            "ja3_full": [h.ja3_full for h in hellos],
            "shape": [h.shape for h in hellos],
            "ja4": df.get("tls.handshake.ja4", pd.Series([""] * len(df))),
        }
    )
