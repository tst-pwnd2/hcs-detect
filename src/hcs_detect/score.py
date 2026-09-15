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

"""Drive the detectors over every session and return one scored row per session."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .baseline import build_baselines
from .config import Config
from .detectors import REGISTRY, ClassBaseline, SessionContext, compute_all
from .ingest import load_packets
from .logs import get_logger
from .payload import PayloadIndex
from .sessions import SessionTable, build_sessions

log = get_logger(__name__)

META_COLUMNS = ("sid", "host", "service", "svc_class", "label", "pkts")


def make_context(
    g: pd.DataFrame, row: pd.Series, baseline: ClassBaseline | None, payload: PayloadIndex
) -> SessionContext:
    g = g.sort_values("ts", kind="stable")
    key = (int(row["host"]), int(row["svc"]), int(row["port"]))
    return SessionContext(
        ts=g["ts"].to_numpy(),
        is_req=g["is_req"].to_numpy(),
        syn=g["syn"].to_numpy(),
        plen=g["plen"].to_numpy(),
        len=g["len"].to_numpy(),
        cport=g["cport"].to_numpy(),
        svc_class=str(row["svc_class"]),
        baseline=baseline,
        first_segments=payload.first_segments.get(key, []),
        payload=payload.prefixes.get(key, b""),
    )


def score_table(table: SessionTable, payload: PayloadIndex) -> pd.DataFrame:
    """Score every session in the table. Baselines are built from its benign sessions."""
    baselines = build_baselines(table)
    groups = table.packets.groupby("sid", sort=False)
    rows = []
    with log.timed("sessions_scored", detectors=len(REGISTRY)) as t:
        for _, row in table.sessions.iterrows():
            g = groups.get_group(int(row["sid"]))
            ctx = make_context(g, row, baselines.get(row["svc_class"]), payload)
            scores = compute_all(ctx)
            rows.append(
                {
                    "sid": int(row["sid"]),
                    "host": row["host_ip"],
                    "service": f"{row['svc_ip']}:{int(row['port'])}",
                    "svc_class": row["svc_class"],
                    "label": row["label"],
                    "pkts": int(row["pkts"]),
                    **scores,
                }
            )
        cols = list(META_COLUMNS) + [d.name for d in REGISTRY]
        out = pd.DataFrame(rows, columns=cols).sort_values(["label", "svc_class", "host"]).reset_index(drop=True)
        t["sessions"] = len(out)
        t["covert"] = int((out["label"] != "benign").sum())
        t["benign"] = int((out["label"] == "benign").sum())
    return out


def run_detect(extract_dir: Path, cfg: Config) -> tuple[pd.DataFrame, SessionTable]:
    """Load the packet caches, sessionize, score. Returns (scores, session table)."""
    with log.timed("detect_done", scenario=cfg.name, extract_dir=extract_dir):
        packets = load_packets(extract_dir, cfg.chunks)
        table = build_sessions(packets, cfg)
        del packets
        payload = PayloadIndex.load(extract_dir, first_segment_bytes=cfg.first_segment_bytes)
        scores = score_table(table, payload)
    return scores, table
