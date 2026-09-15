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

"""Sessionization: orient packets into request and response, then group them by (host, service, port).

A session is all traffic between one host and one (service IP, port) pair, every
connection included. This is the unit a dead-drop poller cannot hide inside: a client
opening 442 tiny connections looks unremarkable per connection and glaring per session.

The orientation rules follow from the topology. A request is a packet from a client-net
or peer-net host to a service net, or from a client-net host to a peer-net host on a
configured service port; a response is the mirror image. Packets matching neither (the
router fabric, DNS infrastructure, peer-to-peer traffic) are dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config
from .logs import get_logger
from .netutil import in_any_cidr, u32_to_ip

log = get_logger(__name__)


@dataclass
class SessionTable:
    """``packets``: oriented packet rows with a ``sid``; ``sessions``: one row per session."""

    packets: pd.DataFrame
    sessions: pd.DataFrame

    def group(self, sid: int) -> pd.DataFrame:
        return self.packets[self.packets["sid"] == sid]


def orient(packets: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Keep request/response packets and relabel endpoints as host/svc/port/cport."""
    t = cfg.topology
    src = packets["src"].to_numpy()
    dst = packets["dst"].to_numpy()
    sport = packets["sport"].to_numpy()
    dport = packets["dport"].to_numpy()

    cli_s = in_any_cidr(src, t.client_nets)
    cli_d = in_any_cidr(dst, t.client_nets)
    peer_s = in_any_cidr(src, t.peer_nets)
    peer_d = in_any_cidr(dst, t.peer_nets)
    svc_s = in_any_cidr(src, t.service_nets.keys())
    svc_d = in_any_cidr(dst, t.service_nets.keys())
    ports = np.array(t.all_service_ports + tuple(s.port for s in cfg.covert), dtype=np.uint16)

    req = (svc_d & (cli_s | peer_s)) | (peer_d & cli_s & np.isin(dport, ports))
    rsp = (svc_s & (cli_d | peer_d)) | (peer_s & cli_d & np.isin(sport, ports))
    keep = req | rsp
    r = req[keep]

    out = pd.DataFrame(
        {
            "ts": packets["ts"].to_numpy()[keep],
            "host": np.where(r, src[keep], dst[keep]).astype(np.uint32),
            "svc": np.where(r, dst[keep], src[keep]).astype(np.uint32),
            "port": np.where(r, dport[keep], sport[keep]).astype(np.uint16),
            "cport": np.where(r, sport[keep], dport[keep]).astype(np.uint16),
            "len": packets["len"].to_numpy()[keep],
            "plen": packets["plen"].to_numpy()[keep],
            "syn": packets["syn"].to_numpy()[keep],
            "is_req": r,
        }
    )
    return out


def build_sessions(packets: pd.DataFrame, cfg: Config) -> SessionTable:
    """Orient, assign session ids, and attach ground-truth labels and service classes."""
    pk = orient(packets, cfg)
    pk["sid"] = pk.groupby(["host", "svc", "port"], sort=False).ngroup().astype(np.int32)
    sess = (
        pk.groupby("sid", sort=True)
        .agg(
            host=("host", "first"),
            svc=("svc", "first"),
            port=("port", "first"),
            pkts=("ts", "size"),
            first_ts=("ts", "min"),
            last_ts=("ts", "max"),
        )
        .reset_index()
    )
    sess["host_ip"] = sess["host"].map(u32_to_ip)
    sess["svc_ip"] = sess["svc"].map(u32_to_ip)
    sess["svc_class"] = [
        cfg.topology.service_class(s, int(p)) for s, p in zip(sess["svc_ip"], sess["port"], strict=True)
    ]
    sess["label"] = [
        cfg.label(h, s, int(p)) for h, s, p in zip(sess["host_ip"], sess["svc_ip"], sess["port"], strict=True)
    ]
    n_all = len(sess)
    sess = sess[sess["pkts"] >= cfg.min_session_packets].reset_index(drop=True)
    kept = pk["sid"].isin(sess["sid"])
    log.info(
        "sessions_built",
        packets_in=len(packets),
        packets_oriented=len(pk),
        packets_kept=int(kept.sum()),
        sessions=n_all,
        sessions_kept=len(sess),
        min_session_packets=cfg.min_session_packets,
        labels=sess["label"].value_counts().to_dict(),
    )
    pk = pk[kept].reset_index(drop=True)
    return SessionTable(packets=pk, sessions=sess)
