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

"""Benign baselines per service class, i.e., the cover model the divergence detectors compare against.

We build one baseline per service class (``minio``, ``mastodon``, ``irc``, and so on) from
the sessions the ground truth labels ``benign``. Histograms are summed across sessions;
the KS pool is a capped per-session subsample, so that a single large download cannot
dominate it.

Note that purity matters. A covert receiver mislabelled as benign contaminates the
baseline for its whole service class, so the ground-truth config must list both ends of
every channel.
"""

from __future__ import annotations

import numpy as np

from .detectors import ClassBaseline
from .detectors.divergence import iat_hist, len_hist
from .logs import get_logger
from .sessions import SessionTable

log = get_logger(__name__)

POOL_PER_SESSION = 5_000


def build_baselines(table: SessionTable, *, seed: int = 0) -> dict[str, ClassBaseline]:
    rng = np.random.default_rng(seed)
    acc: dict[str, dict] = {}
    groups = table.packets.groupby("sid", sort=False)
    benign = table.sessions[table.sessions["label"] == "benign"]
    for _sid, row in benign.iterrows():
        g = groups.get_group(int(row["sid"]))
        cls = row["svc_class"]
        a = acc.setdefault(cls, {"len": 0, "iat": 0, "pool": [], "n": 0})
        ts = np.sort(g["ts"].to_numpy())
        lens = g["len"].to_numpy()
        a["len"] = a["len"] + len_hist(lens)
        a["iat"] = a["iat"] + iat_hist(ts)
        a["pool"].append(lens if lens.size <= POOL_PER_SESSION else rng.choice(lens, POOL_PER_SESSION, replace=False))
        a["n"] += 1
    if not acc:
        log.warning("baselines_empty", reason="no sessions labelled benign; divergence detectors will be NaN")
    log.info("baselines_built", classes={cls: a["n"] for cls, a in acc.items()})
    return {
        cls: ClassBaseline(
            len_hist=np.asarray(a["len"]),
            iat_hist=np.asarray(a["iat"]),
            len_pool=np.concatenate(a["pool"]) if a["pool"] else np.array([]),
            n_sessions=a["n"],
        )
        for cls, a in acc.items()
    }
