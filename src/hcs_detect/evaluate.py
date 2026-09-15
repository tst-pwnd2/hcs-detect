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

"""Turn scores into evidence: covert sessions ranked against the benign population.

With a handful of covert sessions, ROC and AUC would be theatre. The honest statistic is
each covert session's percentile rank among the benign sessions on each detector, in the
direction the detector declares covert-like; 100 means more extreme than every benign
session. :func:`benign_summary` gives the baseline quantiles those ranks are relative to.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .detectors import REGISTRY, Detector, rankable


def detector_columns() -> list[str]:
    return [d.name for d in REGISTRY]


def percentile_rank(value: float, benign: np.ndarray, direction: str) -> float:
    """Share (0 to 100) of benign values the covert value is more covert-like than."""
    b = benign[~np.isnan(benign)]
    if np.isnan(value) or b.size == 0:
        return float("nan")
    return float(100.0 * (np.mean(b < value) if direction == "high" else np.mean(b > value)))


def rank_covert(scores: pd.DataFrame, *, within_class: bool = False) -> pd.DataFrame:
    """One row per covert session, one column per detector, values = percentile rank.

    ``within_class=True`` ranks against benign sessions of the same service class only;
    the default ranks against the whole benign population.
    """
    benign = scores[scores["label"] == "benign"]
    covert = scores[scores["label"] != "benign"]
    out = []
    for _, c in covert.iterrows():
        pool = benign[benign["svc_class"] == c["svc_class"]] if within_class else benign
        row = {
            "label": c["label"],
            "host": c["host"],
            "service": c["service"],
            "svc_class": c["svc_class"],
            "n_benign": len(pool),
        }
        for d in rankable():
            row[d.name] = percentile_rank(c[d.name], pool[d.name].to_numpy(dtype=float), d.direction)
        out.append(row)
    return pd.DataFrame(out)


def benign_summary(scores: pd.DataFrame, quantiles=(0.1, 0.5, 0.9)) -> pd.DataFrame:
    """Per service class: quantiles and extremes of every detector over benign sessions."""
    benign = scores[scores["label"] == "benign"]
    cols = detector_columns()
    frames = []
    for cls, g in benign.groupby("svc_class"):
        q = g[cols].quantile(list(quantiles)).T
        q.columns = [f"p{int(x * 100)}" for x in quantiles]
        q["min"] = g[cols].min()
        q["max"] = g[cols].max()
        q["n"] = len(g)
        q.insert(0, "svc_class", cls)
        frames.append(q.reset_index().rename(columns={"index": "detector"}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def decisive(ranks: pd.DataFrame, threshold: float = 100.0) -> dict[str, list[str]]:
    """For each covert session, the detectors on which it is at/above ``threshold`` percentile."""
    out = {}
    for _, r in ranks.iterrows():
        hits = [d.name for d in rankable() if not np.isnan(r[d.name]) and r[d.name] >= threshold]
        out[f"{r['label']} ({r['host']} -> {r['service']})"] = hits
    return out


def directions() -> dict[str, str]:
    return {d.name: d.direction for d in REGISTRY}


def layers() -> dict[str, str]:
    return {d.name: d.layer for d in REGISTRY}


__all__ = [
    "rank_covert",
    "benign_summary",
    "decisive",
    "percentile_rank",
    "detector_columns",
    "directions",
    "layers",
    "Detector",
]
