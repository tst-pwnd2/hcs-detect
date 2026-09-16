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

"""Detector registry.

A :class:`Detector` is a pure function from a :class:`SessionContext` to a float, tagged
with the observation layer it belongs to and the direction in which covert sessions are
expected to be extreme. Ranking and reporting use only those tags. As such, adding a
detector is one ``register(...)`` call, and no other module changes.

The layers mirror the detection methodology's matrix:

``structure``   connection count and shape (cheap, and partly an artifact of the range;
                see the results document)
``beaconing``   regularity of request events, which is the dead-drop sender's signature
``divergence``  KL and KS distances of the packet-size and inter-arrival distributions from
                the benign users of the same service, which is the dead-drop receiver's
                signature and, in practice, the most broadly useful layer
``fep``         fully-encrypted-protocol tests on the first bytes of each flow, which is
                the obfs4 signature
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Direction = Literal["high", "low"]


@dataclass
class ClassBaseline:
    """Pooled benign statistics for one service class (see :mod:`hcs_detect.baseline`)."""

    len_hist: np.ndarray
    iat_hist: np.ndarray
    len_pool: np.ndarray
    n_sessions: int


@dataclass
class SessionContext:
    """Everything a detector may look at for one session. Arrays are packet-aligned and time-sorted."""

    ts: np.ndarray
    is_req: np.ndarray
    syn: np.ndarray
    plen: np.ndarray
    len: np.ndarray
    cport: np.ndarray
    svc_class: str
    baseline: ClassBaseline | None
    first_segments: list[bytes] = field(default_factory=list)
    payload: bytes = b""

    # ---- derived conveniences shared by several detectors ----
    @property
    def conns(self) -> int:
        return int(np.unique(self.cport).size)

    @property
    def duration(self) -> float:
        return float(self.ts[-1] - self.ts[0]) if len(self.ts) > 1 else 0.0

    def request_events(self, multi_conn_threshold: int = 6) -> np.ndarray:
        """Times of 'requests': new-connection SYNs when the session is many-connection,
        otherwise payload-bearing client-to-server packets. Sorted."""
        req = self.is_req
        if self.conns >= multi_conn_threshold:
            ev = self.ts[req & self.syn]
        else:
            ev = self.ts[req & (self.plen > 0)]
        return np.sort(ev)


@dataclass(frozen=True)
class Detector:
    name: str
    layer: str
    direction: Direction
    fn: Callable[[SessionContext], float]
    doc: str = ""
    rank: bool = True  # False = context metric (sample sizes, size-biased stats): reported, never ranked


REGISTRY: list[Detector] = []


def register(
    name: str, layer: str, direction: Direction, doc: str = "", *, rank: bool = True
) -> Callable[[Callable[[SessionContext], float]], Callable[[SessionContext], float]]:
    def deco(fn: Callable[[SessionContext], float]) -> Callable[[SessionContext], float]:
        REGISTRY.append(Detector(name, layer, direction, fn, doc or (fn.__doc__ or "").strip(), rank))
        return fn

    return deco


def rankable() -> list[Detector]:
    """Detectors that participate in percentile ranking (excludes context metrics)."""
    return [d for d in REGISTRY if d.rank]


def compute_all(ctx: SessionContext) -> dict[str, float]:
    out: dict[str, float] = {}
    for d in REGISTRY:
        try:
            out[d.name] = float(d.fn(ctx))
        except Exception:  # a detector failing must not sink the run; NaN is 'not computable'
            out[d.name] = float("nan")
    return out


def by_layer() -> dict[str, list[Detector]]:
    layers: dict[str, list[Detector]] = {}
    for d in REGISTRY:
        layers.setdefault(d.layer, []).append(d)
    return layers


# Import modules for their registration side effects (order = display order).
from . import structure, beaconing, divergence, fep  # noqa: E402, F401, I001  (registration order is display order)

__all__ = ["ClassBaseline", "SessionContext", "Detector", "REGISTRY", "register", "rankable", "compute_all", "by_layer"]
