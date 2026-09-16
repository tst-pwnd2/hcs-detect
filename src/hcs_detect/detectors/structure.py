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

"""Connection-structure detectors.

These are cheap and, against dead-drop pollers that open a fresh connection per poll,
often decisive. Note, however, the caveat in the results document: where the benign
models hold persistent connections, as they did in the spot-check range, these detectors
overstate the separation one would see against real browsers.
"""

from __future__ import annotations

import numpy as np

from . import SessionContext, register


@register("conns", "structure", "high")
def conns(ctx: SessionContext) -> float:
    """Distinct client ports = connections in the session."""
    return float(ctx.conns)


@register("pkts_per_conn", "structure", "low")
def pkts_per_conn(ctx: SessionContext) -> float:
    """Mean packets per connection; pollers are many-and-tiny."""
    return len(ctx.ts) / max(ctx.conns, 1)


@register("duration_s", "structure", "high")
def duration_s(ctx: SessionContext) -> float:
    """Session span in seconds."""
    return ctx.duration


@register("pps", "structure", "low")
def pps(ctx: SessionContext) -> float:
    """Packets per second over the session span; persistent tunnels are ultra-sparse."""
    return len(ctx.ts) / max(ctx.duration, 1.0)


@register("bytes_per_pkt", "structure", "low")
def bytes_per_pkt(ctx: SessionContext) -> float:
    """Mean frame size; control-heavy covert sessions run small."""
    return float(np.mean(ctx.len)) if len(ctx.len) else float("nan")
