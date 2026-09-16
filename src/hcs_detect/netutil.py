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

"""IPv4 helpers that stay in ``uint32``, so that tables of ten million rows never materialize Python strings."""

from __future__ import annotations

import ipaddress
from collections.abc import Iterable

import numpy as np

U32 = np.uint32


def ip_to_u32(ip: str) -> int:
    """Dotted quad to unsigned 32-bit integer."""
    return int(ipaddress.IPv4Address(ip))


def u32_to_ip(value: int) -> str:
    """Unsigned 32-bit integer to dotted quad."""
    return str(ipaddress.IPv4Address(int(value)))


def cidr_to_net_mask(cidr: str) -> tuple[int, int]:
    """``"10.1.0.0/16"`` to ``(network_u32, mask_u32)``. A bare host address is a /32."""
    net = ipaddress.IPv4Network(cidr, strict=False)
    return int(net.network_address), int(net.netmask)


def in_cidr(addrs: np.ndarray, cidr: str) -> np.ndarray:
    """Vectorized membership test of a ``uint32`` array against one CIDR."""
    net, mask = cidr_to_net_mask(cidr)
    return (addrs.astype(np.uint64) & np.uint64(mask)) == np.uint64(net)


def in_any_cidr(addrs: np.ndarray, cidrs: Iterable[str]) -> np.ndarray:
    """Vectorized membership test against any of several CIDRs."""
    out = np.zeros(len(addrs), dtype=bool)
    for c in cidrs:
        out |= in_cidr(addrs, c)
    return out


def dotted_quads_to_u32(values: Iterable[str]) -> np.ndarray:
    """Convert an iterable of dotted quads (e.g. categorical levels) to a ``uint32`` array."""
    return np.fromiter((ip_to_u32(v) for v in values), dtype=U32)
