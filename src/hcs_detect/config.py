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

"""Scenario configuration: topology, ground-truth covert sessions, and pipeline settings.

Everything specific to one capture lives here rather than in code, namely which subnets
host clients, where the services are, and which (host, service, port) sessions are the
covert ones. The detectors themselves never see an IP literal.

A config is a TOML file; ``configs/cp3_spotcheck.toml`` is a complete example.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

from .netutil import cidr_to_net_mask, ip_to_u32


class ConfigError(ValueError):
    """Raised when a scenario config is missing or malformed."""


@dataclass(frozen=True)
class CovertSpec:
    """One ground-truth covert session: ``host`` talking to ``service`` (IP or CIDR) on ``port``."""

    name: str
    host: str
    service: str
    port: int
    role: str = ""

    def matches(self, host: str, service_ip: str, port: int) -> bool:
        if host != self.host or port != self.port:
            return False
        net, mask = cidr_to_net_mask(self.service)
        return (ip_to_u32(service_ip) & mask) == net


@dataclass(frozen=True)
class Topology:
    """Where hosts and services live.

    ``client_nets``      subnets that may host covert clients *and* benign neighbors.
    ``peer_nets``        subnets hosting covert peers (bobs) and cover services; hosts here
                         may also be clients of the service nets.
    ``service_nets``     CIDR to service class name (e.g., ``"10.5.5.0/24" = "minio"``).
    ``peer_service_ports`` port to class name for services hosted on ``peer_nets``; an
                         unlisted port is classed ``"server:<port>"``.
    """

    client_nets: tuple[str, ...]
    peer_nets: tuple[str, ...]
    service_nets: Mapping[str, str]
    peer_service_ports: Mapping[int, str]

    def service_class(self, service_ip: str, port: int) -> str:
        u = ip_to_u32(service_ip)
        for cidr, name in self.service_nets.items():
            net, mask = cidr_to_net_mask(cidr)
            if (u & mask) == net:
                return name
        return self.peer_service_ports.get(port, f"server:{port}")

    @property
    def all_service_ports(self) -> tuple[int, ...]:
        return tuple(sorted(self.peer_service_ports))


@dataclass(frozen=True)
class Config:
    name: str
    chunks: tuple[str, ...]
    topology: Topology
    covert: tuple[CovertSpec, ...]
    min_session_packets: int = 40
    payload_prefix_bytes: int = 8192
    first_segment_bytes: int = 64
    payload_ports: tuple[int, ...] = ()
    tshark: str | None = None
    source: Path | None = field(default=None, compare=False)

    def label(self, host: str, service_ip: str, port: int) -> str:
        """Ground-truth label for a session: the covert spec's name, or ``"benign"``."""
        for spec in self.covert:
            if spec.matches(host, service_ip, port):
                return spec.name
        return "benign"

    @property
    def covert_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.covert)

    @property
    def extract_ports(self) -> tuple[int, ...]:
        """Destination ports whose client-to-server payload is captured for the FEP test."""
        ports = set(self.payload_ports) | set(self.topology.peer_service_ports) | {s.port for s in self.covert}
        return tuple(sorted(ports))


def _require(d: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in d:
        raise ConfigError(f"missing required key '{key}' in [{where}]")
    return d[key]


def load_config(path: str | Path) -> Config:
    """Parse and validate a scenario TOML file."""
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config not found: {p}")
    with p.open("rb") as fh:
        raw = tomllib.load(fh)

    scen = raw.get("scenario", {})
    topo_raw = _require(raw, "topology", "topology")
    topology = Topology(
        client_nets=tuple(_require(topo_raw, "client_nets", "topology")),
        peer_nets=tuple(topo_raw.get("peer_nets", ())),
        service_nets=dict(topo_raw.get("service_nets", {})),
        peer_service_ports={int(k): str(v) for k, v in topo_raw.get("peer_service_ports", {}).items()},
    )
    for cidr in (*topology.client_nets, *topology.peer_nets, *topology.service_nets):
        cidr_to_net_mask(cidr)  # validates

    covert = tuple(
        CovertSpec(
            name=_require(c, "name", "covert"),
            host=_require(c, "host", "covert"),
            service=_require(c, "service", "covert"),
            port=int(_require(c, "port", "covert")),
            role=str(c.get("role", "")),
        )
        for c in raw.get("covert", [])
    )
    pipe = raw.get("pipeline", {})
    return Config(
        name=str(scen.get("name", p.stem)),
        chunks=tuple(scen.get("chunks", ())),
        topology=topology,
        covert=covert,
        min_session_packets=int(pipe.get("min_session_packets", 40)),
        payload_prefix_bytes=int(pipe.get("payload_prefix_bytes", 8192)),
        first_segment_bytes=int(pipe.get("first_segment_bytes", 64)),
        payload_ports=tuple(int(x) for x in pipe.get("payload_ports", ())),
        tshark=pipe.get("tshark"),
        source=p,
    )
