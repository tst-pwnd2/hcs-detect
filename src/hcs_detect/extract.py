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

"""Feature extraction from pcaps via ``tshark``.

We run three passes per pcap, each a single ``tshark -T fields`` invocation that writes a
``|``-separated table into ``out_dir``:

``<stem>.features.psv``  every IP packet: timing, endpoints, sizes, flags, and TTL, with no payload;
``<stem>.payload.psv``   the first ``payload_prefix_bytes`` of client-to-server TCP payload per
                         flow (bounded by relative ``tcp.seq``), toward the configured service ports only;
``<stem>.hello.psv``     TLS ClientHello components (version, ciphers, extensions, groups, EC
                         point formats, and JA4 where the build supports it) for fingerprinting.

tshark is an external dependency; :func:`find_tshark` locates it on PATH and then in the
macOS Wireshark.app bundle. Passes run in parallel across pcaps.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .logs import get_logger

log = get_logger(__name__)

MAC_TSHARK = "/Applications/Wireshark.app/Contents/MacOS/tshark"

FEATURE_FIELDS: tuple[str, ...] = (
    "frame.time_epoch",
    "ip.src",
    "ip.dst",
    "ip.proto",
    "ip.id",
    "ip.ttl",
    "tcp.srcport",
    "tcp.dstport",
    "udp.srcport",
    "udp.dstport",
    "frame.len",
    "tcp.len",
    "udp.length",
    "tcp.flags",
    "tcp.seq_raw",
)
PAYLOAD_FIELDS: tuple[str, ...] = (
    "frame.time_epoch",
    "ip.src",
    "ip.dst",
    "tcp.srcport",
    "tcp.dstport",
    "tcp.seq",
    "tcp.len",
    "tcp.payload",
)
HELLO_FIELDS: tuple[str, ...] = (
    "frame.time_epoch",
    "ip.src",
    "ip.dst",
    "tcp.dstport",
    "tls.handshake.version",
    "tls.handshake.ciphersuite",
    "tls.handshake.extension.type",
    "tls.handshake.extensions_supported_group",
    "tls.handshake.extensions_ec_point_format",
    "tls.handshake.ja4",
)


class TsharkNotFound(RuntimeError):
    pass


def find_tshark(explicit: str | None = None) -> str:
    """Resolve the tshark binary: the explicit path, then PATH, then the macOS Wireshark bundle."""
    if explicit:
        if Path(explicit).exists():
            return explicit
        raise TsharkNotFound(f"configured tshark not found: {explicit}")
    found = shutil.which("tshark")
    if found:
        return found
    if Path(MAC_TSHARK).exists():
        return MAC_TSHARK
    raise TsharkNotFound(
        "tshark not found on PATH or in /Applications/Wireshark.app; install Wireshark or set pipeline.tshark"
    )


def _set_literal(values: Iterable[int]) -> str:
    """Wireshark 4.x and later require commas in set literals: ``{443, 8443}``."""
    return "{" + ", ".join(str(v) for v in sorted(set(values))) + "}"


def _net_clause(field: str, cidrs: Sequence[str]) -> str:
    return "(" + " || ".join(f"{field}=={c}" for c in cidrs) + ")"


def payload_filter(cfg: Config) -> str:
    """Display filter for the payload pass: client-to-service TCP data within the first N bytes."""
    src_nets = (*cfg.topology.client_nets, *cfg.topology.peer_nets)
    return (
        f"tcp.len>0 && tcp.seq<={cfg.payload_prefix_bytes} && "
        f"{_net_clause('ip.src', src_nets)} && (tcp.dstport in {_set_literal(cfg.extract_ports)})"
    )


HELLO_FILTER = "tls.handshake.type==1"


@dataclass(frozen=True)
class Pass:
    suffix: str
    fields: tuple[str, ...]
    display_filter: str | None
    occurrence: str = "f"  # first value for multi-valued fields, or ...
    aggregator: str | None = None  # ... aggregate all values with this separator

    def command(self, tshark: str, pcap: Path) -> list[str]:
        cmd = [
            tshark,
            "-r",
            str(pcap),
            "-n",
            "-T",
            "fields",
            "-E",
            "separator=|",
            "-E",
            "header=y",
            "-E",
            f"occurrence={self.occurrence}",
        ]
        if self.aggregator:
            cmd += ["-E", f"aggregator={self.aggregator}"]
        if self.display_filter:
            cmd += ["-Y", self.display_filter]
        for f in self.fields:
            cmd += ["-e", f]
        return cmd


def passes_for(cfg: Config) -> tuple[Pass, ...]:
    return (
        Pass("features", FEATURE_FIELDS, None),
        Pass("payload", PAYLOAD_FIELDS, payload_filter(cfg)),
        Pass("hello", HELLO_FIELDS, HELLO_FILTER, occurrence="a", aggregator="-"),
    )


@dataclass(frozen=True)
class ExtractResult:
    pcap: Path
    outputs: dict[str, Path]
    rows: dict[str, int]
    errors: dict[str, str]


def _run_pass(tshark: str, p: Pass, pcap: Path, out_dir: Path) -> tuple[str, Path, int, str]:
    out = out_dir / f"{pcap.stem}.{p.suffix}.psv"
    err = out_dir / f"{pcap.stem}.{p.suffix}.err"
    log.debug("tshark_pass_start", pcap=pcap.name, stage=p.suffix, display_filter=p.display_filter)
    with log.timed("tshark_pass_done", pcap=pcap.name, stage=p.suffix) as t:
        with out.open("w") as fo, err.open("w") as fe:
            proc = subprocess.run(p.command(tshark, pcap), stdout=fo, stderr=fe, check=False)
        rows = max(sum(1 for _ in out.open()) - 1, 0)
        t["rows"] = rows
        t["returncode"] = proc.returncode
    stderr = err.read_text().strip()
    if stderr:
        log.warning("tshark_pass_stderr", pcap=pcap.name, stage=p.suffix, stderr=stderr.splitlines()[0])
    return p.suffix, out, rows, stderr


def extract(
    pcaps: Sequence[Path],
    out_dir: Path,
    cfg: Config,
    *,
    tshark: str | None = None,
    only: Sequence[str] | None = None,
    workers: int | None = None,
) -> list[ExtractResult]:
    """Run the extraction passes for every pcap, in parallel. Returns per-pcap results."""
    binary = find_tshark(tshark or cfg.tshark)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = [p for p in passes_for(cfg) if not only or p.suffix in only]
    jobs = [(p, pc) for pc in pcaps for p in selected]
    results: dict[Path, ExtractResult] = {pc: ExtractResult(pc, {}, {}, {}) for pc in pcaps}
    n_workers = workers or min(len(jobs), 8)
    log.info(
        "extract_start",
        pcaps=len(pcaps),
        stages=[p.suffix for p in selected],
        workers=n_workers,
        tshark=binary,
        out_dir=out_dir,
    )
    with log.timed("extract_done", pcaps=len(pcaps)) as t:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            outcomes = ex.map(lambda j: _run_pass(binary, j[0], j[1], out_dir), jobs)
            for (_p, pc), (suffix, out, rows, err) in zip(jobs, outcomes, strict=True):
                r = results[pc]
                r.outputs[suffix] = out
                r.rows[suffix] = rows
                if err:
                    r.errors[suffix] = err
        t["errors"] = sum(len(r.errors) for r in results.values())
    return list(results.values())
