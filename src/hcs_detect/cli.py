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

"""``hcs-detect`` command-line interface.

hcs-detect extract  -c CONFIG -o EXTRACT_DIR PCAP [PCAP ...]   # tshark passes
hcs-detect reduce   -c CONFIG -e EXTRACT_DIR                    # typed packet caches
hcs-detect facts    -c CONFIG -e EXTRACT_DIR                    # contiguity and vantage checks
hcs-detect detect   -c CONFIG -e EXTRACT_DIR [-o OUT_DIR]       # score, rank, report
hcs-detect run      -c CONFIG -o EXTRACT_DIR PCAP [PCAP ...]    # all of the above
hcs-detect detectors                                            # list the registry

Diagnostics are structured log records on stderr (``--log-level``, ``--log-format``).
The report itself is the program's output and is written to stdout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import Config, ConfigError, load_config
from .detectors import REGISTRY, by_layer
from .detectors.tls import load_hellos
from .extract import TsharkNotFound, extract
from .facts import capture_facts
from .ingest import reduce_chunk
from .logs import configure, get_logger
from .report import console_report, write_outputs
from .score import run_detect

log = get_logger(__name__)

EXIT_OK, EXIT_FAILURE, EXIT_USAGE = 0, 1, 2


def emit(text: str) -> None:
    """Write primary output (a report, a listing) to stdout."""
    sys.stdout.write(text)
    if not text.endswith("\n"):
        sys.stdout.write("\n")
    sys.stdout.flush()


def _load(args: argparse.Namespace) -> Config | None:
    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        log.error("config_invalid", path=str(args.config), reason=str(e))
        return None
    log.debug("config_loaded", path=str(cfg.source), scenario=cfg.name, chunks=list(cfg.chunks), covert=len(cfg.covert))
    return cfg


def cmd_extract(args: argparse.Namespace) -> int:
    cfg = _load(args)
    if cfg is None:
        return EXIT_USAGE
    pcaps = [Path(p) for p in args.pcaps]
    missing = [str(p) for p in pcaps if not p.exists()]
    if missing:
        log.error("pcap_not_found", paths=missing)
        return EXIT_USAGE
    try:
        results = extract(pcaps, Path(args.out), cfg, tshark=args.tshark, only=args.only, workers=args.workers)
    except TsharkNotFound as e:
        log.error("tshark_not_found", reason=str(e))
        return EXIT_USAGE
    failed = [r for r in results if r.errors]
    for r in failed:
        log.error("extract_pcap_failed", pcap=r.pcap.name, stages=sorted(r.errors))
    return EXIT_FAILURE if failed else EXIT_OK


def cmd_reduce(args: argparse.Namespace) -> int:
    cfg = _load(args)
    if cfg is None:
        return EXIT_USAGE
    ext = Path(args.extract)
    try:
        for c in cfg.chunks:
            reduce_chunk(ext, c, force=args.force)
    except FileNotFoundError as e:
        log.error("features_missing", reason=str(e))
        return EXIT_FAILURE
    return EXIT_OK


def cmd_facts(args: argparse.Namespace) -> int:
    cfg = _load(args)
    if cfg is None:
        return EXIT_USAGE
    try:
        facts = capture_facts(Path(args.extract), cfg.chunks)
    except FileNotFoundError as e:
        log.error("packet_cache_missing", reason=str(e), hint="run `hcs-detect reduce` first")
        return EXIT_FAILURE
    emit(facts.render())
    return EXIT_OK


def cmd_detect(args: argparse.Namespace) -> int:
    cfg = _load(args)
    if cfg is None:
        return EXIT_USAGE
    ext = Path(args.extract)
    try:
        scores, _table = run_detect(ext, cfg)
    except FileNotFoundError as e:
        log.error("packet_cache_missing", reason=str(e), hint="run `hcs-detect reduce` first")
        return EXIT_FAILURE
    hellos = load_hellos(ext)
    if len(hellos):
        hellos["label"] = [
            cfg.label(s, d, int(p)) for s, d, p in zip(hellos["src"], hellos["dst"], hellos["port"], strict=True)
        ]
        log.info("client_hellos_loaded", hellos=len(hellos), covert=int((hellos["label"] != "benign").sum()))
    else:
        log.info("client_hellos_absent", effect="JA3 consistency cannot be evaluated")
    emit(console_report(scores, cfg, within_class=args.within_class))
    out_dir = Path(args.out) if args.out else ext
    write_outputs(scores, cfg, out_dir, hellos if len(hellos) else None)
    return EXIT_OK


def cmd_run(args: argparse.Namespace) -> int:
    rc = cmd_extract(args)
    if rc:
        return rc
    args.extract = args.out
    rc = cmd_reduce(args)
    if rc:
        return rc
    rc = cmd_facts(args)
    if rc:
        return rc
    args.out = args.report_out
    return cmd_detect(args)


def cmd_detectors(_args: argparse.Namespace) -> int:
    lines = []
    for layer, dets in by_layer().items():
        lines.append(f"[{layer}]")
        for d in dets:
            tag = "high" if d.direction == "high" else "low "
            note = "" if d.rank else " (context, not ranked)"
            lines.append(f"  {d.name:20s} {tag}{note:24s} {d.doc.splitlines()[0] if d.doc else ''}")
    lines.append(f"\n{len(REGISTRY)} detectors")
    emit("\n".join(lines))
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hcs-detect", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--version", action="version", version=f"hcs-detect {__version__}")
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="threshold for diagnostic records on stderr (default INFO)",
    )
    p.add_argument(
        "--log-format",
        default="console",
        choices=["console", "json"],
        help="key=value lines for people, or one JSON object per line for machines",
    )
    p.add_argument("-q", "--quiet", action="store_true", help="shorthand for --log-level WARNING")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_cfg(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("-c", "--config", required=True, help="scenario TOML")

    s = sub.add_parser("extract", help="run tshark feature passes over pcaps")
    add_cfg(s)
    s.add_argument("-o", "--out", required=True, help="extract dir for .psv tables")
    s.add_argument("pcaps", nargs="+")
    s.add_argument("--tshark")
    s.add_argument("--workers", type=int)
    s.add_argument("--only", nargs="+", choices=["features", "payload", "hello"], help="run only these passes")
    s.set_defaults(func=cmd_extract)

    s = sub.add_parser("reduce", help="build typed packet caches")
    add_cfg(s)
    s.add_argument("-e", "--extract", required=True)
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_reduce)

    s = sub.add_parser("facts", help="capture contiguity and vantage checks")
    add_cfg(s)
    s.add_argument("-e", "--extract", required=True)
    s.set_defaults(func=cmd_facts)

    s = sub.add_parser("detect", help="score sessions, rank covert against benign, write the report")
    add_cfg(s)
    s.add_argument("-e", "--extract", required=True)
    s.add_argument("-o", "--out", help="output dir (default: the extract dir)")
    s.add_argument(
        "--within-class", action="store_true", help="rank against benign sessions of the same service class only"
    )
    s.set_defaults(func=cmd_detect)

    s = sub.add_parser("run", help="extract, reduce, facts, detect in one run")
    add_cfg(s)
    s.add_argument("-o", "--out", required=True, help="extract dir")
    s.add_argument("pcaps", nargs="+")
    s.add_argument("--report-out", help="report dir (default: the extract dir)")
    s.add_argument("--tshark")
    s.add_argument("--workers", type=int)
    s.add_argument("--only", nargs="+", choices=["features", "payload", "hello"])
    s.add_argument("--force", action="store_true")
    s.add_argument("--within-class", action="store_true")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("detectors", help="list registered detectors")
    s.set_defaults(func=cmd_detectors)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure(level="WARNING" if args.quiet else args.log_level, fmt=args.log_format)
    log.debug("invoked", command=args.cmd, version=__version__)
    try:
        return int(args.func(args) or EXIT_OK)
    except KeyboardInterrupt:
        log.warning("interrupted", command=args.cmd)
        return 130
    except Exception:
        log.exception("unhandled_error", command=args.cmd)
        return EXIT_FAILURE


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
