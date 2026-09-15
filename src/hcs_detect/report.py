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

"""Human-readable output: console tables and a Markdown report."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import Config
from .detectors import by_layer
from .evaluate import benign_summary, decisive, rank_covert
from .logs import get_logger

log = get_logger(__name__)


def _fmt(df: pd.DataFrame, digits: int = 3) -> str:
    with pd.option_context("display.width", 250, "display.max_columns", 60, "display.max_rows", 500):
        return df.round(digits).to_string(index=False)


def console_report(scores: pd.DataFrame, cfg: Config, *, within_class: bool = False) -> str:
    n_cov = int((scores["label"] != "benign").sum())
    n_ben = int((scores["label"] == "benign").sum())
    ranks = rank_covert(scores, within_class=within_class)
    parts = [f"scenario: {cfg.name}   sessions: {len(scores)}  (covert {n_cov}, benign {n_ben})", ""]
    parts += ["=== covert sessions: raw detector values ===", _fmt(scores[scores["label"] != "benign"]), ""]
    parts += [
        "=== percentile rank among benign "
        f"({'same service class' if within_class else 'all benign sessions'}; 100 = beyond every benign) ===",
        _fmt(ranks, 1),
        "",
    ]
    parts += ["=== decisive detectors (100th percentile) per covert session ==="]
    for k, v in decisive(ranks).items():
        parts.append(f"  {k}: {', '.join(v) if v else 'none'}")
    parts += ["", "=== benign baseline by service class ===", _fmt(benign_summary(scores))]
    return "\n".join(parts)


def markdown_report(scores: pd.DataFrame, cfg: Config, hellos: pd.DataFrame | None = None) -> str:
    ranks = rank_covert(scores)
    cov = scores[scores["label"] != "benign"]
    n_ben = int((scores["label"] == "benign").sum())
    layers = by_layer()
    md = [
        f"# hcs-detect report: {cfg.name}",
        "",
        f"Sessions scored: **{len(scores)}** ({len(cov)} covert, {n_ben} benign). "
        "Percentile rank = share of benign sessions the covert session is more covert-like than, "
        "in the detector's declared direction; **100** = beyond every benign session.",
        "",
    ]
    md += ["## Percentile ranks (all benign as the reference population)", ""]
    for layer, dets in layers.items():
        dets = [d for d in dets if d.rank]
        if not dets:
            continue
        names = [d.name for d in dets]
        cols = ["label", "host", "service"] + names
        sub = ranks[cols].copy()
        header = (
            "| "
            + " | ".join(
                ["channel", "host", "service"]
                + [f"{d.name} ({'high' if d.direction == 'high' else 'low'})" for d in dets]
            )
            + " |"
        )
        md += [f"### {layer}", "", header, "|" + "---|" * (len(cols))]
        for _, r in sub.iterrows():
            cells = [str(r["label"]), str(r["host"]), str(r["service"])] + [
                ("**100**" if (not pd.isna(r[n]) and r[n] >= 100) else ("n/a" if pd.isna(r[n]) else f"{r[n]:.0f}"))
                for n in names
            ]
            md.append("| " + " | ".join(cells) + " |")
        md.append("")
    md += ["## Decisive detectors per covert session", ""]
    for k, v in decisive(ranks).items():
        md.append(f"- **{k}**: {', '.join(f'`{x}`' for x in v) if v else '_none at the 100th percentile_'}")
    md += ["", "## Benign baseline (per service class)", "", "```", _fmt(benign_summary(scores)), "```", ""]
    if hellos is not None and len(hellos):
        md += [
            "## TLS ClientHello fingerprints (JA3)",
            "",
            f"{len(hellos)} ClientHellos observed. Fingerprint consistency is only evaluable when benign clients "
            "also establish sessions inside the capture window.",
            "",
            "```",
            _fmt(hellos.groupby(["label", "ja3", "shape"]).size().reset_index(name="hellos")),
            "```",
            "",
        ]
    md += [
        "## Caveats",
        "",
        "- Percentile ranks are not detection rates; with few covert sessions they are the honest statistic.",
        "- `structure` detectors overstate separation when benign models hold persistent connections "
        "(see spot-check results).",
        "- `divergence` detectors depend on baseline purity: every covert endpoint must be labelled in the config.",
        "- `fep` detectors need the first data segment of each flow; "
        "sessions established before the capture have none.",
        "",
    ]
    return "\n".join(md)


def write_outputs(
    scores: pd.DataFrame, cfg: Config, out_dir: Path, hellos: pd.DataFrame | None = None
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "scores": out_dir / f"{cfg.name}.sessions.csv",
        "ranks": out_dir / f"{cfg.name}.ranks.csv",
        "report": out_dir / f"{cfg.name}.report.md",
    }
    scores.to_csv(paths["scores"], index=False)
    rank_covert(scores).to_csv(paths["ranks"], index=False)
    paths["report"].write_text(markdown_report(scores, cfg, hellos))
    log.info("outputs_written", **{k: str(v) for k, v in paths.items()})
    return paths
