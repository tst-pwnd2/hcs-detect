# hcs-detect

hcs-detect is a passive detector for hidden communication systems, namely RACE-style
covert channels, in packet captures. Given a pcap and a short TOML describing the
topology, it sessionizes the traffic, scores every (host, service) session on a registry
of detectors, and ranks the known covert sessions against the benign population.

We built it for the PWND CP3 challenge problems. The detectors and the evaluation frame
come from our hidden communication systems detection methodology, and we validated them
on the CP3 spot-check capture (see the project documents `hcs-detection-methodology.md`
and `hcs-detection-spotcheck-results.md`).

## What it finds, and how

Each class of channel gives itself away differently, and a different layer of the
detector registry catches it.

| Channel type | What gives it away | Detector layer |
|---|---|---|
| Dead-drop sender polling on a timer (Skyhook PublicUser, one curl connection per poll) | regular inter-request intervals | `beaconing`: `mode_frac`, `cv_iat`, `spec_peak` |
| Dead-drop receiver inside keep-alive connections (Skyhook AccountHolder, AWS SDK) | a packet-size distribution unlike the service's real users | `divergence`: `kl_len`, `ks_len`, `kl_iat` |
| Social-media poster and poller (Mastodon) | thousands of tiny connections | `structure` and `divergence` |
| Fully-encrypted tunnel (obfs4) | random first bytes, i.e., no protocol magic and a popcount near 4 bits per byte | `fep`: `fep_magic`, `fep_popcount`, `fep_entropy` |
| TLS mimicry tunnel (WebTunnel) | its sparseness alone; TLS fingerprint consistency (JA3) is the intended detector | `structure` (weak) and `tls` |

The unit of analysis is the session, that is, all connections between one host and one
(service IP, port) pair. A poller opening 442 tiny connections is unremarkable per
connection and glaring per session.

## Install

The project is managed with [uv](https://docs.astral.sh/uv/). It requires `uv` and, for
the extraction stage, tshark (Wireshark 4.x or later). Note that `uv` fetches the pinned
interpreter (`.python-version`) itself, so the system Python is irrelevant.

```bash
uv sync                      # creates .venv and installs the dependencies, the package (editable), and the dev group
uv run hcs-detect --version
uv run hcs-detect detectors  # lists the registry
```

Everything below can be run as `uv run <cmd>` without activating the venv; alternatively,
activate once with `source .venv/bin/activate` and drop the prefix.

For development:

```bash
uv run pytest                # or: uv run python -m unittest discover -s tests
uv run ruff check .          # lint
uv add <package>             # add a runtime dependency (updates pyproject.toml and uv.lock)
uv add --dev <package>       # add a development dependency
uv lock --upgrade            # refresh the lockfile deliberately
```

`uv.lock` is committed, so `uv sync` reproduces the exact environment on any machine.

## Quick start

```bash
# one shot: extract, reduce, facts, detect
uv run hcs-detect run -c configs/cp3_spotcheck.toml -o extract/ captures/*.pcap

# or stage by stage
uv run hcs-detect extract -c configs/cp3_spotcheck.toml -o extract/ captures/*.pcap
uv run hcs-detect reduce  -c configs/cp3_spotcheck.toml -e extract/
uv run hcs-detect facts   -c configs/cp3_spotcheck.toml -e extract/   # read this before trusting anything
uv run hcs-detect detect  -c configs/cp3_spotcheck.toml -e extract/ -o results/
```

`detect` prints a console summary and writes three files to the output directory:
`<scenario>.sessions.csv`, which holds every session with every detector value (the raw
material); `<scenario>.ranks.csv`, which holds each covert session's percentile rank among
the benign sessions; and `<scenario>.report.md`, which holds the ranked tables per layer,
the decisive detectors, the baselines, and the caveats.

## Pipeline

The pipeline has four stages.

1. `extract` runs tshark and writes three tables per pcap: `*.features.psv` (every IP
   packet: timing, endpoints, sizes, flags, TTL), `*.payload.psv` (the first 8 KB of
   client-to-server payload per flow), and `*.hello.psv` (TLS ClientHello components for
   JA3).
2. `reduce` turns each features table into an all-numeric packet cache,
   `<chunk>.packets.pkl`, at roughly 30 bytes per packet.
3. `facts` reports chunk contiguity, the TTL-based vantage check, and a sanity figure for
   the naive duplicate heuristic.
4. `detect` sessionizes, builds the benign baselines, scores, ranks, and reports.

On memory: IPs stay `uint32`, ports and lengths `uint16`, and flags `bool`. Fourteen
million packets fit in about 450 MB, and the whole `detect` stage on the spot-check
capture runs in about 6 s on a 4 GB VM. Nothing ever materializes a per-packet Python
string.

## Logging

Diagnostics and results are kept apart. Every stage emits structured log records, that
is, an event name with key-value fields, and these go to stderr. The report, the facts
rendering, and the detector listing are the program's output, and these go to stdout.
As such, `uv run hcs-detect detect ... > report.txt` captures the report alone, and
`2> run.log` captures the diagnostics alone.

Two renderings of the same records are available. The default is a key=value console
line for reading by eye:

```
16:02:11 INFO    ingest       reduce_chunk seconds=17.4 chunk=ens60 source=ixp-router_ens60.features.psv rows=4818581
16:02:31 INFO    sessions     sessions_built packets_in=14103107 packets_kept=10853964 sessions=118 sessions_kept=91 ...
16:02:36 INFO    score        sessions_scored seconds=4.9 detectors=18 sessions=91 covert=6 benign=85
```

With `--log-format json` each record is one JSON object per line, suitable for a log
pipeline or for `jq`:

```
{"ts":"2026-09-15T21:02:36.412+00:00","level":"INFO","logger":"hcs_detect.score","event":"sessions_scored","seconds":4.9,"detectors":18,"sessions":91,"covert":6,"benign":85}
```

`--log-level DEBUG` adds per-pass and cache-hit detail; `-q` (or `--log-level WARNING`)
leaves only warnings and errors, e.g., a chunk boundary that is not contiguous, a TTL
distribution that suggests double capture, or a missing payload table. Timed events carry
a `seconds` field, and a stage that fails still emits its event with `ok=false`, so a
failed run leaves a complete record behind.

Note that the library itself never configures logging (the usual convention); the
command line does so once at startup. Code embedding `hcs_detect` can call
`hcs_detect.logs.configure(level, fmt, stream)` or attach its own handler to the
`hcs_detect` logger. Exit codes are 0 on success, 1 on a failed stage, and 2 on a usage
error such as a missing config or pcap.

## Configuration

Everything specific to one capture is in the TOML, never in code:

```toml
[scenario]
name = "cp3_spotcheck_1"
chunks = ["ens60", "ens61", "ens62"]        # filename tags of the capture chunks

[topology]
client_nets = ["10.1.0.0/16"]               # may host covert clients and benign neighbours alike
peer_nets   = ["10.2.0.0/24"]               # covert peers (bobs) and the cover services

[topology.service_nets]                     # CIDR to service class
"10.5.5.0/24" = "minio"
"10.3.3.0/24" = "mastodon"

[topology.peer_service_ports]               # services hosted on peer_nets, by port
443 = "https"
6697 = "irc"

[[covert]]                                  # ground truth; list both ends of every channel
name = "skyhook"
host = "10.1.1.4"
service = "10.5.5.0/24"                     # an IP or a CIDR
port = 8443
```

The service class drives the baseline: a session's divergence is measured against the
benign users of the same service (that is to say, we model the cover, not the covert).
Baseline purity therefore matters. A covert receiver left unlabelled contaminates its
whole class.

## Adding a detector

A detector is a pure function of a `SessionContext`, registered with its layer and the
direction in which covert sessions are extreme. Nothing else changes; ranking, the
report, and `hcs-detect detectors` all read the registry.

```python
from hcs_detect.detectors import SessionContext, register


@register("req_burstiness", "beaconing", "low")
def req_burstiness(ctx: SessionContext) -> float:
    """Fano factor of request counts in 10 s bins (low = clocked)."""
    ...
```

Use `rank=False` for context metrics (sample sizes, size-biased statistics) that should be
reported but never ranked.

## Evaluation

With a handful of covert sessions, ROC curves are theatre. hcs-detect reports each covert
session's percentile rank among the benign sessions on each detector; 100 means more
covert-like than every benign session. The `--within-class` flag ranks against the benign
sessions of the same service class only. Note that these ranks are evidence, and not
detection rates.

## Caveats

The report carries these caveats, and they bear repeating here.

- The `structure` detectors overstate separation when the benign models hold persistent
  connections. This was true of the spot-check range and is not true of real browsers.
- The `fep` detectors need each flow's first data segment. Sessions established before
  the capture window have none and score `NaN`.
- JA3 consistency is evaluable only when benign clients establish TLS sessions inside the
  capture.
- Read the output of `hcs-detect facts` first. Capture files named after interfaces may be
  time-contiguous chunks of one capture, and a router tap may or may not see each packet
  twice; the TTL distribution settles both questions.

## Layout

```
src/hcs_detect/
  config.py        TOML to Config, Topology, CovertSpec
  netutil.py       uint32 IP helpers and vectorized CIDR membership
  extract.py       tshark passes and display filters
  ingest.py        PSV to typed packet cache
  sessions.py      orientation, session ids, labels
  payload.py       client-to-server prefixes and first segments
  baseline.py      benign per-class baselines
  detectors/       the registry, with structure, beaconing, divergence, fep, and tls
  score.py         drives the detectors over the sessions
  evaluate.py      percentile ranks and baseline summaries
  report.py        console, Markdown, and CSV output
  logs.py          structured logging: event loggers, console and JSON renderers
  facts.py         capture sanity checks
  cli.py           the hcs-detect command
configs/           scenario TOMLs
tests/             the test suite (uv run pytest)
pyproject.toml     PEP 621 metadata, the hatchling build, and the uv dependency groups
uv.lock            the pinned, reproducible environment
.python-version    the interpreter uv provisions for the venv
```

## License

No license is being offered to the source code beyond that stipulated in
Agreement HR001125CE021.
