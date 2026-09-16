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

"""Structured logging for hcs_detect.

A log record here is an event name plus key-value fields, e.g.,
``log.info("reduce_chunk", chunk="ens60", rows=4818581, seconds=17.4)``. The event name is
a constant; everything that varies travels in the fields. We provide two renderings of the
same record: a key=value console line for people and one JSON object per line for
machines, so that a run can be read by eye and parsed by a pipeline without changing the
code that emits the events.

Library modules get a logger with: func:`get_logger` and never configure handlers. The
command line configures the ``hcs_detect`` logger once via: func:`configure`; a library
user who never calls it sees nothing, which is the usual convention for libraries.

Diagnostics go to the log (on stderr by default). The program's primary output, namely the
report, is written to stdout by the CLI and is deliberately not a log record.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

ROOT = "hcs_detect"
_HANDLER_NAME = "hcs_detect.structured"


def _plain(value: Any) -> Any:
    """Coerce a field value to something both renderers can print."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if hasattr(value, "item") and callable(value.item):  # numpy scalars
        try:
            return value.item()
        except (TypeError, ValueError):
            return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class JsonFormatter(logging.Formatter):
    """One JSON object per line: ts, level, logger, event, then the fields."""

    def format(self, record: logging.LogRecord) -> str:
        obj: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        obj.update(_plain(getattr(record, "fields", {})))
        if record.exc_info:
            obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(obj, separators=(",", ":"), default=str)


class ConsoleFormatter(logging.Formatter):
    """``HH:MM:SS LEVEL logger event key=value ...`` for reading by eye."""

    @staticmethod
    def _render(value: Any) -> str:
        value = _plain(value)
        if isinstance(value, float):
            return f"{value:.4g}"
        if isinstance(value, str):
            return json.dumps(value) if (" " in value or not value) else value
        if isinstance(value, (list, dict)):
            return json.dumps(value, separators=(",", ":"))
        return str(value)

    def format(self, record: logging.LogRecord) -> str:
        ts = time.strftime("%H:%M:%S", time.localtime(record.created))
        fields = getattr(record, "fields", {})
        kv = " ".join(f"{k}={self._render(v)}" for k, v in fields.items())
        short = record.name.removeprefix(ROOT + ".") if record.name != ROOT else ROOT
        line = f"{ts} {record.levelname:7s} {short:12s} {record.getMessage()} {kv}".rstrip()
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


class EventLogger:
    """A thin wrapper over :class:`logging.Logger` that takes fields as keyword arguments."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    def _log(self, level: int, event: str, exc_info: Any = None, **fields: Any) -> None:
        if self._logger.isEnabledFor(level):
            self._logger.log(level, event, extra={"fields": fields}, exc_info=exc_info, stacklevel=3)

    def debug(self, event: str, **fields: Any) -> None:
        self._log(logging.DEBUG, event, **fields)

    def info(self, event: str, **fields: Any) -> None:
        self._log(logging.INFO, event, **fields)

    def warning(self, event: str, **fields: Any) -> None:
        self._log(logging.WARNING, event, **fields)

    def error(self, event: str, **fields: Any) -> None:
        self._log(logging.ERROR, event, **fields)

    def exception(self, event: str, **fields: Any) -> None:
        self._log(logging.ERROR, event, exc_info=True, **fields)

    @contextmanager
    def timed(self, event: str, level: int = logging.INFO, **fields: Any) -> Iterator[dict[str, Any]]:
        """Emit ``event`` on exit with a ``seconds`` field. The yielded dict collects result fields.

        Note that the event is emitted on exceptions too, with ``ok=False``, so a failed
        stage still leaves a timed record behind.
        """
        collected: dict[str, Any] = dict(fields)
        start = time.perf_counter()
        try:
            yield collected
        except BaseException:
            collected["ok"] = False
            self._log(level, event, seconds=round(time.perf_counter() - start, 3), **collected)
            raise
        self._log(level, event, seconds=round(time.perf_counter() - start, 3), **collected)


def get_logger(name: str) -> EventLogger:
    """Logger for a module; pass ``__name__``."""
    return EventLogger(logging.getLogger(name))


def configure(level: int | str = logging.INFO, fmt: str = "console", stream: TextIO | None = None) -> logging.Logger:
    """Install one handler on the ``hcs_detect`` logger. Idempotent; calling again replaces it."""
    root = logging.getLogger(ROOT)
    root.setLevel(level)
    root.propagate = False
    for h in list(root.handlers):
        if h.get_name() == _HANDLER_NAME:
            root.removeHandler(h)
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.set_name(_HANDLER_NAME)
    handler.setFormatter(JsonFormatter() if fmt == "json" else ConsoleFormatter())
    root.addHandler(handler)
    return root
