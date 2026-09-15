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

"""Structured logging: renderers, the event logger, and events emitted by the library."""

import io
import json
import logging
import unittest
from pathlib import Path

import numpy as np

from hcs_detect import logs
from hcs_detect.logs import ConsoleFormatter, JsonFormatter, configure, get_logger


def capture(fmt: str) -> tuple[io.StringIO, logging.Logger]:
    stream = io.StringIO()
    root = configure(level=logging.DEBUG, fmt=fmt, stream=stream)
    return stream, root


class RendererTests(unittest.TestCase):
    def test_json_line_carries_event_and_fields(self):
        stream, _ = capture("json")
        get_logger("hcs_detect.test").info("reduce_chunk", chunk="ens60", rows=4818581, cache=Path("/x/ens60.pkl"))
        obj = json.loads(stream.getvalue().strip())
        self.assertEqual(obj["event"], "reduce_chunk")
        self.assertEqual(obj["level"], "INFO")
        self.assertEqual(obj["logger"], "hcs_detect.test")
        self.assertEqual(obj["chunk"], "ens60")
        self.assertEqual(obj["rows"], 4818581)
        self.assertEqual(obj["cache"], "/x/ens60.pkl")  # Path is coerced to str
        self.assertTrue(obj["ts"].endswith("+00:00"))

    def test_json_coerces_numpy_and_containers(self):
        stream, _ = capture("json")
        get_logger("hcs_detect.test").info(
            "x", n=np.int64(7), share=np.float64(0.25), classes={"minio": 20}, tags=("a", "b")
        )
        obj = json.loads(stream.getvalue())
        self.assertEqual(obj["n"], 7)
        self.assertEqual(obj["share"], 0.25)
        self.assertEqual(obj["classes"], {"minio": 20})
        self.assertEqual(obj["tags"], ["a", "b"])

    def test_console_line_is_key_value(self):
        stream, _ = capture("console")
        get_logger("hcs_detect.ingest").warning(
            "payload_missing", extract_dir="/tmp/e", effect="fep detectors will be NaN"
        )
        line = stream.getvalue().strip()
        self.assertIn("WARNING", line)
        self.assertIn("ingest", line)
        self.assertIn("payload_missing", line)
        self.assertIn("extract_dir=/tmp/e", line)
        self.assertIn('effect="fep detectors will be NaN"', line)  # values with spaces are quoted

    def test_exception_is_rendered_in_both_formats(self):
        for fmt, marker in (("json", '"exception":'), ("console", "Traceback")):
            stream, _ = capture(fmt)
            try:
                raise ValueError("boom")
            except ValueError:
                get_logger("hcs_detect.test").exception("unhandled_error", command="detect")
            self.assertIn(marker, stream.getvalue())
            self.assertIn("boom", stream.getvalue())

    def test_formatters_ignore_records_without_fields(self):
        rec = logging.LogRecord("hcs_detect.x", logging.INFO, __file__, 1, "plain", (), None)
        self.assertIn("plain", ConsoleFormatter().format(rec))
        self.assertEqual(json.loads(JsonFormatter().format(rec))["event"], "plain")


class EventLoggerTests(unittest.TestCase):
    def test_timed_adds_seconds_and_collected_fields(self):
        stream, _ = capture("json")
        log = get_logger("hcs_detect.test")
        with log.timed("stage_done", stage="reduce") as t:
            t["rows"] = 3
        obj = json.loads(stream.getvalue())
        self.assertEqual(obj["event"], "stage_done")
        self.assertEqual(obj["stage"], "reduce")
        self.assertEqual(obj["rows"], 3)
        self.assertIsInstance(obj["seconds"], float)
        self.assertNotIn("ok", obj)

    def test_timed_marks_failure_and_reraises(self):
        stream, _ = capture("json")
        log = get_logger("hcs_detect.test")
        with self.assertRaises(RuntimeError):
            with log.timed("stage_done"):
                raise RuntimeError("x")
        self.assertFalse(json.loads(stream.getvalue())["ok"])

    def test_level_threshold_is_respected(self):
        stream, _ = capture("json")
        configure(level="WARNING", fmt="json", stream=stream)
        log = get_logger("hcs_detect.test")
        log.info("hidden")
        log.warning("shown")
        events = [json.loads(line)["event"] for line in stream.getvalue().splitlines()]
        self.assertEqual(events, ["shown"])

    def test_configure_is_idempotent(self):
        s1, root = capture("json")
        s2, root2 = capture("json")
        self.assertIs(root, root2)
        self.assertEqual(sum(1 for h in root.handlers if h.get_name() == logs._HANDLER_NAME), 1)
        get_logger("hcs_detect.test").info("once")
        self.assertEqual(s1.getvalue(), "")  # the replaced handler no longer receives records
        self.assertEqual(len(s2.getvalue().splitlines()), 1)

    def test_library_stays_silent_without_configuration(self):
        root = logging.getLogger(logs.ROOT)
        for h in list(root.handlers):
            root.removeHandler(h)
        self.assertFalse(root.handlers)
        get_logger("hcs_detect.test").info("nobody_listens")  # must not raise or print


class LibraryEventTests(unittest.TestCase):
    def test_sessions_built_event_has_counts(self):
        from hcs_detect.config import load_config
        from hcs_detect.sessions import build_sessions
        from test_pipeline import MINI_TOML, synthetic_packets, write_cfg

        stream, _ = capture("json")
        build_sessions(synthetic_packets(), load_config(write_cfg(MINI_TOML)))
        events = {json.loads(line)["event"]: json.loads(line) for line in stream.getvalue().splitlines()}
        self.assertIn("sessions_built", events)
        e = events["sessions_built"]
        self.assertEqual(e["sessions_kept"], 3)
        self.assertEqual(e["labels"], {"benign": 2, "sky": 1})
        self.assertGreater(e["packets_in"], e["packets_kept"])  # the fabric packet was dropped

    def test_baselines_built_event_lists_classes(self):
        from hcs_detect.baseline import build_baselines
        from hcs_detect.config import load_config
        from hcs_detect.sessions import build_sessions
        from test_pipeline import MINI_TOML, synthetic_packets, write_cfg

        stream, _ = capture("json")
        build_baselines(build_sessions(synthetic_packets(), load_config(write_cfg(MINI_TOML))))
        events = {json.loads(line)["event"]: json.loads(line) for line in stream.getvalue().splitlines()}
        self.assertEqual(events["baselines_built"]["classes"], {"minio": 1, "irc": 1})


if __name__ == "__main__":
    unittest.main()
