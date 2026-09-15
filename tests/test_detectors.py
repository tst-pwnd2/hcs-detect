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

"""Detector unit tests on synthetic inputs where the right answer is known.

Run with ``python -m unittest discover -s tests`` (or pytest)."""

import unittest

import numpy as np

from hcs_detect.detectors import ClassBaseline, SessionContext, compute_all
from hcs_detect.detectors.beaconing import periodicity
from hcs_detect.detectors.divergence import IAT_BINS, LEN_BINS, iat_hist, kl_divergence, len_hist
from hcs_detect.detectors.fep import has_known_magic, popcount_per_byte, shannon_entropy


def periodic_times(n=200, period=5.0, jitter=0.02, seed=0):
    rng = np.random.default_rng(seed)
    return np.cumsum(period + rng.normal(0, jitter, n))


def poisson_times(n=200, rate=0.2, seed=0):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.exponential(1 / rate, n))


class BeaconingTests(unittest.TestCase):
    def test_periodic_is_regular(self):
        cv, mode_frac, spr = periodicity(periodic_times())
        self.assertLess(cv, 0.05)
        self.assertGreater(mode_frac, 0.95)
        self.assertGreater(spr, 0.1)

    def test_poisson_is_irregular(self):
        cv, mode_frac, spr = periodicity(poisson_times())
        self.assertGreater(cv, 0.7)  # exponential IATs have CV close to 1
        self.assertLess(mode_frac, 0.3)
        self.assertLess(spr, 0.1)

    def test_periodic_beats_poisson_on_every_statistic(self):
        p = periodicity(periodic_times())
        q = periodicity(poisson_times())
        self.assertLess(p[0], q[0])
        self.assertGreater(p[1], q[1])
        self.assertGreater(p[2], q[2])

    def test_too_few_events_is_nan(self):
        self.assertTrue(all(np.isnan(x) for x in periodicity(np.array([1.0, 2.0, 3.0]))))

    def test_long_gaps_do_not_wash_out_mode_frac(self):
        t = periodic_times(100)
        t = np.concatenate([t, t[-1] + 600 + periodic_times(100)])  # one 10-minute outage
        _, mode_frac, _ = periodicity(t)
        self.assertGreater(mode_frac, 0.9)


class FepTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)
        self.random64 = self.rng.integers(0, 256, 64, dtype=np.uint8).tobytes()
        self.tls64 = bytes.fromhex("160301004a010000460303") + b"\x00" * 21 + bytes.fromhex("0002130100") + b"\x00" * 27
        self.http64 = b"GET /api/v1/timelines/tag/x HTTP/1.1\r\nHost: 10.3.3.6\r\n\r\n".ljust(64, b" ")

    def test_random_bytes_look_fep(self):
        self.assertFalse(has_known_magic(self.random64))
        self.assertGreater(popcount_per_byte(self.random64), 3.6)
        self.assertLess(popcount_per_byte(self.random64), 4.4)
        self.assertGreater(shannon_entropy(self.random64), 5.5)  # ceiling is log2(64)=6 for 64 samples

    def test_tls_and_http_carry_magic(self):
        self.assertTrue(has_known_magic(self.tls64))
        self.assertTrue(has_known_magic(self.http64))

    def test_structured_is_lower_entropy_and_popcount_than_random(self):
        self.assertLess(shannon_entropy(self.tls64), shannon_entropy(self.random64))
        self.assertLess(popcount_per_byte(self.tls64), popcount_per_byte(self.random64))

    def test_empty_is_nan(self):
        self.assertTrue(np.isnan(shannon_entropy(b"")))
        self.assertTrue(np.isnan(popcount_per_byte(b"")))
        self.assertFalse(has_known_magic(b"\x16"))


class DivergenceTests(unittest.TestCase):
    def test_kl_self_is_zero(self):
        h = np.array([5, 10, 20, 10, 5])
        self.assertAlmostEqual(kl_divergence(h, h), 0.0, places=9)

    def test_kl_grows_with_shift(self):
        base = np.array([100, 50, 10, 0, 0, 0])
        near = np.array([90, 60, 10, 0, 0, 0])
        far = np.array([0, 0, 0, 10, 50, 100])
        self.assertLess(kl_divergence(near, base), kl_divergence(far, base))
        self.assertGreater(kl_divergence(far, base), 3.0)

    def test_kl_handles_empty_reference_bins(self):
        self.assertTrue(np.isfinite(kl_divergence(np.array([0, 0, 10]), np.array([10, 0, 0]))))

    def test_histogram_shapes(self):
        self.assertEqual(len_hist(np.array([66, 1514, 300])).size, LEN_BINS.size - 1)
        self.assertEqual(iat_hist(np.array([0.0, 0.5, 1.0])).size, IAT_BINS.size - 1)


def make_context(ts, *, is_req=None, syn=None, plen=None, lens=None, cport=None, baseline=None, segs=None):
    n = len(ts)
    return SessionContext(
        ts=np.asarray(ts, float),
        is_req=np.ones(n, bool) if is_req is None else is_req,
        syn=np.zeros(n, bool) if syn is None else syn,
        plen=np.full(n, 100) if plen is None else plen,
        len=np.full(n, 166) if lens is None else lens,
        cport=np.full(n, 50000) if cport is None else cport,
        svc_class="minio",
        baseline=baseline,
        first_segments=segs or [],
    )


class ContextAndRegistryTests(unittest.TestCase):
    def test_poller_vs_downloader_end_to_end(self):
        # poller: 200 connections, one SYN each every 5 s, tiny; downloader: 3 connections, bulk 1514-byte frames
        t = periodic_times(200)
        poller = make_context(t, syn=np.ones(200, bool), cport=np.arange(200) + 40000, lens=np.full(200, 120))
        rng = np.random.default_rng(3)
        td = np.sort(rng.uniform(0, 1000, 5000))
        down = make_context(td, syn=np.zeros(5000, bool), cport=rng.choice([1, 2, 3], 5000), lens=np.full(5000, 1514))
        base = ClassBaseline(len_hist=len_hist(down.len), iat_hist=iat_hist(down.ts), len_pool=down.len, n_sessions=1)
        poller.baseline = base
        down.baseline = base
        p = compute_all(poller)
        d = compute_all(down)
        self.assertGreater(p["conns"], d["conns"])
        self.assertLess(p["pkts_per_conn"], d["pkts_per_conn"])
        self.assertGreater(p["mode_frac"], 0.9)
        self.assertGreater(p["kl_len"], d["kl_len"])
        self.assertAlmostEqual(d["kl_len"], 0.0, places=6)

    def test_request_events_switch_on_connection_count(self):
        t = np.arange(20, dtype=float)
        single = make_context(t, syn=np.zeros(20, bool), plen=np.r_[np.zeros(10), np.full(10, 50)])
        self.assertEqual(single.request_events().size, 10)  # payload-bearing packets
        multi = make_context(t, syn=np.r_[np.ones(8, bool), np.zeros(12, bool)], cport=np.arange(20) + 1)
        self.assertEqual(multi.request_events().size, 8)  # SYNs

    def test_every_registered_detector_returns_float_and_never_raises(self):
        ctx = make_context(np.arange(50, dtype=float))
        out = compute_all(ctx)
        self.assertEqual(len(out), len(__import__("hcs_detect.detectors", fromlist=["REGISTRY"]).REGISTRY))
        self.assertTrue(all(isinstance(v, float) for v in out.values()))


if __name__ == "__main__":
    unittest.main()
