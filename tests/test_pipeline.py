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

"""Config, sessionization, evaluation, TLS fingerprinting and extract command construction."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from hcs_detect.config import ConfigError, load_config
from hcs_detect.detectors.tls import ja3_from_components
from hcs_detect.evaluate import percentile_rank, rank_covert
from hcs_detect.extract import PAYLOAD_FIELDS, Pass, passes_for, payload_filter
from hcs_detect.netutil import in_cidr, ip_to_u32, u32_to_ip
from hcs_detect.sessions import build_sessions

CFG = Path(__file__).resolve().parents[1] / "configs" / "cp3_spotcheck.toml"

MINI_TOML = """
[scenario]
name = "mini"
chunks = ["a"]
[topology]
client_nets = ["10.1.0.0/16"]
peer_nets = ["10.2.0.0/24"]
[topology.service_nets]
"10.5.5.0/24" = "minio"
[topology.peer_service_ports]
6697 = "irc"
[[covert]]
name = "sky"
host = "10.1.1.4"
service = "10.5.5.0/24"
port = 8443
[pipeline]
min_session_packets = 3
"""


def write_cfg(text: str) -> Path:
    f = tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False)
    f.write(text)
    f.close()
    return Path(f.name)


class NetUtilTests(unittest.TestCase):
    def test_roundtrip(self):
        self.assertEqual(u32_to_ip(ip_to_u32("10.1.2.4")), "10.1.2.4")

    def test_in_cidr_vectorized(self):
        a = np.array([ip_to_u32("10.1.9.9"), ip_to_u32("10.2.0.1"), ip_to_u32("10.1.255.1")], dtype=np.uint32)
        self.assertEqual(in_cidr(a, "10.1.0.0/16").tolist(), [True, False, True])
        self.assertEqual(in_cidr(a, "10.2.0.1").tolist(), [False, True, False])  # bare host = /32


class ConfigTests(unittest.TestCase):
    def test_load_spotcheck(self):
        cfg = load_config(CFG)
        self.assertEqual(cfg.chunks, ("ens60", "ens61", "ens62"))
        self.assertEqual(cfg.label("10.1.1.4", "10.5.5.6", 8443), "skyhook")
        self.assertEqual(cfg.label("10.2.0.86", "10.3.3.6", 443), "mastodon-bob")
        self.assertEqual(cfg.label("10.1.2.4", "10.2.0.72", 8675), "obfs4")
        self.assertEqual(cfg.label("10.1.1.18", "10.5.5.6", 8443), "benign")  # neighbour on the same subnet
        self.assertEqual(cfg.label("10.1.1.4", "10.5.5.6", 443), "benign")  # right host, wrong port
        self.assertEqual(cfg.topology.service_class("10.5.5.6", 8443), "minio")
        self.assertEqual(cfg.topology.service_class("10.2.0.44", 443), "https")
        self.assertEqual(cfg.topology.service_class("10.2.0.44", 9999), "server:9999")
        self.assertIn(8675, cfg.extract_ports)
        self.assertIn(8443, cfg.extract_ports)

    def test_missing_key_is_config_error(self):
        p = write_cfg("[scenario]\nname='x'\n")
        with self.assertRaises(ConfigError):
            load_config(p)

    def test_missing_file(self):
        with self.assertRaises(ConfigError):
            load_config("/nonexistent/x.toml")


def synthetic_packets():
    """Alice (10.1.1.4) polls minio 10.5.5.6:8443 with 5 tiny connections; neighbour 10.1.1.18 downloads once;
    a peer host talks IRC to server_net; a router-fabric packet should be dropped."""
    rows = []
    t = 0.0
    for i in range(5):  # covert poller: SYN + 2 data + response per connection
        cp = 40000 + i
        rows += [
            (t, "10.1.1.4", "10.5.5.6", cp, 8443, 74, 0, True),
            (t + 0.1, "10.1.1.4", "10.5.5.6", cp, 8443, 200, 134, False),
            (t + 0.2, "10.5.5.6", "10.1.1.4", 8443, cp, 300, 234, False),
        ]
        t += 5
    for k in range(30):  # benign bulk download on one connection
        rows.append((t + k * 0.01, "10.5.5.6", "10.1.1.18", 8443, 50000, 1514, 1448, False))
    rows.append((t, "10.1.1.18", "10.5.5.6", 50000, 8443, 74, 0, True))
    for k in range(4):  # client -> peer irc
        rows.append((t + k, "10.1.1.18", "10.2.0.9", 51000, 6697, 120, 54, k == 0))
    rows.append((t, "203.0.2.3", "203.0.2.2", 179, 179, 80, 20, False))  # fabric, dropped
    df = pd.DataFrame(rows, columns=["ts", "src", "dst", "sport", "dport", "len", "plen", "syn"])
    df["src"] = df["src"].map(ip_to_u32).astype(np.uint32)
    df["dst"] = df["dst"].map(ip_to_u32).astype(np.uint32)
    for c in ("sport", "dport", "len", "plen"):
        df[c] = df[c].astype(np.uint16)
    df["proto"] = np.int8(6)
    df["ttl"] = np.uint8(63)
    return df


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(write_cfg(MINI_TOML))
        self.table = build_sessions(synthetic_packets(), self.cfg)

    def test_sessions_and_labels(self):
        s = self.table.sessions.set_index(["host_ip", "svc_ip", "port"])
        self.assertEqual(s.loc[("10.1.1.4", "10.5.5.6", 8443), "label"], "sky")
        self.assertEqual(s.loc[("10.1.1.18", "10.5.5.6", 8443), "label"], "benign")
        self.assertEqual(s.loc[("10.1.1.18", "10.2.0.9", 6697), "svc_class"], "irc")
        self.assertEqual(len(s), 3)  # fabric packet produced no session

    def test_orientation_and_connection_count(self):
        pk = self.table.packets
        sid = self.table.sessions.set_index("host_ip").loc["10.1.1.4", "sid"]
        g = pk[pk.sid == sid]
        self.assertEqual(int(g.is_req.sum()), 10)  # 5 SYN + 5 data client->server
        self.assertEqual(int((~g.is_req).sum()), 5)  # 5 responses
        self.assertEqual(g.cport.nunique(), 5)  # 5 connections
        self.assertTrue((g.port == 8443).all())

    def test_min_session_packets_filter(self):
        cfg = load_config(write_cfg(MINI_TOML.replace("min_session_packets = 3", "min_session_packets = 10")))
        table = build_sessions(synthetic_packets(), cfg)
        self.assertEqual(set(table.sessions.label), {"sky", "benign"})
        self.assertEqual(len(table.sessions), 2)  # irc (4 pkts) dropped


class EvaluateTests(unittest.TestCase):
    def test_percentile_rank_directions(self):
        b = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(percentile_rank(5.0, b, "high"), 100.0)
        self.assertEqual(percentile_rank(0.0, b, "low"), 100.0)
        self.assertEqual(percentile_rank(2.5, b, "high"), 50.0)
        self.assertTrue(np.isnan(percentile_rank(np.nan, b, "high")))
        self.assertTrue(np.isnan(percentile_rank(1.0, np.array([np.nan]), "high")))

    def test_rank_covert_uses_registry_directions(self):
        from hcs_detect.detectors import REGISTRY, rankable

        cols = [d.name for d in REGISTRY]
        ranked = [d.name for d in rankable()]
        rows = []
        for i in range(4):
            rows.append(
                {
                    "label": "benign",
                    "host": f"h{i}",
                    "service": "s",
                    "svc_class": "minio",
                    **{c: float(i) for c in cols},
                }
            )
        rows.append(
            {
                "label": "cov",
                "host": "x",
                "service": "s",
                "svc_class": "minio",
                **{c: (10.0 if d.direction == "high" else -1.0) for c, d in zip(cols, REGISTRY, strict=True)},
            }
        )
        ranks = rank_covert(pd.DataFrame(rows))
        self.assertEqual(len(ranks), 1)
        self.assertTrue(all(ranks.iloc[0][c] == 100.0 for c in ranked))
        self.assertNotIn("fep_flows", ranks.columns)  # context metrics are never ranked


class TlsTests(unittest.TestCase):
    def test_ja3_deterministic_and_grease_stripped(self):
        a = ja3_from_components("0x0303", "0x1301-0x1302-0x0a0a", "0-23-10-0x0a0a", "0x001d-0x0017", "0")
        b = ja3_from_components("0x0303", "0x1301-0x1302", "0-23-10", "0x001d-0x0017", "0")
        self.assertEqual(a.ja3, b.ja3)
        self.assertEqual(a.ja3_full, "771,4865-4866,0-23-10,29-23,0")
        self.assertEqual(a.shape, "v771 2cs 3ext")

    def test_empty_components(self):
        h = ja3_from_components("", "", "", "", "")
        self.assertEqual(h.ja3_full, ",,,,")


class ExtractTests(unittest.TestCase):
    def test_payload_filter_uses_comma_sets_and_config_nets(self):
        cfg = load_config(CFG)
        f = payload_filter(cfg)
        self.assertIn("ip.src==10.1.0.0/16", f)
        self.assertIn("ip.src==10.2.0.0/24", f)
        self.assertIn("tcp.seq<=8192", f)
        self.assertRegex(f, r"tcp\.dstport in \{\d+(, \d+)+\}")  # Wireshark 4.x set literal
        self.assertIn("8675", f)

    def test_pass_command_shape(self):
        p = Pass("payload", PAYLOAD_FIELDS, "tcp.len>0")
        cmd = p.command("/usr/bin/tshark", Path("/x/a.pcap"))
        self.assertEqual(cmd[:3], ["/usr/bin/tshark", "-r", "/x/a.pcap"])
        self.assertIn("-Y", cmd)
        self.assertEqual(cmd.count("-e"), len(PAYLOAD_FIELDS))
        self.assertIn("occurrence=f", cmd)

    def test_hello_pass_aggregates(self):
        hello = [p for p in passes_for(load_config(CFG)) if p.suffix == "hello"][0]
        cmd = hello.command("t", Path("a.pcap"))
        self.assertIn("occurrence=a", cmd)
        self.assertIn("aggregator=-", cmd)


if __name__ == "__main__":
    unittest.main()
