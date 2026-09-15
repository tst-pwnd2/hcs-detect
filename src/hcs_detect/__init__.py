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

"""hcs_detect: passive detection of hidden communication systems in packet captures.

The pipeline has four stages. ``extract`` runs tshark over the pcaps and writes feature
tables; ``reduce`` turns those tables into a compact, typed packet cache; ``detect``
sessionizes the traffic, scores every session on every registered detector, and ranks
the known covert sessions against the benign population; ``report`` renders the result.

The unit of analysis is the session, that is, all traffic between one host and one
(service IP, port) pair. We take this unit because a dead-drop poller spreads its
signature across hundreds of short connections; per-connection statistics dilute what
per-session statistics make plain.

Detectors are registered in :mod:`hcs_detect.detectors` and grouped by the observation
layer they belong to (structure, beaconing, divergence, fep, tls). Ground truth, namely
which host and service pairs are covert, is configuration rather than code, so the same
detectors run unchanged across scenarios.
"""

from .config import Config, CovertSpec, Topology, load_config

__all__ = ["Config", "CovertSpec", "Topology", "load_config", "__version__"]
__version__ = "0.1.0"
