"""
FedFQServer
===========
Server-side logic for FedFQ.

FedFQ's server-side update rule (Eq. 4 in the paper):

    θ^{t+1} = θ^t + (1 / |W|) · Σ_{i ∈ W}  Q_f( h_i^{t,τ} )

Each client uploads  θ_global + Q_f(h_i)  as a regular state-dict
(see FedFQClient._quantize_update), so the weighted-average aggregator
inherited from the base Server performs exactly the operation above —
no FedFQ-specific aggregation logic is required.

This mirrors how the existing FedAvgServer is structured in the framework.
"""

from servers.server import Server


class FedFQServer(Server):
    def __init__(self, clients, model, device, args):
        super().__init__(clients, model, device, args)

    def _average_aggregate(self, weights_list):
        # Standard weighted (by dataset size) averaging — equivalent to
        # Eq. 4 when client selection probabilities are proportional to
        # dataset size (the framework's default).
        self._weight_aggregation(weights_list)