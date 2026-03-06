"""
pFed1BS Server
==============
Personalized Federated Learning via One-Bit Random Sketching.

Server-side communication protocol per round t:
─────────────────────────────────────────────────────────────────────────────
  DOWN-LINK : broadcast v^t ∈ {-1,+1}^m to all selected clients   (m bits)
  UP-LINK   : collect z_k = sign(Φw_k) ∈ {-1,+1}^m from each client (m bits)
  AGGREGATE : v^{t+1} = sign( Σ_k p_k · z_k )   (weighted majority vote)
─────────────────────────────────────────────────────────────────────────────

Key differences from FedAvg:
  • The server never distributes or receives full model weights after round 0.
  • Aggregation is a closed-form sign operation (Lemma 1 in the paper),
    not a weighted average of state_dicts.
  • self.model on the server is ONLY used for round-0 initialisation.
"""

import random
import numpy as np
import torch
from servers.server import Server


class pFed1BSServer(Server):
    """
    Server for pFed1BS.

    Additional constructor args (passed through `args` dict):
        sketch_seed       (int,   default 42)   – shared random seed for Phi
        compression_ratio (float, default 0.1)  – m/n ratio
    """

    def __init__(self, clients, model, device, args):
        # Base __init__ calls _distribute_model() and _init_clients(),
        # which handles round-0 model broadcasting and projector setup.
        super().__init__(clients, model, device, args)

        # Infer sketch dimension m from the first client's projector.
        # All clients share the same m after init_client() is called.
        first_client = self.clients[0]
        if first_client.m is None:
            raise RuntimeError(
                "Client projector not initialised. "
                "Ensure init_client() was called inside super().__init__()."
            )
        self.m = first_client.m

        # Dataset-size weights p_k for all clients (used in aggregation).
        # p_k = N_k / sum(N_i), matching paper notation.
        total = sum(self.datasets_len)
        self.p = torch.tensor(
            [length / total for length in self.datasets_len],
            dtype=torch.float32, device=self.device,
        )                                           # shape (K,)

        # Map client.id → index in self.clients list (for p_k look-up)
        self._client_id_to_idx = {c.id: i for i, c in enumerate(self.clients)}

        # Global one-bit consensus vector v^t in {-1,+1}^m.
        # Initialised to zeros (neutral) matching paper's v^0 = 0.
        self.v = torch.zeros(self.m, device=self.device)

    # ====================================================================
    # Down-link helpers
    # ====================================================================

    def _distribute_model(self):
        """
        Round-0 bootstrap: send the initial global model to every client.

        Called once by base __init__. After this, down-link communication
        switches to _distribute_consensus() for rounds 1 … T.
        """
        for client in self.clients:
            client.receive_model(self.model)

    def _distribute_consensus(self):
        """
        Broadcast the current one-bit consensus vector v^t to all selected
        clients for the upcoming round.

        Down-link cost: m bits per client  (vs. 32n bits for FedAvg).
        """
        for client in self.selected_clients:
            # Send a detached copy so each client owns its own tensor.
            client.receive_consensus(self.v.clone())

    # ====================================================================
    # Aggregation  —  Lemma 1 / Lemma 6 in the paper
    # ====================================================================

    def _average_aggregate(self, sketches: dict):
        """
        Compute the next consensus vector by weighted majority vote (Eq. 14):

            v^{t+1} = sign( Σ_{k in S^t} p_k · z_k )

        This is provably the exact minimiser of the server's discrete
        objective (Lemma 1 in the paper) — not a heuristic.

        Args:
            sketches: dict { client_id : z_k }
                      z_k is a CPU float32 Tensor, shape (m,),
                      values in {-1.0, +1.0}.
        """
        weighted_sum = torch.zeros(self.m, device=self.device)

        for client_id, z_k in sketches.items():
            idx = self._client_id_to_idx[client_id]
            p_k = self.p[idx]                           # scalar weight
            weighted_sum.add_(p_k * z_k.to(self.device))

        # Element-wise majority vote
        v_new = torch.sign(weighted_sum)
        # Tie-break: sign(0) = 0; map to +1 by convention (same as client)
        v_new[v_new == 0] = 1.0

        self.v = v_new

    # ====================================================================
    # Override _clone_and_detach to handle Tensor sketches
    # ====================================================================

    def _clone_and_detach(self, data):
        """
        Base class only handles dict (state_dict).
        pFed1BS clients return a Tensor, so we extend this method.
        """
        if isinstance(data, torch.Tensor):
            return data.clone().detach().cpu()
        # Delegate dicts to the base implementation (keeps compatibility)
        return super()._clone_and_detach(data)

    # ====================================================================
    # Override _execute_train_client for Tensor return type
    # ====================================================================

    def _execute_train_client(self, client, return_dict, seed):
        """
        Mirrors base-class logic but handles the fact that client.train()
        returns a Tensor z (not a state_dict dict) in pFed1BS.
        """
        try:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            if client.device.type == "cuda" and torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                torch.cuda.manual_seed_all(seed)

            z = client.train()                          # Tensor shape (m,)
            return_dict[client.id] = self._clone_and_detach(z)

        except Exception as e:
            print(f"Error training client {client.id}: {str(e)}")

    # ====================================================================
    # Main round  (overrides Server.train)
    # ====================================================================

    def train(self):
        """
        Execute one full round of Algorithm 1 (pFed1BS):

          1. Sample subset S^t of clients.
          2. Broadcast v^t to S^t                    ← down-link (m bits each)
          3. Each client trains locally and returns z_k = sign(Φw_k)
                                                     ← up-link   (m bits each)
          4. v^{t+1} = sign( Σ_{k in S^t} p_k z_k ) ← server aggregation

        The server does NOT call _distribute_model() here.
        Full model weights are never sent after round-0 initialisation.
        """
        # 1. Sample clients for this round
        self._sample_clients()

        # 2. Down-link: broadcast current consensus vector v^t
        print("Broadcasting consensus vector...")
        self._distribute_consensus()

        # 3. Up-link: collect one-bit sketches from selected clients
        print("Training clients...")
        sketches = self._clients_train()        # dict { client_id: z_k }

        # 4. Aggregate: weighted majority vote -> v^{t+1}
        print("Aggregating sketches...")
        self._average_aggregate(sketches)

        # No _distribute_model() — down-link is only v, never full weights.