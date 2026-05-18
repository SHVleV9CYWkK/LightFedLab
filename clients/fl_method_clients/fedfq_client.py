"""
FedFQClient
===========
Client-side logic for FedFQ (Li et al., 2024).

Per round each selected client performs τ local SGD updates (same as FedAvg),
then computes the model update  h = θ_local - θ_global,  applies the
fine-grained mixed-precision quantizer  Q_f(h)  to every weight tensor, and
finally uploads the quantized state-dict  θ_global + Q_f(h).

The server receives a state-dict (so it can use the existing
_weight_aggregation routine without modification) whose deviation from the
global model is exactly the FedFQ-quantized local update.

Key design choices kept faithful to the paper
---------------------------------------------
- Local objective and optimizer are unchanged from FedAvg (Eq. 2).
- Quantization operates on the *update* h = θ_local - θ_global  (Eq. 4),
  NOT on the raw weights, so the unbiasedness theorem (Theorem 2) holds.
- Bit-widths are chosen *per scalar parameter*  (the paper's defining trait,
  Fig. 1: "individual parameter level").  We run CGSA per tensor — running it
  jointly across all tensors would not change the objective's separability
  (the squared-norm denominator factors out) and keeps the search tractable
  for tens of thousands of params per tensor.
- BatchNorm running stats, integer buffers, and 1-D bias/affine vectors are
  passed through unquantized (they are tiny and quantizing them yields no
  meaningful compression while hurting accuracy).
"""

import torch
from copy import deepcopy

from clients.client import Client
from utils.fedfq_quantization import fedfq_quantize


class FedFQClient(Client):
    """
    FedFQ client.

    Hyperparameters (paper § Setup of CGSA):
        target_avg_bits     - desired AVERAGE bit-width per quantized scalar.
                              The total budget is target_avg_bits * (#elements
                              in the tensor).  A value of 1.0 corresponds to
                              ~32× compression vs FP32 (the paper's FedFQ-32×
                              setting); 0.5 → ~64×; 0.25 → ~128×.
        initial_temperature - SA initial temperature  (paper: 1000).
        cooling_rate        - SA cooling rate         (paper: 0.95).
        max_iter            - SA iteration cap        (paper: 100).
        bit_options         - allowed bit-widths      (paper: 2, 4, 8;
                              we add 0 to permit dropping the smallest
                              components when the budget is tight).
    """

    def __init__(self, client_id, dataset_index, full_dataset, hyperparam,
                 device, **kwargs):
        super().__init__(client_id, dataset_index, full_dataset, hyperparam,
                         device, kwargs.get('dl_n_job', 0))

        # ---- FedFQ-specific hyperparameters ----
        # Map sparse_factor to a target average bit-width so the existing
        # `--sparse_factor` CLI flag controls FedFQ's compression ratio
        # without adding new arguments. The paper's three operating points
        # (32× / 64× / 128×) correspond to average bits per scalar of
        # roughly 1, 0.5, 0.25.
        # Users can also override directly via hyperparam['target_avg_bits'].
        sparse_factor = kwargs.get('sparse_factor', None)
        default_avg_bits = (32.0 * sparse_factor) if sparse_factor else 1.0
        self.target_avg_bits = hyperparam.get('target_avg_bits',
                                              default_avg_bits)

        # CGSA hyperparameters (paper § Setup of CGSA)
        self.initial_temperature = hyperparam.get('initial_temperature', 1000.0)
        self.cooling_rate        = hyperparam.get('cooling_rate', 0.95)
        self.min_temperature     = hyperparam.get('min_temperature', 1e-3)
        self.max_iter            = hyperparam.get('cgsa_max_iter', 100)
        self.bit_options         = tuple(hyperparam.get('bit_options',
                                                        (0, 2, 4, 8)))

        # Snapshot of the global model received at the start of each round.
        # Used to compute  h = θ_local - θ_global  after local training.
        self.global_state = None

    # ------------------------------------------------------------------
    # Down-link: keep a snapshot of the global weights for diff computation
    # ------------------------------------------------------------------
    def receive_model(self, global_model):
        super().receive_model(global_model)
        # Snapshot AFTER the base class has loaded the global weights into
        # self.model.  Detached CPU/GPU clone keeps memory low and is
        # decoupled from subsequent local updates.
        self.global_state = {
            k: v.detach().clone()
            for k, v in self.model.state_dict().items()
        }

    # ------------------------------------------------------------------
    # Helper: decide which tensors should be quantized
    # ------------------------------------------------------------------
    @staticmethod
    def _is_quantizable(key, tensor):
        """
        Quantize floating-point weight tensors only.

        Pass-through (unquantized) cases:
        - non-floating buffers (e.g. BN num_batches_tracked, int counters)
        - 1-D vectors (biases, BN gamma/beta) — tiny, quantization hurts more
          than it saves
        - any tensor with very few elements (CGSA degenerates)
        """
        if not tensor.is_floating_point():
            return False
        if tensor.ndim <= 1:
            return False
        if tensor.numel() < 16:
            return False
        return True

    # ------------------------------------------------------------------
    # Core: apply Q_f(·) to the update h = θ_local - θ_global
    # ------------------------------------------------------------------
    def _quantize_update(self, local_state):
        """
        Returns a state-dict whose value at key k equals
            θ_global[k] + Q_f( θ_local[k] - θ_global[k] )   if k is quantizable
            θ_local[k]                                       otherwise.

        The server's existing weighted-average aggregator will then produce
        exactly  θ_global + (1/|W|) Σ_i Q_f(h_i)  (Eq. 4 of the paper) when
        |W| clients are selected uniformly.
        """
        out_state = {}
        for key, w_local in local_state.items():
            w_global = self.global_state[key].to(w_local.device)

            if not self._is_quantizable(key, w_local):
                # Pass through unchanged (tensor goes to server as-is)
                out_state[key] = w_local.detach().clone()
                continue

            # Compute update tensor h, flatten, quantize, reshape back
            h = (w_local - w_global).detach()
            original_shape = h.shape
            h_flat = h.reshape(-1)

            budget = max(2, int(round(self.target_avg_bits * h_flat.numel())))

            h_q_flat = fedfq_quantize(
                h_flat=h_flat,
                budget=budget,
                bit_options=self.bit_options,
                initial_temperature=self.initial_temperature,
                cooling_rate=self.cooling_rate,
                min_temperature=self.min_temperature,
                max_iter=self.max_iter,
            )
            h_q = h_q_flat.reshape(original_shape)

            # Reconstruct the (quantized) local weight: server-side aggregation
            # of these tensors then becomes a plain weighted average — see Eq. 4.
            out_state[key] = (w_global + h_q).detach().clone()

        return out_state

    # ------------------------------------------------------------------
    # Training: identical to FedAvg locally; quantize the diff at upload time
    # ------------------------------------------------------------------
    def train(self):
        # Local SGD updates (Eq. 2 of the paper) — inherited from base Client.
        self._local_train()

        # Quantize the model update with the fine-grained CGSA scheme.
        local_state = self.model.state_dict()
        return self._quantize_update(local_state)