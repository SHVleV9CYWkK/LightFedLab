"""
FedFQ Quantization Module
=========================
Implements the fine-grained adaptive quantization strategy from the FedFQ paper.

Core components:
1. Random Uniform Quantization Q(h, b)  -  Eq. (5) in the paper
   Q(h) = ||h||_2 · sign(h) · ξ(|h|/||h||_2, s),  s = 2^b - 1
   where ξ is a stochastic rounding function that maps each normalized value
   to {0, 1/s, 2/s, ..., s/s} in an unbiased manner.

2. Constraint-Guided Simulated Annealing (CGSA)  -  Algorithm 1
   Minimises the variance upper bound
        q_f = Σ_j  d / 4^{b_j}  ·  |h_j|^2 / ||h||_2^2
   subject to Σ_j b_j = B (total bit-width budget).
   Available bit-widths per element: {0, 2, 4, 8}.

3. Fine-grained quantization Q_f(h)  -  applies element-wise bit allocation
   from CGSA and returns the dequantized (float) tensor that is sent to the
   server (mathematically identical to sending integer codes; we keep float
   form so the server-side aggregation logic of the existing framework can be
   reused without modification).
"""

import math
import random
import torch


# ---------------------------------------------------------------------------
# 1. Random Uniform Quantization (Alistarh et al., 2017 / FedFQ Eq. 5)
# ---------------------------------------------------------------------------

def random_uniform_quantize(h_flat, bits):
    """
    Element-wise random uniform quantization with per-element bit-widths.

    Implements Eq. (5) of the FedFQ paper:
        Q(h)_j = ||h||_2 · sign(h_j) · ξ_j(|h_j| / ||h||_2, s_j)
    where  s_j = 2^{b_j} - 1   and   ξ_j  is the stochastic rounding operator:
        ξ(x, s) = ⌊x · s⌋ / s          with probability  1 - (x · s - ⌊x · s⌋)
        ξ(x, s) = (⌊x · s⌋ + 1) / s    with probability      (x · s - ⌊x · s⌋)
    This rounding is unbiased: E[ξ(x, s)] = x.

    For bits == 0  the element is dropped (output = 0).

    Args:
        h_flat: 1-D float tensor, the flattened update.
        bits:   1-D int tensor of the same length, values in {0, 2, 4, 8}.

    Returns:
        h_quant: 1-D float tensor, same shape as h_flat, the dequantized update.
    """
    device = h_flat.device
    norm = torch.norm(h_flat, p=2)
    if norm.item() == 0.0:
        return torch.zeros_like(h_flat)

    abs_normed = h_flat.abs() / norm                # in [0, 1]
    sign = torch.sign(h_flat)

    # s = 2^b - 1   (for b = 0 we will mask to zero afterwards)
    bits_f = bits.float()
    s = torch.pow(2.0, bits_f) - 1.0                # 0, 3, 15, 255

    # Safe scaled value (avoid division-by-zero where s == 0)
    s_safe = torch.where(s > 0, s, torch.ones_like(s))
    scaled = abs_normed * s_safe                    # in [0, s]

    floor_v = torch.floor(scaled)
    frac = scaled - floor_v                         # in [0, 1)

    # Unbiased stochastic rounding
    rand_v = torch.rand_like(frac)
    rounded = floor_v + (rand_v < frac).float()     # ⌊·⌋ or ⌊·⌋+1
    xi = rounded / s_safe                           # in [0, 1]

    h_quant = norm * sign * xi

    # Elements assigned 0 bits are dropped entirely (b=0 ⇒ s=0 ⇒ output 0)
    zero_bit_mask = (bits == 0)
    if zero_bit_mask.any():
        h_quant = torch.where(zero_bit_mask, torch.zeros_like(h_quant), h_quant)

    return h_quant


# ---------------------------------------------------------------------------
# 2. Constraint-Guided Simulated Annealing (CGSA)  -  Algorithm 1
# ---------------------------------------------------------------------------

def cgsa_allocate_bits(
    h_flat,
    budget,
    bit_options=(0, 2, 4, 8),
    initial_temperature=1000.0,
    cooling_rate=0.95,
    min_temperature=1e-3,
    max_iter=100,
):
    """
    Constraint-Guided Simulated Annealing for element-wise bit-width allocation.

    Faithful implementation of Algorithm 1 in the FedFQ paper.

    Objective (variance upper bound, Eq. 12):
        q_f  ∝  Σ_j  (d / 4^{b_j}) · |h_j|^2
    minimised subject to Σ_j b_j ≤ budget,  b_j ∈ {0, 2, 4, 8}.

    Initial solution (paper § "Initial solution"):
        Sort indices by |h_j| descending; assign 2 bits to the largest
        components until the budget is exhausted; the rest stay at 0.

    Neighbourhood (paper § "Neighborhood search", Corollary 3):
        Pick i < j in the magnitude-sorted order.  If the larger-magnitude
        element i can be doubled (b_i ∈ {2,4} → {4,8}) AND the smaller
        element j can be halved (b_j ∈ {8,4,2} → {4,2,1}), apply that swap.
        This preserves the total bit budget and respects the
        "larger components deserve more bits" direction from Corollary 3.

    Acceptance (standard SA):
        Always accept improvements.  Accept worse moves with probability
        exp(-Δ / T).  Cool T by factor `cooling_rate` per iteration.

    Args:
        h_flat:               1-D float tensor (the local update).
        budget:               int, total bit-width budget Σ b_j.
        bit_options:          allowed bit-widths (paper uses 2, 4, 8;
                              we include 0 to enable dropping elements
                              when the budget cannot cover everyone).
        initial_temperature:  T_0 (paper: 1000).
        cooling_rate:         α   (paper: 0.95).
        min_temperature:      stop when T falls below this.
        max_iter:             cap on iterations (paper: 100, dataset-dependent).

    Returns:
        bits:  1-D int tensor (same length as h_flat) of element bit-widths.
    """
    device = h_flat.device
    d = h_flat.numel()

    # Precompute |h_j|^2 — the per-element "weight" in the objective.
    h_sq = (h_flat.float() ** 2)

    # Sort indices by |h_j| descending (used for initial allocation AND
    # for the neighbourhood-search "i < j ⇒ |h_i| ≥ |h_j|" ordering).
    sorted_idx = torch.argsort(-h_flat.abs())       # descending by magnitude

    # ---------------- Initial solution ----------------
    # Assign 2 bits to the largest components until budget is exhausted.
    bits = torch.zeros(d, dtype=torch.long, device=device)
    n_two_bit = min(d, budget // 2)
    if n_two_bit > 0:
        bits[sorted_idx[:n_two_bit]] = 2

    # If budget is odd, one extra bit is wasted (cannot assign 1-bit in our set).

    # ---------------- Objective ----------------
    # We minimise  Σ_j  |h_j|^2 / 4^{b_j}  (the constant d / ||h||_2^2 is
    # irrelevant for argmin).  Use a per-element coefficient 4^{b_j} so that
    # b_j = 0 ⇒ coefficient 1 (full variance term) and b_j ↑ ⇒ coefficient ↑
    # (shrinking variance term).
    def objective(b_vec):
        coef = torch.pow(4.0, b_vec.float())        # 1, 4, 16, 256
        return (h_sq / coef).sum().item()

    cur_value = objective(bits)
    best_bits = bits.clone()
    best_value = cur_value

    # Build a lookup so a uniform random pair (i_rank, j_rank) with i<j in the
    # sorted ranking maps directly to element indices (preserves Corollary 3's
    # direction: rank i has larger magnitude than rank j).
    sorted_idx_cpu = sorted_idx.tolist()
    bit_set = set(bit_options)

    T = initial_temperature
    iters_left = max_iter
    current_total = int(bits.sum().item())

    while T > min_temperature and iters_left > 0:
        # ---- Neighbourhood search ----
        # Pick two ranks i < j (so |h_{idx_i}| ≥ |h_{idx_j}|).
        i_rank = random.randint(0, d - 2)
        j_rank = random.randint(i_rank + 1, d - 1)
        idx_i = sorted_idx_cpu[i_rank]
        idx_j = sorted_idx_cpu[j_rank]

        bi = int(bits[idx_i].item())
        bj = int(bits[idx_j].item())

        # The paper's Algorithm 1 moves (lines 12-14):
        #   new_b_i = 2 · b_i        (give MORE bits to the LARGER component)
        #   new_b_j =     b_j / 2    (take bits from the SMALLER component)
        # We restrict the move to the discrete set {0, 2, 4, 8}.  When the
        # naive multiply-by-2 / divide-by-2 falls outside the set we skip
        # (matches the algorithm's "if" guards on lines 12-15).
        new_bi = bi * 2
        new_bj = bj // 2

        # Validity checks (Algorithm 1 lines 12, 15)
        if new_bi not in bit_set or new_bj not in bit_set:
            # Special case: bj=2 ⇒ bj/2=1 ∉ set ⇒ snap to 0 (drop the element).
            # This is a natural extension of the paper's "halving" direction
            # and is consistent with our bit_options that include 0.
            if new_bj == 1 and 0 in bit_set:
                new_bj = 0
            else:
                iters_left -= 1
                T *= cooling_rate
                continue

        delta_bits = (new_bi - bi) + (new_bj - bj)
        new_total = current_total + delta_bits

        # Strict budget enforcement: never let the move take us above B.
        # (Going *below* B is fine — fewer bits = more compression, and the
        # objective is monotone non-increasing in remaining budget.)
        if new_total > budget:
            iters_left -= 1
            T *= cooling_rate
            continue

        # Tentatively apply the move
        bits[idx_i] = new_bi
        bits[idx_j] = new_bj
        new_value = objective(bits)
        delta = new_value - cur_value

        # ---- Acceptance criterion (Algorithm 1 line 19) ----
        if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-12)):
            cur_value = new_value
            current_total = new_total
            if cur_value < best_value:
                best_value = cur_value
                best_bits = bits.clone()
        else:
            # Reject: roll back
            bits[idx_i] = bi
            bits[idx_j] = bj

        # ---- Cooling (Algorithm 1 line 24) ----
        T *= cooling_rate
        iters_left -= 1

    return best_bits


# ---------------------------------------------------------------------------
# 3. Top-level: FedFQ fine-grained quantizer  Q_f(·)
# ---------------------------------------------------------------------------

def fedfq_quantize(
    h_flat,
    budget,
    bit_options=(0, 2, 4, 8),
    initial_temperature=1000.0,
    cooling_rate=0.95,
    min_temperature=1e-3,
    max_iter=100,
):
    """
    Apply the FedFQ fine-grained quantization operator Q_f(·) to a flat update.

    Pipeline:
        1. CGSA → per-element bit-widths  b_j
        2. random_uniform_quantize → dequantized float tensor (unbiased)

    Args / Returns: see the two functions above.
    """
    if h_flat.numel() == 0:
        return h_flat.clone()

    bits = cgsa_allocate_bits(
        h_flat=h_flat,
        budget=budget,
        bit_options=bit_options,
        initial_temperature=initial_temperature,
        cooling_rate=cooling_rate,
        min_temperature=min_temperature,
        max_iter=max_iter,
    )
    return random_uniform_quantize(h_flat, bits)