import torch
from copy import deepcopy

from clients.client import Client


# ============================================================================
# SRHT Projector  –  O(n log n) structured random projection
# ============================================================================

class SRHTProjector:
    """
    Subsampled Randomized Hadamard Transform (SRHT).

    Implements the matrix-free projection described in the paper (Eq. 16):

        forward : Φw  = sqrt(n'/m) · S · H · D · pad(w)    shape (n,) → (m,)
        backward: Φᵀv = Ptrunc · D · Hᵀ · S'ᵀ · v          shape (m,) → (n,)

    where
        D   – diagonal random ±1 sign matrix          shape (n',)
        H   – normalised Walsh-Hadamard matrix         shape (n'×n')
        S   – random row-subsampling matrix            selects m rows from n'
        pad – zero-padding from n to n' (next power of 2)

    The projection is fully determined by `seed`, which is shared between
    server and all clients so Φ never needs to be transmitted explicitly.
    """

    def __init__(self, n: int, m: int, seed: int, device: torch.device):
        self.n      = n
        self.m      = m
        self.device = device

        # n' = smallest power of 2 >= n  (required by the FHT algorithm)
        self.n_prime = 1 << (max(n, 1) - 1).bit_length()
        if self.n_prime < n:          # edge-case: n is already a power of 2
            self.n_prime = n

        rng = torch.Generator()
        rng.manual_seed(seed)

        # D: random ±1 signs, shape (n_prime,)
        d_bits = torch.randint(0, 2, (self.n_prime,), generator=rng)
        self.D = (d_bits * 2 - 1).float().to(device)

        # S: m distinct column indices sampled without replacement from [0, n_prime)
        self.S_idx = torch.randperm(self.n_prime, generator=rng)[:m].to(device)

        # Scaling factor sqrt(n'/m)  (paper Eq. 16, S' = sqrt(n'/m) · S)
        self.scale = (self.n_prime / self.m) ** 0.5

    # ------------------------------------------------------------------
    # Fast Walsh-Hadamard Transform (iterative, no recursion)
    # ------------------------------------------------------------------

    @staticmethod
    def _fht(x: torch.Tensor) -> torch.Tensor:
        """
        Normalised Fast Walsh-Hadamard Transform.

        Input length must be a power of 2.
        Satisfies H @ Hᵀ = I  (orthonormal).
        Returns a new tensor (not in-place) to avoid autograd issues.
        """
        length = x.shape[-1]
        out = x.clone()
        h = 1
        while h < length:
            # butterfly step — clone slices to avoid in-place aliasing
            a = out[..., :length:2 * h].clone()
            b = out[..., h::2 * h].clone()
            out[..., :length:2 * h] = a + b
            out[..., h::2 * h]      = a - b
            h *= 2
        return out / (length ** 0.5)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """
        Φw = scale · S(H(D · pad(w)))

        Args:
            w: Flat model parameter vector, shape (n,). Must be on self.device.
        Returns:
            Sketch, shape (m,).
        """
        # 1. Zero-pad to n_prime
        w_pad = w.new_zeros(self.n_prime)
        w_pad[:self.n] = w

        # 2. Element-wise sign flip D
        w_pad = w_pad * self.D

        # 3. Normalised FHT
        w_fht = self._fht(w_pad)

        # 4. Subsample m coordinates and apply scaling
        return self.scale * w_fht[self.S_idx]

    def backward(self, v: torch.Tensor) -> torch.Tensor:
        """
        Φᵀv = Ptrunc(D(Hᵀ(S'ᵀv)))

        Args:
            v: Low-dimensional vector, shape (m,). Must be on self.device.
        Returns:
            Reconstructed vector in original parameter space, shape (n,).
        """
        # 1. S'ᵀ : scatter v (scaled) into a zero-padded n_prime vector
        v_lift = v.new_zeros(self.n_prime)
        v_lift[self.S_idx] = v * self.scale

        # 2. Hᵀ = H for the normalised Walsh-Hadamard matrix
        v_fht = self._fht(v_lift)

        # 3. D again (D is involutory: D² = I, so Dᵀ = D)
        v_fht = v_fht * self.D

        # 4. Ptrunc : keep only the first n coordinates
        return v_fht[:self.n]


# ============================================================================
# Bit-packing utilities  (wire-level encoding of one-bit vectors)
# ============================================================================

def pack_bits(z: torch.Tensor) -> torch.Tensor:
    """
    Pack a {-1.0, +1.0}^m tensor into ceil(m/8) bytes (torch.uint8).

    Encoding convention:  +1 → bit 1,   -1 → bit 0.

    Use this before sending z over the network to achieve the advertised
    m-bit communication cost (instead of 32m bits for float32).

    Args:
        z: shape (m,), float32, values in {-1.0, +1.0}.
    Returns:
        packed: shape (ceil(m/8),), dtype torch.uint8, on same device as z.
    """
    m    = z.shape[0]
    bits = ((z + 1) / 2).bool()           # {-1,+1} → {False,True}
    pad  = (8 - m % 8) % 8
    if pad:
        bits = torch.cat([bits, bits.new_zeros(pad)])
    n_bytes = bits.shape[0] // 8
    packed  = torch.zeros(n_bytes, dtype=torch.uint8, device=z.device)
    for i in range(8):
        packed |= bits[i::8].to(torch.uint8) << i
    return packed


def unpack_bits(packed: torch.Tensor, m: int) -> torch.Tensor:
    """
    Inverse of pack_bits.  Recover {-1.0, +1.0}^m from uint8 bytes.

    Args:
        packed: shape (ceil(m/8),), dtype torch.uint8.
        m:      original vector length before packing.
    Returns:
        z: shape (m,), float32, values in {-1.0, +1.0}.
    """
    bits_list = [((packed >> i) & 1).bool() for i in range(8)]
    bits = torch.stack(bits_list, dim=1).reshape(-1)[:m]
    return bits.float() * 2 - 1           # {False,True} → {-1.0,+1.0}


# ============================================================================
# Helpers
# ============================================================================

def _model_to_vec(model: torch.nn.Module) -> torch.Tensor:
    """Flatten all model parameters into a single 1-D tensor (detached copy)."""
    return torch.cat([p.data.view(-1) for p in model.parameters()])


# ============================================================================
# pFed1BS Client
# ============================================================================

class pFed1BSClient(Client):
    """
    pFed1BS client.
    • train() → returns z = sign(Φw) ∈ {-1,+1}^m as float32 tensor.
        For actual byte-level transmission call pack_bits(z) on the result.
        Communication cost: m bits = compression_ratio × n bits.

    Local objective (paper Eq. 6):
        F̃_k(w; v) = f_k(w)
                   + λ · (h_γ(Φw) − ⟨v, Φw⟩)
                   + (μ/2) · ‖w‖²

    Gradient used for each SGD step (paper Eq. 11):
        ∇F̃_k = ∇f̂_k(w) + λ · Φᵀ(tanh(γΦw) − v) + μ · w
    """

    def __init__(
        self,
        client_id: int,
        dataset_index: dict,
        full_dataset,
        hyperparam: dict,
        device: torch.device,
        **kwargs,
    ):
        super().__init__(
            client_id, dataset_index, full_dataset, hyperparam, device,
            kwargs.get('dl_n_job', 0),
        )

        # ── pFed1BS hyperparameters ──────────────────────────────────────
        self.lam = hyperparam.get('lam', 5e-4)        # λ  sign-alignment strength
        self.mu = hyperparam.get('mu',  1e-5)         # μ  L2 penalty
        self.gamma = hyperparam.get('gamma', 1e4)        # γ  tanh smoothing
        self.compression_ratio = hyperparam.get('compression_ratio', 0.1)  # m/n
        self.sketch_seed = hyperparam.get('sketch_seed', 42)   # shared Φ seed

        # ── State initialised later ──────────────────────────────────────
        self.projector: SRHTProjector = None   # built once model dim is known
        self.m: int = None # sketch dimension

        # One-bit consensus vector  v ∈ {-1,+1}^m  (updated each round)
        self.v: torch.Tensor = None

    # ====================================================================
    # Down-link:  server → client
    # ====================================================================

    def receive_model(self, global_model: torch.nn.Module):
        """
        Initialise the local personalised model from the server's global model.

        Called ONCE before training begins (round-0 bootstrap).
        All subsequent rounds the server communicates only via receive_consensus().

        Overrides base-class behaviour: the base class would overwrite local
        weights on every round, which would destroy personalisation.
        """
        if self.model is None:
            # First call: deep-copy architecture + initial weights from server
            self.model = deepcopy(global_model).to(self.device)
        # Subsequent calls are intentionally ignored.

    def receive_consensus(self, v: torch.Tensor):
        """
        Receive the global one-bit consensus vector broadcast by the server.

        This is the SOLE down-link payload for rounds 1 … T.
        Communication cost: m bits  (m = compression_ratio × n).

        Args:
            v: Tensor shape (m,), float32, values in {-1.0, +1.0}.
               If the server transmitted packed bytes, call unpack_bits() first:
                   v = unpack_bits(packed_bytes, self.m)
        """
        self.v = v.to(self.device).float()

    # ====================================================================
    # Initialisation  (server calls this after receive_model)
    # ====================================================================

    def init_client(self):
        """
        Build optimizer, LR scheduler, and SRHT projector.

        Must be called AFTER receive_model() so that self.model is not None.
        """
        super().init_client()       # creates self.optimizer + self.lr_scheduler
        self._init_projector()      # builds SRHTProjector using model dimension

    def _init_projector(self):
        """Instantiate the SRHT projector (idempotent – safe to call multiple times)."""
        if self.projector is not None:
            return
        if self.model is None:
            raise RuntimeError(
                "pFed1BSClient._init_projector() called before receive_model(). "
                "Please call receive_model() first."
            )
        n = sum(p.numel() for p in self.model.parameters())
        self.m = max(1, int(n * self.compression_ratio))
        self.projector = SRHTProjector(n, self.m, self.sketch_seed, self.device)

    # ====================================================================
    # Up-link:  local training → one-bit sketch
    # ====================================================================

    def train(self) -> torch.Tensor:
        """
        Run local personalised training then return the one-bit sketch.

        Implements Algorithm 1 (ClientUpdate) from the paper:
          For each local epoch and each mini-batch:
            (a) Forward pass + task-loss backward  → ∇f̂_k accumulates in p.grad
            (b) Compute regulariser gradient manually:
                    λ · Φᵀ(tanh(γΦw) − v)
            (c) Compute L2 gradient manually:
                    μ · w
            (d) Accumulate (b)+(c) into p.grad, then call optimizer.step()
          After all local steps:
            z = sign(Φw)   ← one-bit up-link payload

        Returns:
            z: torch.Tensor shape (m,), float32, values in {-1.0, +1.0}.
               This is the ONLY information uploaded to the server each round.
               For wire-level transmission: pack_bits(z) → ceil(m/8) bytes.
        """
        # Safety: build projector lazily if init_client() was not called
        if self.projector is None:
            self._init_projector()

        # Round 0: no consensus received yet → use neutral zero vector
        if self.v is None:
            self.v = torch.zeros(self.m, device=self.device)

        self.model.train()

        for _epoch in range(self.epochs):
            for x, labels in self.client_train_loader:
                x, labels = x.to(self.device), labels.to(self.device)

                # ── (a) Task-loss gradient via autograd ───────────────────
                self.optimizer.zero_grad()
                outputs   = self.model(x)
                task_loss = self.criterion(outputs, labels).mean()
                task_loss.backward()
                # p.grad now holds ∇f̂_k for every parameter p.

                # ── (b) Sign-regulariser gradient  λ·Φᵀ(tanh(γΦw)−v) ─────
                # ── (c) L2 penalty gradient  μ·w ─────────────────────────
                # Both are computed analytically (no autograd), then added
                # to the existing p.grad so optimizer.step() sees the full
                # gradient ∇F̃_k = ∇f̂_k + λ·reg_grad + l2_grad.
                with torch.no_grad():
                    w = _model_to_vec(self.model)            # flat copy, shape (n,)

                    phi_w    = self.projector.forward(w)                     # (m,)
                    residual = torch.tanh(self.gamma * phi_w) - self.v      # (m,)
                    reg_grad = self.projector.backward(residual)             # (n,)
                    l2_grad  = self.mu * w                                   # (n,)

                    # Write into p.grad, parameter by parameter
                    offset = 0
                    for p in self.model.parameters():
                        numel = p.numel()
                        extra = (
                            self.lam * reg_grad[offset: offset + numel].view_as(p)
                            + l2_grad[offset: offset + numel].view_as(p)
                        )
                        if p.grad is None:
                            p.grad = extra.clone()
                        else:
                            p.grad.data.add_(extra)
                        offset += numel

                # ── (d) Optimizer step:  w ← w − η · ∇F̃_k ───────────────
                self.optimizer.step()

        # ── Compute one-bit sketch  z = sign(Φw)  (entire up-link payload) ─
        with torch.no_grad():
            w     = _model_to_vec(self.model)
            phi_w = self.projector.forward(w)       # (m,)
            z     = torch.sign(phi_w)               # values in {-1, 0, +1}
            z[z == 0] = 1.0                         # tie-break: 0 → +1

        return z