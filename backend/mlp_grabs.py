import torch
import torch.nn as nn
import numpy as np

from .utils import ensure_torch, ensure_numpy

def get_Win(model: nn.Module, *, detach: bool = True, **kwargs):
    Win = model.input_layer.weight
    b_in = model.input_layer.bias

    if detach:
        Win = Win.detach().clone()
        b_in = b_in.detach().clone() if b_in is not None else None

    return Win, b_in


def get_Wout(model: nn.Module, *, detach: bool = True, **kwargs):
    Wout = model.output_layer.weight
    bout = model.output_layer.bias

    if detach:
        Wout = Wout.detach().clone()
        bout = bout.detach().clone() if bout is not None else None

    return Wout, bout


def get_W_gram(W: torch.Tensor, concatenate_outside: bool = True, **kwargs):
    """
    Concatenate_outside: if True, computes W^T W (so gram matrix in output space)
    """
    return W.T @ W if concatenate_outside else W @ W.T


def get_W_ii(W: torch.Tensor, i: int=None, monomial=None, **kwargs):
    """
    So Nintendo doesn't sue us.

    Assumes W is a gram matrix.
    """

    if monomial is not None and i is None:
        eyes = [int(k) for k in monomial.basis().keys()]
        return [W[i, i].item() for i in eyes]
    Wii = W[i, i] #grab i if specified
    return Wii.item()


def get_W_trace(W: torch.Tensor, **kwargs):
    """
    Check the trace of W_ii.

    Assumes W is a gram matrix.
    """
    return torch.trace(W).item()


def empirical_ntk(model: nn.Module,
                  X: torch.Tensor,
                  create_graph: bool = False) -> torch.Tensor:
    """
    Compute empirical NTK matrix K_ij = <∂θ f(x_i), ∂θ f(x_j)>
    for a scalar-output model.

    Args
    ----
    model : nn.Module
        PyTorch model; output must be scalar per example.
    X : torch.Tensor
        Input data of shape [N, d]. Must require grad on params, not on X.
    create_graph : bool
        If True, keep graph for higher-order derivatives (usually False).

    Returns
    -------
    K : torch.Tensor
        NTK matrix of shape [N, N].
    """
    model.eval()  # just to be safe; no dropout/bn updates
    device = next(model.parameters()).device
    X = X.to(device)

    # collect parameters we differentiate w.r.t.
    params = [p for p in model.parameters() if p.requires_grad]
    num_params = sum(p.numel() for p in params)

    N = X.shape[0]
    J = torch.zeros(N, num_params, device=device)

    # build Jacobian row-by-row
    offset_slices = []
    start = 0
    for p in params:
        n = p.numel()
        offset_slices.append(slice(start, start + n))
        start += n

    for i in range(N):
        model.zero_grad(set_to_none=True)
        out_i = model(X[i:i+1]).squeeze()  # scalar
        # d out_i / d theta
        grads = torch.autograd.grad(
            out_i,
            params,
            retain_graph=False,
            create_graph=create_graph,
            allow_unused=False
        )
        # flatten and stuff into J[i]
        row = []
        for g in grads:
            row.append(g.reshape(-1))
        J[i] = torch.cat(row, dim=0)

    # NTK = J J^T
    K = J @ J.t()
    return K



import math
@torch.no_grad()
def T_mixed_bias(model, data_eigvals, monomial, num_gh=32, **kwargs):
    """
    Parameter-space proxy for the Hermite coefficient along a given monomial.

    Model assumed:
        f(x) = (1/sqrt(m)) * sum_j a_j * ReLU(w_j^T x + b_j),
        x ~ N(0, diag(gamma)).

    Monomial:
        monomial: dict-like mapping {dim_index: degree}
          - {0:1}      ~ h_1(e_0)
          - {0:1,3:2}  ~ h_1(e_0) * h_2(e_3)
        All unspecified coords are understood to have degree 0 (h_0 = 1).

    gamma:
        1D tensor of length d: diagonal entries of Γ, where x ~ N(0, Γ).

    Returns:
        C_mono: scalar tensor, approximate parameter-space proxy for
                E[f(e) * Π_i h_{α_i}(e_i)]
                for the given monomial α.

    Notes:
        - For monomials involving a *single* coordinate (e.g. {i:k}), this
          reduces to the more principled single-coordinate derivation:
              C_{i,k} ≈ E[f(e) h_k(e_i)].
        - For monomials involving multiple coordinates, this uses a
          factorized approximation:
              E[ψ_κ(g) Π_i h_{α_i}(e_i)]
              ≈ σ * Π_i [ t_{α_i}(κ) ρ_i^{α_i} α_i! ]
          where ψ_κ(g) = ReLU(κ+g), g ~ N(0,1),
                ρ_i = Corr(e_i, g), and t_k(κ) is the 1D Hermite coefficient.
          This keeps everything parameter-only but is an approximation.
    """
    # ------------------------------------------------------------------
    # 1. Extract weights from your model using your helpers
    # ------------------------------------------------------------------
    Win, b_in = get_Win(model)      # Win: [m, d], b_in: [m]
    Wout, _   = get_Wout(model)     # Wout: [1, m] or [m, 1] etc.
    a = Wout.view(-1)               # [m]

    device = Win.device
    dtype = Win.dtype

    if kwargs.get("force_monomial", False):
        monomial = kwargs["force_monomial"]

    W = Win.to(device=device, dtype=dtype)          # [m, d]
    b = b_in.to(device=device, dtype=dtype)         # [m]
    a = a.to(device=device, dtype=dtype)            # [m]
    gamma = ensure_torch(data_eigvals)

    m, d = W.shape
    assert gamma.shape[0] == d, "gamma must be length d (diag entries of Γ)."

    # Clean monomial: drop any zero-degree entries just in case
    mono = {int(i): int(k) for i, k in monomial.items() if int(k) != 0}
    if len(mono) == 0:
        # The "monomial" is just h_0 everywhere; the coefficient is just E[f],
        # which is zero for symmetric setups. Return 0 for now.
        return torch.zeros((), device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # 2. Data-aware neuron geometry: v_j, σ_j, κ_j, ρ_{j,i}
    # ------------------------------------------------------------------
    sqrt_gamma = torch.sqrt(gamma)        # [d]
    v = W * sqrt_gamma                    # [m, d]   v_j = Γ^{1/2} w_j

    eps = 1e-12
    sigma = torch.linalg.norm(v, dim=1) + eps   # [m], std of z_j under x~N(0,Γ)
    kappa = b / sigma                           # [m], normalized bias μ_j / σ_j

    # ρ_{j,i} = Corr(e_i, z_j) = v_{j,i} / σ_j
    rho = v / sigma.unsqueeze(1)                # [m, d]

    # ------------------------------------------------------------------
    # 3. Gauss–Hermite quadrature in 1D for t_k(κ)
    #    t_k(κ) = (1/k!) * E_G[ ReLU(κ + G) * h_k(G) ],  G ~ N(0,1)
    # ------------------------------------------------------------------
    degrees = sorted(set(mono.values()))
    max_degree = max(degrees)

    # hermgauss for weight e^{-x^2}; convert to standard normal
    nodes, weights = np.polynomial.hermite.hermgauss(num_gh)
    nodes_t = torch.from_numpy(nodes).to(device=device, dtype=dtype)     # [num_gh]
    weights_t = torch.from_numpy(weights).to(device=device, dtype=dtype) # [num_gh]

    # Map to standard normal: G = sqrt(2) * X
    g = math.sqrt(2.0) * nodes_t     # [num_gh]

    # Probabilists' Hermite polynomials He_n(g), n = 0..max_degree
    herm = torch.zeros((max_degree + 1, g.shape[0]),
                       device=device, dtype=dtype)
    herm[0] = 1.0
    if max_degree >= 1:
        herm[1] = g
    for n in range(1, max_degree):
        herm[n + 1] = g * herm[n] - n * herm[n - 1]

    # ψ_κ(g) = ReLU(κ + g), for each neuron j and GH node
    # psi[j,ℓ] = ReLU(kappa_j + g_ℓ)
    psi = torch.relu(kappa.unsqueeze(1) + g.unsqueeze(0))  # [m, num_gh]

    # Convert hermgauss weights to standard normal expectation:
    # E_G[f(G)] ≈ sum_ℓ w_ℓ / √π * f(√2 x_ℓ)
    weight_factor = weights_t / math.sqrt(math.pi)  # [num_gh]

    # Compute t_k(κ_j) for all k that appear in the monomial
    t_by_k = {}
    for k in degrees:
        hk = herm[k]  # [num_gh]
        # E[ψ_κ(G) * He_k(G)] for each neuron j, via quadrature:
        E = (psi * (hk * weight_factor).unsqueeze(0)).sum(dim=1)  # [m]
        t_kappa = E / math.factorial(k)                           # [m]
        t_by_k[k] = t_kappa

    # ------------------------------------------------------------------
    # 4. Combine into monomial coefficient C_mono
    #
    #    Approximate per-neuron contribution as:
    #      contrib_j ≈ (1/√m) * a_j * σ_j *
    #                  Π_{(i,k) in monomial} [ t_k(κ_j) * ρ_{j,i}^k * k! ]
    #
    #    Then C_mono = Σ_j contrib_j.
    # ------------------------------------------------------------------
    # sqrt_m = math.sqrt(m)

    # Start with a_j * σ_j / sqrt(m)
    contrib = torch.abs(a) * sigma*m**(0.5)#/model.output_layer._multiplier# / sqrt_m    # [m]

    for i, k in mono.items():
        t_kappa = t_by_k[k]         # [m]
        rho_i = rho[:, i]           # [m]
        # factor = t_k(κ_j) * ρ_{j,i}^k * k!
        factor = t_kappa * (rho_i ** k) * math.factorial(k)
        contrib = contrib * factor  # [m]

    C_mono = contrib.sum()          # scalar

    return ensure_numpy(C_mono)