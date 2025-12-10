import torch
import torch.nn as nn
import numpy as np

import sys
sys.path.append("../")

from utils import ensure_torch, ensure_numpy

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

import math
@torch.no_grad()
# compute_monomial_param_proxy
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


@torch.no_grad()
# T_mixed_bias
def T_mixed_bis(model, data_eigvals, monomial, *, eps: float = 1e-12,
                 normalize: str = "unsigned_rms_calib", forward_gain: float = 1.0, preact_gain: float = 1.0,
                 **kwargs):
    # --- helpers (device-safe) ---
    def _phi(x): return torch.exp(-0.5 * x * x) / x.new_tensor((2.0*torch.pi)**0.5)
    def _Phi(x): return 0.5 * (1.0 + torch.erf(x / x.new_tensor(2.0**0.5)))
    def _C_n(bhat, n: int):
        if n == 0: return _phi(bhat) + bhat * _Phi(bhat)
        if n == 1: return _Phi(bhat)
        if n == 2: return 0.5 * _phi(bhat)
        if n == 3: return -(bhat * _phi(bhat)) / 6.0
        if n == 4: return ((3.0*bhat*bhat - 6.0) * _phi(bhat)) / 24.0
        raise NotImplementedError("C_n implemented for n=0..4 only.")

    data_eigvals = ensure_torch(data_eigvals)
    Win, b_in = get_Win(model)
    Wout, _   = get_Wout(model)
    a = Wout.view(-1)
    _, d = Win.shape

    if kwargs.get("force_monomial", False):
        monomial = kwargs["force_monomial"]
    # print(monomial)
    basis = monomial.basis() if hasattr(monomial,"basis") else monomial
    axes_sorted = sorted(basis.keys())
    powers_sorted = [int(basis[ax]) for ax in axes_sorted]
    axes_sorted = [int(ax) for ax in axes_sorted]
    
    idx = torch.as_tensor([ax for ax in axes_sorted],
                          device=Win.device, dtype=torch.long)
    if not ((idx >= 0).all() and (idx < d).all()):
        raise IndexError("Axis out of range.")

    # --- build u, α, û, b̂ ---
    W_sub = Win[:, idx]
    gamma_sqrt = data_eigvals[idx].sqrt().to(Win.dtype)
    u = (W_sub * gamma_sqrt) * preact_gain

    alpha = u.norm(dim=1) # scaled pre-activation norm per input axis
    mask  = (alpha > eps).to(Win.dtype)
    uhat  = torch.where(alpha[:,None] > eps, u / (alpha[:,None] + eps), torch.zeros_like(u))
    bhat  = torch.where(alpha > eps, b_in / (alpha + eps), torch.zeros_like(alpha))

    n_total = int(sum(powers_sorted))
    Cn = _C_n(bhat, n_total)

    exps = torch.as_tensor(powers_sorted, device=Win.device, dtype=Win.dtype)
    axis_factor = torch.pow(uhat, exps).prod(dim=1)
    
    # contrib = a * (alpha**n_total) * Cn * axis_factor * mask

    contrib = a * alpha * Cn * axis_factor * mask
    
    numer   = contrib.sum()

    def _psi_for(order, alpha, bhat, uhat_i, mask):
        # α^1 homogeneity, per-order C_k, per-axis angular power
        # assumes you have _C_n(bhat, n) already
        return (alpha * _C_n(bhat, order) * (uhat_i ** order) * mask)

    if normalize == "abs_corr_orth":
        # build lower-order ψ's for this axis i
        u_i = uhat[:, 0]               # the selected axis (one-axis probe)
        # cache ψ_ℓ for ℓ = 1..k
        k = int(sum(powers_sorted))
        psis = [ _psi_for(ell, alpha, bhat, u_i, mask) for ell in range(1, k+1) ]

        # Gram–Schmidt: orthogonalize ψ_k against ψ_1..ψ_{k-1}
        psi_k = psis[-1]
        for psi_l in psis[:-1]:
            denom = (psi_l @ psi_l).clamp_min(1e-20)
            psi_k = psi_k - ((psi_k @ psi_l) / denom) * psi_l

        # correlation with a (scale-free; no forward_gain)
        num  = (a * psi_k).sum()
        den  = (a.norm() * psi_k.norm()).clamp_min(1e-20)
        return ensure_numpy((num.abs() / den))


    if normalize == "none":
        return ensure_numpy(forward_gain * numer)
    
    if normalize == "unsigned_l1":
        return ensure_numpy(forward_gain * contrib.abs().sum())

    if normalize == "unsigned_rms":
        return ensure_numpy(forward_gain * (torch.linalg.vector_norm(contrib))**(1/n_total))

    if normalize == "unsigned_rms_calib":
        k = int(n_total)
        valid = (alpha > eps)
        if valid.sum() == 0:
            return ensure_numpy(torch.tensor(0.0))

        # --- 1) ReLU coefficient calibration (data-driven, per step) ---
        # RMS over active units of C_k(b̂); also compute for a reference order (default 1 or 3)
        bh = bhat[valid]
        # C1(b̂) = Φ(b̂), Cn already computed for this k
        Phi = 0.5 * (1.0 + torch.erf(bh / (2.0**0.5)))
        C1  = Phi
        Ck  = Cn[valid]

        rms = lambda x: torch.sqrt((x*x).mean().clamp_min(1e-20))
        Rk  = rms(Ck)
        Rref = rms(C1)         # set reference order = 1; see note below

        # --- 2) Angular calibration (theoretical baseline) ---
        # μ_{d,p} = E[|V_1|^p], V ~ Unif(S^{d-1})
        d = Win.shape[1]
        def mu_d(p):
            return torch.exp(
                torch.lgamma(torch.tensor((p+1)/2, device=Win.device, dtype=Win.dtype)) +
                torch.lgamma(torch.tensor(d/2,       device=Win.device, dtype=Win.dtype)) -
                0.5*torch.log(torch.tensor(torch.pi, device=Win.device, dtype=Win.dtype)) -
                torch.lgamma(torch.tensor((d+p)/2,   device=Win.device, dtype=Win.dtype))
            )

        # overall scalar to put order-k on the same nominal scale as the reference
        s_k = (Rref / Rk) * torch.sqrt( (mu_d(2) / mu_d(2*k)).clamp_min(1e-20) )

        val = s_k * torch.linalg.vector_norm(contrib)
        return ensure_numpy(forward_gain * val)

    # correlation-style (scale/gauge/width invariant)
    if normalize in ("corr", "abs_corr"):
        den_a  = a.norm() + eps
        den_w  = torch.linalg.vector_norm((alpha**n_total) * Cn * axis_factor * mask) + eps
        val = numer / (den_a * den_w)
        if normalize == "abs_corr":
            val = val.abs()
        return ensure_numpy(forward_gain * val)

    raise ValueError("normalize must be 'none' | 'corr' | 'abs_corr'")