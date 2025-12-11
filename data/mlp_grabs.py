import torch
import torch.nn as nn
import numpy as np

from backend.utils import ensure_torch, ensure_numpy

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
                  X_tr: torch.Tensor,
                  create_graph: bool = False, **kwargs) -> torch.Tensor:
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
    model.eval()
    device = next(model.parameters()).device
    X = ensure_torch(X_tr)

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
    return ensure_numpy(K)

def empirical_ntk_trace(model: nn.Module,
                  X_tr: torch.Tensor,
                  create_graph: bool = False, **kwargs) -> torch.Tensor:
    return empirical_ntk(model, X_tr, create_graph=create_graph, **kwargs).trace()