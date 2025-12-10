## relu ntk level coeffs

import numpy as np
from backend.utils import ensure_numpy

@staticmethod
def kernel_parse_kwargs(kwargs):
    if ("weight_variance" in kwargs and "bias_variance" in kwargs) or ("w_var" in kwargs and "b_var" in kwargs):
        w_var = kwargs.get("weight_variance", kwargs.get("w_var", 1))
        b_var = kwargs.get("bias_variance",   kwargs.get("b_var", 1))
    else:
        k_width = kwargs.get("kernel_width", 1.0)
        w_var = np.sqrt((2*np.pi)/(1 + 2*np.pi*k_width))
        b_var = k_width * w_var
    return w_var, b_var

@staticmethod
def get_relu_level_coeff_fn(data_eigvals, **kwargs):
    w_var, b_var = kernel_parse_kwargs(kwargs)

    rho = ensure_numpy(data_eigvals).sum()
    q = b_var + w_var * rho
    c0 = b_var / q
    if abs(c0) >= 1.0:
        raise ValueError("|c0|=1 makes higher derivatives singular; ensure w_var>0.")

    pref = w_var * q / (2*np.pi)
    scale = w_var / q
    
    def _poly_add(a, b, sa=1.0, sb=1.0):
        n = max(len(a), len(b))
        out = [0.0]*n
        for i in range(n):
            va = a[i] if i < len(a) else 0.0
            vb = b[i] if i < len(b) else 0.0
            out[i] = sa*va + sb*vb
        return out

    def _poly_eval(cs, c):
        p = 0.0
        for k in reversed(range(len(cs))):
            p = p*c + cs[k]
        return p

    def eval_level_coeff(ell):
        P = [3.0, 0.0, -2.0]
        if ell == 0:
            c = float(np.clip(c0, -1.0, 1.0))
            Gk_at_c0 = np.sqrt(max(0.0, 1.0 - c**2)) + 2.0*(np.pi - np.arccos(c))*c
        elif ell == 1:
            c = float(np.clip(c0, -1.0, 1.0))
            denom = np.sqrt(max(0.0, 1.0 - c**2))
            Gk_at_c0 = 2.0*(np.pi - np.arccos(c)) + (c/denom if denom > 0 else float('inf'))
        elif ell == 2:
            Gk_at_c0 = _poly_eval(P, c0) / (1.0 - c0*c0)**1.5
        else:
            for cur_ell in range(2, ell):
                term1 = [k*P[k] for k in range(1, len(P))]
                term1 = _poly_add(term1, [0.0, 0.0] + term1, 1.0, -1.0)
                term2 = [0.0] + list(P)
                P = _poly_add(term1, term2, 1.0, (2*cur_ell - 1))
            Gk_at_c0 = _poly_eval(P, c0) / (1.0 - c0*c0)**(ell - 0.5)
        return pref * (scale**ell) * float(Gk_at_c0)
    
    return eval_level_coeff