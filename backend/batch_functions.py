from .utils import ensure_torch
import torch
from data.data import get_new_polynomial_data

def polynomial_batch_fn(lambdas, Vt, monomials, bsz, data_eigvals, N=10,
                  X=None, y=None, data_creation_fn=get_new_polynomial_data, gen=None):
    lambdas, Vt, data_eigvals = map(ensure_torch, (lambdas, Vt, data_eigvals))
    dim = len(data_eigvals)
    def batch_fn(step: int, X=X, y=y):
        if (X is not None) and (y is not None):
            X_fixed = ensure_torch(X)
            y_fixed = ensure_torch(y)
            return X_fixed, y_fixed
        with torch.no_grad():
            dcf_args = dict(lambdas=lambdas, Vt=Vt, monomials=monomials, dim=dim,
                            N=bsz, data_eigvals=data_eigvals, N_original=N, gen=gen)
            X, y = data_creation_fn(**dcf_args)
        X, y = map(ensure_torch, (X, y))
        return X, y
    
    return batch_fn

BATCH_FNS = {
    "polynomial_batch_fn": polynomial_batch_fn,
}