import numpy as np
import torch
from mupify import mupify, rescale
from model import centeredMLP

def train_MLP(model, batch_function, lr=1e-2, max_iter=int(1e3), loss_checkpoints=None, percent_thresholds=None,
                  gamma=1., ema_smoother=0.0, X_tr=None, y_tr=None, X_te=None, y_te=None, only_thresholds=False,
                  verbose=False, **kwargs):
    """
    Returns:
        dict of model, train_losses, test_losses, timekeys, others

    - timekeys[j]: first gradient step where the loss drops below the j-th threshold.
    - If RELATIVE mode, thresholds are `percent_thresholds * init_loss`.
    - If ABSOLUTE mode, thresholds are raw absolutes and the comparison metric is raw loss.
    - If only_thresholds is True, only returns the timekeys and not the full loss curves.
    """

    def return_statement(model, ema_tr, ema_te, timekeys, tr_losses=None, te_losses=None, extras={}):
        if only_thresholds:
            extras.update({"model": model, "train_losses": ema_tr, "test_losses": ema_te, "timekeys": timekeys})
            return extras
        # if otherreturns is not None:
        #     for name, _ in items:
        #         extras[name] = np.array(extras[name])
        extras.update({"model": model, "train_losses": tr_losses, "test_losses": te_losses, "timekeys": timekeys})
        return extras 

    def fill_losses(tr_losses, te_losses, extras, i):
        tr_losses[i:] = tr_losses[i]
        te_losses[i:] = te_losses[i]
        if otherreturns is not None:
            for name, _ in list(otherreturns.items()):
                extras[name][i:] = extras[name][i]
        return tr_losses, te_losses, extras
    
    #checking if losses are absolute or relative
    has_abs = (loss_checkpoints is not None) and len(loss_checkpoints) > 0
    is_relative = (percent_thresholds is not None) and len(percent_thresholds) > 0
    if has_abs and is_relative or not has_abs and not is_relative:
        raise ValueError("Provide exactly one of loss_checkpoints OR percent_thresholds.")

    # model stuff
    lr = lr * gamma if gamma >= 1 else lr * (gamma**2.)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    mupify(model, opt, param="mup")
    rescale(model, gamma)
    model = centeredMLP(model).to(next(model.parameters()).device)
    loss_fn = torch.nn.MSELoss()

    # thresholding
    thresholds = np.asarray(percent_thresholds if is_relative else loss_checkpoints, dtype=float)
    thresholds = np.sort(thresholds)[::-1] # descending
    timekeys = np.full(thresholds.shape, 0, dtype=int)

    otherreturns = kwargs.get("otherreturns", None)
    if type(otherreturns) is dict:
        items = list(otherreturns.items())
    extras = {}

    if not(only_thresholds):
        tr_losses = np.empty(max_iter, dtype=float)
        te_losses = np.empty(max_iter, dtype=float)
        # if otherreturns is not None:
        #     for name, _ in items:
                # extras[name] = np.empty((max_iter, 200, 200), dtype=float) #temporary, this is gonna need to be CHANGED
                # extras[name] = []#np.empty(max_iter, dtype=float)
    else:
        tr_losses = te_losses = None
        if otherreturns is not None:
            for name, _ in items:
                extras[name] = None
    ema = None
    pointer = 0   
    X_tr_not_provided = X_tr is None

    # training loop 
    # print("Starting training loop...")
    for i in range(int(max_iter)):
        X_tr, y_tr = batch_function(i)
        
        opt.zero_grad()
        out = model(X_tr).squeeze()
        loss = loss_fn(out, y_tr.squeeze())
        loss.backward()
        opt.step()

        tr_loss_val = float(loss.item())
        if X_tr_not_provided:
            te_loss_val = tr_loss_val
        else:
            with torch.no_grad():
                out = model(X_te).squeeze()
                loss = loss_fn(out, y_te.squeeze())
                te_loss_val = float(loss.item())
        # initialize thresholds & loss trace baseline at first step
        if i == 0:
            ema_tr = tr_loss_val
            ema_te = te_loss_val
            if is_relative:
                thresholds *= tr_loss_val
            # prefill losses after init val calculated
            if not(only_thresholds):
                if otherreturns is not None:
                    for name, fn in items:
                        val = np.array(fn(model=model.model, X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te, **kwargs)) #the model.model to unwrap from centeredMLP
                        shape = (max_iter,) + val.shape
                        extras[name] = np.empty(shape, dtype=val.dtype)
                        # extras[name][:]= val
                tr_losses, te_losses, extras=fill_losses(tr_losses, te_losses, extras, i)

        ema_tr = (ema_smoother * ema_tr + (1.0 - ema_smoother) * tr_loss_val)
        ema_te = (ema_smoother * ema_te + (1.0 - ema_smoother) * te_loss_val)
        if not(only_thresholds):
            tr_losses[i] = ema_tr
            te_losses[i] = ema_te
            if otherreturns is not None:
                for name, fn in items:
                    val = fn(model=model.model, X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te, monomial=kwargs.get("monomial", None)) #the model.model to unwrap from centeredMLP
                    extras[name][i] = val
                    
        if verbose:
            print(f"Step {i}: ema train loss {ema_tr}, ema test loss {ema_te}")

        # if a threshold is hit, update pointer
        while pointer < len(thresholds) and ema_tr < thresholds[pointer]:
            timekeys[pointer] = i
            if not(only_thresholds):
                tr_losses, te_losses, extras=fill_losses(tr_losses, te_losses, extras, i)
            pointer += 1

        # early exit if all thresholds crossed
        if pointer == len(thresholds):
            if otherreturns is not None and only_thresholds:
                for name, fn in items:
                    val = fn(model=model.model, X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te, **kwargs)
                    extras[name] = val
            return return_statement(model, ema_tr, ema_te, timekeys, tr_losses, te_losses, extras)

    return return_statement(model, ema_tr, ema_te, timekeys, tr_losses, te_losses, extras)