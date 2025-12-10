import torch
import gc

from .utils import seed_everything, derive_seed
from model import MLP
from .trainloop import train_MLP
from .batch_functions import BATCH_FNS

def run_job(device_id, job, global_config, bfn_config, iterator_names, **kwargs):
    """
    job: (n, trial, else)
    global_config: anything read-only you want to avoid capturing from globals
    bfn: batch function
    """
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")

    base_seed = global_config.get("SEED", None)
    job_seed  = derive_seed(base_seed, device_id)
    GEN, RNG = seed_everything(job_seed, device_id)

    global_config.update({iterator_names[i]: job[i] for i in range(2, len(iterator_names))})

    torch.set_num_threads(1)  # avoid CPU contention when many procs

    bfn_config_copy = bfn_config.copy()
    name = bfn_config_copy.pop("bfn_name")#["bfn_name"]
    base_bfn = BATCH_FNS[name]
    bfn = lambda n, X, y, gen, target_monomial: base_bfn(**bfn_config_copy, monomials=target_monomial, bsz=n, gen=gen, X=X, y=y)
    
    X_te, y_te = bfn(n=global_config["N_TEST"], X=None, y=None, gen=GEN, **{iterator_names[i]: job[i] for i in range(2, len(iterator_names))})(0)
    
    X_tr, y_tr = bfn(job[0], X=None, y=None, gen=GEN, **{iterator_names[i]: job[i] for i in range(2, len(iterator_names))})(job[1]) if not global_config["ONLINE"] else None, None

    bfn = bfn(job[0], X=X_tr, y=y_tr, gen=GEN, **{iterator_names[i]: job[i] for i in range(2, len(iterator_names))})
    
    model = MLP(d_in=global_config["DIM"], depth=global_config["DEPTH"],
                d_out=1, width=global_config["WIDTH"]).to(device)

    outdict = train_MLP(
        model=model,
        batch_function=bfn,
        lr=global_config["LR"],
        max_iter=int(global_config["MAX_ITER"]),
        loss_checkpoints=global_config["LOSS_CHECKPOINTS"],
        gamma=global_config["GAMMA"],
        ema_smoother=global_config["EMA_SMOOTHER"],
        only_thresholds=global_config["ONLYTHRESHOLDS"],
        verbose=global_config["VERBOSE"],
        X_tr=X_tr, y_tr=y_tr,
        X_te=X_te, y_te=y_te,
        otherreturns=global_config.get("otherreturns", {}),
        **kwargs,
    )

    timekeys = outdict["timekeys"]
    train_losses = outdict["train_losses"]
    test_losses = outdict["test_losses"]
    otherouts = [outdict[k] for k in global_config.get("otherreturns", {}).keys()]
    # Cleanup GPU memory
    del model, X_tr, y_tr, X_te, y_te, outdict
    torch.cuda.empty_cache()
    gc.collect()

    return (job, timekeys, train_losses, test_losses, *otherouts)