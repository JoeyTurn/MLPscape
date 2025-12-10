import torch
import gc

from backend.utils import seed_everything, derive_seed
from model import MLP
from backend.trainloop import train_MLP

def run_job(device_id, job, global_config, bfn=None, **kwargs):
    """
    job: (target, n, trial)
    global_config: anything read-only you want to avoid capturing from globals
    bfn: batch function
    """
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")

    base_seed = global_config.get("SEED", None)
    job_seed  = derive_seed(base_seed, device_id)
    GEN, RNG = seed_everything(job_seed, device_id)

    torch.set_num_threads(1)  # avoid CPU contention when many procs

    X_te, y_te = bfn(n=global_config["N_TEST"], X=None, y=None, **kwargs)(0)
    
    X_tr, y_tr = bfn(job[0], X=None, y=None, **kwargs)(job[1]) if not global_config["ONLINE"] else None, None

    bfn = bfn(job[0], X=X_tr, y=y_tr, **kwargs)
    
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

    return (timekeys, train_losses, test_losses, *otherouts)