import torch
import gc
import importlib

from MLPscape.backend.utils import seed_everything, derive_seed, _extract_kwargs_for, ensure_torch
from MLPscape.model import MLP
from MLPscape.backend.trainloop import train_MLP

def load_fn_from_file(path, name):
    """
    Load a function `name` from a Python source file at `path`.
    This does NOT rely on the module being on PYTHONPATH.
    """
    spec = importlib.util.spec_from_file_location("bfn_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, name)


def get_base_bfn(bfn_config):
    if "base_bfn" in bfn_config:
        return bfn_config["base_bfn"]  # actual callable
    if "bfn_file" in bfn_config and "bfn_name" in bfn_config:
        return load_fn_from_file(bfn_config["bfn_file"], bfn_config["bfn_name"])
    raise ValueError("No batch function specified in bfn_config")


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
    
    iter_spec = {iterator_names[i]: job[i] for i in range(2, len(iterator_names))}
    global_config.update(iter_spec)

    torch.set_num_threads(1)  # avoid CPU contention when many procs

    bfn_config_copy = bfn_config.copy()
    base_bfn = get_base_bfn(bfn_config_copy)
    for key in ("base_bfn", "bfn_file", "bfn_name"):
        bfn_config_copy.pop(key, None)

    def make_bfn(n, X, y, **kwargs):
        return base_bfn(**bfn_config_copy, bsz=n, gen=GEN, X=X, y=y, **kwargs)

    bfn_iter_args, _ = _extract_kwargs_for(base_bfn, iter_spec)
    
    if global_config.get("X_te", None) is not None and global_config.get("y_te", None) is not None:
        X_te = ensure_torch(global_config["X_te"])
        y_te = ensure_torch(global_config["y_te"])
    else:
        X_te, y_te = make_bfn(n=global_config["N_TEST"], X=None, y=None, **bfn_iter_args)(0)
    X_tr, y_tr = make_bfn(job[0], X=None, y=None, **bfn_iter_args)(job[1]) if not global_config["ONLINE"] else None, None

    bfn = make_bfn(job[0], X=X_tr, y=y_tr, **bfn_iter_args)
    
    global_config["d_in"] = global_config["DIM"] if "DIM" in global_config else global_config["dim"] if "dim" in global_config else global_config["d_in"]
    mlp_kwargs, _ = _extract_kwargs_for(MLP, global_config)
    model = MLP(**mlp_kwargs).to(device)

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
        **global_config,
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