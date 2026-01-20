import numpy as np
import torch
from torch.multiprocessing import get_context
import torch.multiprocessing as mp
from tqdm import tqdm
from itertools import product

import inspect
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from ExptTrace import ExptTrace

from MLPscape.backend.worker import worker
from MLPscape.backend.job import run_job


def normalize_bfn_config(bfn_config, use_mp=False):
    cfg = dict(bfn_config)  # shallow copy

    if "base_bfn" in cfg and use_mp:
        fn = cfg.pop("base_bfn")

        source_file = inspect.getsourcefile(fn)
        if source_file is None:
            raise RuntimeError(
                f"Can't locate source file for base_bfn {fn}. "
                "Is it defined in an interactive session / notebook?"
            )

        cfg["bfn_file"] = os.path.abspath(source_file)
        cfg["bfn_name"] = fn.__name__
        
    return cfg


def sanitize_expt_trace(obj):
    """
    Recursively walk `obj` and replace any value whose type is not in
    ALLOWED_TYPES with the literal string 'str'.
    """
    ALLOWED_TYPES = (int, float, str, tuple, np.integer, np.floating)

    if isinstance(obj, tuple):
        return tuple(sanitize_expt_trace(v) for v in obj)

    # Base case
    if isinstance(obj, ALLOWED_TYPES):
        return obj
    else:
        return str(obj)
    

## --- Multiprocessing execution ---
def main(iterators, iterator_names=None, global_config=None, bfn_config=None,
         use_mp: bool | None = None):
    """
    Run jobs across multiple iterators in parallel.

    Args:
        iterators: list of list-like objects, e.g. [targets, sample_sizes, trials]
        iterator_names: optional list of strings naming each iterator, e.g. ["target", "ntrain", "trial"].
                       If None, auto-generates names like "iter_0", "iter_1", etc.
        global_config: configuration object with attributes like other_model_grabs, ONLYTHRESHOLDS, etc.

    Returns:
        dict with keys: "jobs", "var_axes", "losses", "timekeys", "extras"
    """
    if iterator_names is None:
        iterator_names = [f"iter_{i}" for i in range(len(iterators))]
    elif len(iterator_names) != len(iterators):
        raise ValueError(f"Length of iterator_names ({len(iterator_names)}) must match iterators ({len(iterators)})")

    #gen all jobs
    jobs = list(product(*iterators))

    var_axes = list(iterator_names)
    et_losses = ExptTrace(var_axes)
    et_timekeys = ExptTrace(var_axes)
    
    # Get grab aliases from global_config if available
    grab_aliases = list(global_config.get("otherreturns", {}).keys()) if global_config is not None else []
    et_extras = {alias: ExptTrace(var_axes) for alias in grab_aliases}
    
    if use_mp is None:
        # If the user explicitly called mp.set_start_method("spawn", ...)
        # somewhere before, this will be 'spawn'; otherwise None.
        start_method = mp.get_start_method(allow_none=True)
        use_mp = (start_method == "spawn")

    bfn_config = normalize_bfn_config(bfn_config, use_mp=use_mp)
    done = 0

    ### No multiprocessing

    # --- single-process path (good for ipynb-local bfns) ---
    if not use_mp:
        total = len(jobs)
        with tqdm(total=total, desc="Runs", dynamic_ncols=True) as pbar:
            for job in jobs:
                try:
                    dev = 0
                    payload = run_job(dev, job, global_config, bfn_config, iterator_names)
                    kind, out = "ok", payload
                except Exception as e:
                    import traceback
                    tb = traceback.format_exc()
                    kind, out = "err", (job, repr(e), tb)

                if kind == "ok":
                    job, timekeys, train_losses, test_losses, *others = payload
                    job = sanitize_expt_trace(job)

                    et_losses[job] = test_losses
                    et_timekeys[job] = timekeys
                    
                    for kidx, k in enumerate(grab_aliases):
                        et_extras[k][job] = others[kidx]
                    
                    if not(global_config["ONLYTHRESHOLDS"]):
                        train_losses = train_losses[-1]
                        test_losses = test_losses[-1]
                    
                    job_str = " | ".join([f"{name}={val}" for name, val in zip(iterator_names, job)])
                    pbar.set_postfix_str(
                        f"train {test_losses:.3g} | test {train_losses:.3g} | timekey {timekeys} | {job_str}",
                        refresh=False
                    )
                else:
                    print(f"[ERROR] {out[0]}: {out[1:]}")

                done += 1
                pbar.update(1)

        result = {
        "jobs": jobs,
        "var_axes": var_axes,
        "losses": et_losses.serialize(),
        "timekeys": et_timekeys.serialize(),
        "extras": {name: et_extras[name].serialize() for name in grab_aliases},
        }

        return result

    ### With multiprocessing

    # Set up multiprocessing context and queues
    ctx = get_context("spawn")
    job_queue = ctx.Queue()
    result_queue = ctx.Queue()

    NUM_GPUS = torch.cuda.device_count()
    
    # Enqueue all jobs
    for job in jobs:
        job_queue.put(job)
    
    # Enqueue sentinel values (None) to signal workers to stop
    for _ in range(NUM_GPUS):
        job_queue.put(None)

    # Create and start worker processes
    # Note: global_config and any needed configs should be passed to worker
    procs = [ctx.Process(target=worker, args=(dev, job_queue, result_queue, global_config, bfn_config, iterator_names))
             for dev in range(NUM_GPUS)]
    for p in procs:
        p.start()

    total = len(jobs)
    
    # Collect results from workers
    with tqdm(total=total, desc="Runs", dynamic_ncols=True) as pbar:
        while done < total:
            kind, payload = result_queue.get()
            if kind == "ok":
                # Unpack job and results
                job, timekeys, train_losses, test_losses, *others = payload
                job = job[:-1] + (str(job[-1]),)
                # Store results indexed by job tuple
                et_losses[job] = test_losses
                et_timekeys[job] = timekeys
                
                # Store any extra outputs from global_config
                for kidx, k in enumerate(grab_aliases):
                    et_extras[k][job] = others[kidx]
                
                if not(global_config["ONLYTHRESHOLDS"]):
                    train_losses = train_losses[-1]
                    test_losses = test_losses[-1]
                
                job_str = " | ".join([f"{name}={val}" for name, val in zip(iterator_names, job)])
                pbar.set_postfix_str(
                    f"train {test_losses:.3g} | test {train_losses:.3g} | timekey {timekeys} | {job_str}",
                    refresh=False
                )
            else:
                job, err = payload
                print(f"[ERROR] {job}: {err}")
            done += 1
            pbar.update(1)

    # Wait for all workers to finish
    for p in procs:
        p.join()

    # Prepare and return results
    result = {
        "jobs": jobs,
        "var_axes": var_axes,
        "losses": et_losses.serialize(),
        "timekeys": et_timekeys.serialize(),
        "extras": {name: et_extras[name].serialize() for name in grab_aliases},
    }

    return result