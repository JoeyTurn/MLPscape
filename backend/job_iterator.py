import numpy as np
import torch
from torch.multiprocessing import get_context
import torch.multiprocessing as mp
from tqdm import tqdm
from itertools import product

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from .utils import ensure_torch

from ExptTrace import ExptTrace

from .worker import worker


## --- Multiprocessing execution ---
def main(iterators, iterator_names=None, global_config=None, bfn=None):
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
    grab_aliases = list(global_config.other_model_grabs.keys()) if global_config and hasattr(global_config, 'other_model_grabs') else []
    et_extras = {alias: ExptTrace(var_axes) for alias in grab_aliases}

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
    procs = [ctx.Process(target=worker, args=(dev, job_queue, result_queue, global_config, iterator_names, bfn))
             for dev in range(NUM_GPUS)]
    for p in procs:
        p.start()

    total = len(jobs)
    done = 0
    
    # Collect results from workers
    with tqdm(total=total, desc="Runs", dynamic_ncols=True) as pbar:
        while done < total:
            kind, payload = result_queue.get()
            if kind == "ok":
                # Unpack job and results
                job, timekeys, train_losses, test_losses, *others = payload
                
                # Store results indexed by job tuple
                et_losses[job] = test_losses
                et_timekeys[job] = timekeys
                
                # Store any extra outputs from global_config
                for kidx, k in enumerate(grab_aliases):
                    et_extras[k][job] = others[kidx]
                
                if not(global_config.ONLYTHRESHOLDS):
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