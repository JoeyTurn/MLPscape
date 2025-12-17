import numpy as np
import torch

import torch.multiprocessing as mp
import sys
sys.path.append("/home/user/MLP/")
from backend.cli import parse_args, build_other_grabs

from data.monomial import Monomial
from backend.job_iterator import main as run_job_iterator
from backend.utils import ensure_torch, load_json

from data.ntk_coeffs import get_relu_level_coeff_fn

import os, sys
from FileManager import FileManager

from backend.utils import ensure_torch
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

if __name__ == "__main__":

    args = parse_args() #default args

    if args.TARGET_MONOMIALS is not None:
        args.TARGET_MONOMIALS = [Monomial(m) for m in args.TARGET_MONOMIALS]
    elif args.target_monomials_path:
        target_monomials_json = load_json(args.target_monomials_path)
        args.TARGET_MONOMIALS = [Monomial(m) for m in target_monomials_json]
    elif args.TARGET_MONOMIALS is None:
        args.TARGET_MONOMIALS = [Monomial({10: 1}), Monomial({190:1}), Monomial({0:2}), Monomial({2:1, 3:1}), Monomial({15:1, 20:1}), Monomial({0:3}),]
        
    if args.datasethps_path:
        args.datasethps = load_json(args.datasethps_path)

    # Set any args that we want to differ
    args.ONLINE = True
    args.N_TRAIN=4000
    args.N_TEST=1000
    args.N_TOT = args.N_TEST+args.N_TRAIN
    args.TARGET_MONOMIALS = [Monomial({0:1}), Monomial({1:1})]
    args.NUM_TRIALS = 2
    args.N_SAMPLES = [1024]

    iterators = [args.N_SAMPLES, range(args.NUM_TRIALS), args.TARGET_MONOMIALS]
    iterator_names = ["ntrain", "trial", "monomials"]
    
    datapath = os.getenv("DATASETPATH") #datapath = os.path.join(os.getenv(...))
    exptpath = os.getenv("EXPTPATH") #same here
    if datapath is None:
        raise ValueError("must set $DATASETPATH environment variable")
    if exptpath is None:
        raise ValueError("must set $EXPTPATH environment variable")
    expt_name = "example_mlp_run"
    dataset = "synthetic"
    expt_dir = os.path.join(exptpath, "example_folder", expt_name, dataset)

    if not os.path.exists(expt_dir):
        os.makedirs(expt_dir)
    expt_fm = FileManager(expt_dir)
    print(f"Working in directory {expt_dir}.")


    from data.data import get_synthetic_X
    from data.monomial import generate_hea_monomials

    X_full, data_eigvals = get_synthetic_X(**args.datasethps, N=args.N_TOT, gen=torch.Generator(device='cuda').manual_seed(args.SEED))
    hea_eigvals, monomials = generate_hea_monomials(data_eigvals, num_monomials=args.datasethps['cutoff_mode'], **args.datasethps,
                                                    eval_level_coeff=get_relu_level_coeff_fn(data_eigvals=data_eigvals, weight_variance=1, bias_variance=1),)
    hea_eigvals = ensure_torch(hea_eigvals)

    U, lambdas, Vt = torch.linalg.svd(X_full, full_matrices=False)
    dim = X_full.shape[1]

    ## --- Target function defs ---
    bfn_config = dict(lambdas=lambdas, Vt=Vt, data_eigvals=data_eigvals, N=args.N_TOT, base_bfn=polynomial_batch_fn)
    

    global_config = dict(DEPTH=args.DEPTH, WIDTH=args.WIDTH, LR=args.LR, GAMMA=args.GAMMA,
        EMA_SMOOTHER=args.EMA_SMOOTHER, MAX_ITER=args.MAX_ITER,
        LOSS_CHECKPOINTS=args.LOSS_CHECKPOINTS, N_TEST=args.N_TEST,
        SEED=args.SEED, ONLYTHRESHOLDS=args.ONLYTHRESHOLDS, DIM=dim,
        ONLINE=args.ONLINE, VERBOSE=args.VERBOSE
        )

    grabs = build_other_grabs(args.other_model_grabs, default_source=args.W_source, concat_outside=args.concat_outside,
        per_alias_gram=args.other_model_gram, per_alias_kwargs=args.other_model_kwargs,)
    global_config.update({"otherreturns": grabs})
    
    mp.set_start_method("spawn", force=True)
    
    result = run_job_iterator(iterators, iterator_names, global_config, bfn_config=bfn_config)
    print(f"Results saved to {expt_dir}")
    expt_fm.save(result, "result.pickle")
    torch.cuda.empty_cache()