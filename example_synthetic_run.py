import numpy as np
import torch

import torch.multiprocessing as mp
from backend.cli import parse_args, build_other_grabs

from data.monomial import Monomial
from data.data import polynomial_batch_fn
from backend.job_iterator import main as run_job_iterator
from backend.utils import ensure_torch, load_json

from ntk_coeffs import get_relu_level_coeff_fn

import os, sys
from FileManager import FileManager


if __name__ == "__main__":

    args = parse_args() #default args

    # Set any args that we want to differ
    args.ONLINE = True
    args.N_TRAIN=4000
    args.N_TEST=1000
    args.NUM_TRIALS = 3
    args.N_TOT = args.N_TEST+args.N_TRAIN
    
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

    
    if args.TARGET_MONOMIALS is not None:
        args.TARGET_MONOMIALS = [Monomial(m) for m in args.TARGET_MONOMIALS]
    elif args.target_monomials_path:
        target_monomials_json = load_json(args.target_monomials_path)
        args.TARGET_MONOMIALS = [Monomial(m) for m in target_monomials_json]
    elif args.TARGET_MONOMIALS is None:
        args.TARGET_MONOMIALS = [Monomial({10: 1}), Monomial({190:1}), Monomial({0:2}), Monomial({2:1, 3:1}), Monomial({15:1, 20:1}), Monomial({0:3}),]
        
    if args.datasethps_path:
        args.datasethps = load_json(args.datasethps_path)

    from data.data import get_synthetic_X
    from data.monomial import generate_hea_monomials

    X_full, data_eigvals = get_synthetic_X(**args.datasethps, N=args.N_TOT, gen=torch.Generator(device='cuda').manual_seed(args.SEED))
    hea_eigvals, monomials = generate_hea_monomials(data_eigvals, num_monomials=args.datasethps['cutoff_mode'], **args.datasethps,
                                                    eval_level_coeff=get_relu_level_coeff_fn(data_eigvals=data_eigvals, weight_variance=1, bias_variance=1),)
    hea_eigvals = ensure_torch(hea_eigvals)

    U, lambdas, Vt = torch.linalg.svd(X_full, full_matrices=False)
    dim = X_full.shape[1]

    ## --- Target function defs ---
    if args.TARGET_FUNCTION_TYPE == "monomial":
        target_monomials = args.TARGET_MONOMIALS
        targets = target_monomials
        bfn_config = dict(lambdas=lambdas, Vt=Vt, data_eigvals=data_eigvals, N=args.N_TOT, bfn_name="polynomial_batch_fn")
    

    global_config = dict(DEPTH=args.DEPTH, WIDTH=args.WIDTH, LR=args.LR, GAMMA=args.GAMMA,
        EMA_SMOOTHER=args.EMA_SMOOTHER, MAX_ITER=args.MAX_ITER,
        LOSS_CHECKPOINTS=args.LOSS_CHECKPOINTS, N_TEST=args.N_TEST,
        SEED=args.SEED, ONLYTHRESHOLDS=args.ONLYTHRESHOLDS, DIM=dim,
        TARGET_FUNCTION_TYPE=args.TARGET_FUNCTION_TYPE,
        ONLINE=args.ONLINE, VERBOSE=args.VERBOSE
        )

    grabs = build_other_grabs(args.other_model_grabs, default_source=args.W_source, concat_outside=args.concat_outside,
        per_alias_gram=args.other_model_gram, per_alias_kwargs=args.other_model_kwargs,)
    global_config.update({"otherreturns": grabs})
    
    mp.set_start_method("spawn", force=True)
    sample_sizes = [100, 500, 1000]
    trials = [0, 1, 2]
    
    iterators = [args.N_SAMPLES, args.NUM_TRIALS, args.TARGET_MONOMIALS]
    iterator_names = ["ntrain", "trial", "target_monomial"]
    

    result = run_job_iterator(iterators, iterator_names, global_config, bfn_config=bfn_config)
    expt_fm.save(result, "result.pickle")
    torch.cuda.empty_cache()