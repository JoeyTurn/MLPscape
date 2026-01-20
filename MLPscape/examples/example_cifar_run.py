import numpy as np
import torch

import torch.multiprocessing as mp
from MLPscape.backend.cli import parse_args
from MLPscape.backend.job_iterator import main as run_job_iterator
from MLPscape.backend.utils import ensure_torch

import os, sys
from FileManager import FileManager

def general_batch_fn(X_total, y_total, X=None, y=None, bsz=128,
                     gen=None, **kwargs):
    def batch_fn(step: int, X=X, y=y):
        if (X is not None) and (y is not None):
            X = ensure_torch(X)
            y = ensure_torch(y)
            return X, y
        with torch.no_grad():
            N_total = X_total.shape[0]
            indices = torch.randint(0, N_total, (bsz,), generator=gen, device=gen.device)
            X_batch = ensure_torch(X_total.to(gen.device)[indices])
            y_batch = ensure_torch(y_total.to(gen.device)[indices])
            return X_batch, y_batch
    return batch_fn

if __name__ == "__main__":

    args = parse_args() #default args

    # Set any args that we want to differ
    args.ONLINE = False
    args.N_TRAIN=4000
    args.N_TEST=1000
    args.N_TOT = args.N_TEST+args.N_TRAIN
    args.CLASSES = [[0], [6]]
    args.NORMALIZED = True
    args.NUM_TRIALS = 2
    args.N_SAMPLES = [1024]
    args.GAMMA = [0.1, 1, 10]
    args.MAX_ITER = 5e2
    args.LR = 2e-1
    args.LOSS_CHECKPOINTS = 0.1

    iterators = [args.N_SAMPLES, range(args.NUM_TRIALS), args.GAMMA]
    iterator_names = ["ntrain", "trial", "GAMMA"]
    
    datapath = os.getenv("DATASETPATH") #datapath = os.path.join(os.getenv(...))
    exptpath = os.getenv("EXPTPATH") #same here
    if datapath is None:
        raise ValueError("must set $DATASETPATH environment variable")
    if exptpath is None:
        raise ValueError("must set $EXPTPATH environment variable")
    expt_dir = os.path.join(exptpath, "example_folder", "example_mlp_run", "synthetic")

    if not os.path.exists(expt_dir):
        os.makedirs(expt_dir)
    expt_fm = FileManager(expt_dir)
    print(f"Working in directory {expt_dir}.")


    from ImageData import ImageData, preprocess
    PIXEL_NORMALIZED =  False # Don't normalize pixels, normalize samples
    classes = args.CLASSES
    normalized = args.NORMALIZED

    if classes is not None:
        imdata = ImageData('cifar10', "../data", classes=classes, onehot=len(classes)!=2)
    else:
        imdata = ImageData('cifar10', "../data", classes=classes, onehot=False)
    X_train, y_train = imdata.get_dataset(args.N_TRAIN, get='train')
    X_train = preprocess(X_train, center=True, greyscale=True, normalize=PIXEL_NORMALIZED)
    X_test, y_test = imdata.get_dataset(args.N_TEST, get='test')
    X_test = preprocess(X_test, center=True, greyscale=True, normalize=PIXEL_NORMALIZED)
    X_train, y_train, X_test, y_test = map(ensure_torch, (X_train, y_train, X_test, y_test))
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()
    X_train, y_train, X_test, y_test = [t/torch.linalg.norm(t) for t in (X_train, y_train, X_test, y_test)] if normalized else (X_train, y_train, X_test, y_test)
    if normalized:
        X_train *= args.N_TRAIN**(0.5); X_test *= args.N_TEST**(0.5)
        y_train *= args.N_TRAIN**(0.5); y_test *= args.N_TEST**(0.5)
    X_full = torch.cat((X_train, X_test), dim=0)
    y_full = torch.cat((y_train, y_test), dim=0)
    data_eigvals = torch.linalg.svdvals(X_full)**2
    data_eigvals /= data_eigvals.sum()

    U, lambdas, Vt = torch.linalg.svd(X_full, full_matrices=False)
    dim = X_full.shape[1]

    bfn_config = dict(X_total = X_full, y_total = y_full, base_bfn=general_batch_fn)
    del X_full, y_full   

    global_config = dict(depth=args.depth, width=args.width, LR=args.LR, GAMMA=args.GAMMA,
        EMA_SMOOTHER=args.EMA_SMOOTHER, MAX_ITER=args.MAX_ITER,
        LOSS_CHECKPOINTS=args.LOSS_CHECKPOINTS, N_TEST=args.N_TEST,
        SEED=args.SEED, ONLYTHRESHOLDS=args.ONLYTHRESHOLDS, DIM=dim,
        ONLINE=args.ONLINE, VERBOSE=args.VERBOSE
        )

    grabs = {}
    global_config.update({"otherreturns": grabs})
    
    mp.set_start_method("spawn", force=True)
    
    result = run_job_iterator(iterators, iterator_names, global_config, bfn_config=bfn_config)
    print(f"Results saved to {expt_dir}")
    expt_fm.save(result, "result.pickle")
    torch.cuda.empty_cache()