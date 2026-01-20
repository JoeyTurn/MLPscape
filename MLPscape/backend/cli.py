import json

import argparse
from pathlib import Path

import torch

def int_from_any(x: str) -> int:
    """Accept '10000', '1e4', '5.0' → int."""
    try:
        v = int(float(x))
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid int: {x}") from e
    return v


def str2bool(v):
    if isinstance(v, bool): return v
    v = v.lower()
    if v in ("yes","true","t","y","1"): return True
    if v in ("no","false","f","n","0"): return False
    raise argparse.ArgumentTypeError("expected a boolean")


def load_json(path: str):
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {p}")
    return json.loads(p.read_text())

def parse_args():
    p = argparse.ArgumentParser(description="Config for MLP training")
    p.add_argument("--ONLINE", type=bool, default=True, help="Whether to use online training (full dataset at once) or fixed dataset.")
    p.add_argument("--N_TRAIN", type=int_from_any, default=4000, help="Number of training samples.")
    p.add_argument("--N_TEST", type=int_from_any, default=10_000, help="Number of test samples.")
    p.add_argument("--DATASET", type=str, default="synthetic", help="Dataset to use, currently: synthetic (gaussian) or cifar10.")
    p.add_argument("--ONLYTHRESHOLDS", type=str2bool, default=True, help="If True, only record last loss instead of full curve.")
    p.add_argument("--N_SAMPLES", nargs="+", type=int, default=[1024], help="Number of samples.")
    p.add_argument("--NUM_TRIALS", type=int_from_any, default=1, help="Number of independent trials.")
    
    p.add_argument("--MAX_ITER", type=int_from_any, default=1e5, help="Steps per trial.")
    p.add_argument("--LR", type=float, default=1e-2, help="Learning rate.")
    p.add_argument("--depth", type=int_from_any, default=2, help="Number of hidden layers+1.")
    p.add_argument("--width", type=int_from_any, default=8192, help="Width of hidden layers.")
    p.add_argument("--GAMMA", type=float, default=1.0, help="Richness parameter for training.")
    p.add_argument("--DEVICES", type=int, nargs="+", default=[0], help="GPU ids, e.g. --DEVICES 2 4")
    
    p.add_argument("--SEED", type=int, default=42, help="RNG seed.")
    p.add_argument("--LOSS_CHECKPOINTS", type=float, nargs="+", default=[0.15, 0.1], help="Loss checkpoints to record.")
    p.add_argument("--EMA_SMOOTHER", type=float, default=0.9, help="EMA smoother for loss tracking.")
    p.add_argument("--DETERMINISTIC", type=str2bool, default=True, help="Whether to use deterministic training.")
    p.add_argument("--VERBOSE", type=str2bool, default=False, help="Whether to print out training info.")
    
    p.add_argument("--EXPT_NAME", type=str, default="mlp-learning-curves", help="Where to save results.")
    return p.parse_args()

def base_args():
    return argparse.Namespace(**{
    "ONLINE": True,
    "N_TRAIN": 4000,
    "N_TEST": 10_000,
    "ONLYTHRESHOLDS": True,
    "N_SAMPLES": [1024],
    "NUM_TRIALS": 1,
    "MAX_ITER": int(1e5),
    "LR": 1e-2,
    "depth": 2,
    "width": 8192,
    "GAMMA": 1.0,
    "DEVICES": [0],
    "SEED": 42,
    "LOSS_CHECKPOINTS": [0.15, 0.1],
    "EMA_SMOOTHER": 0.9,
    "DETERMINSITIC": True,
    "VERBOSE": False,
})