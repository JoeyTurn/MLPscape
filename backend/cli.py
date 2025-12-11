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


# assumes str2bool(v) is already defined in your module

class GrabRunner:
    def __init__(self, base_fn_name, source="in", concat_outside=True, base_kwargs=None, *, input_mode="gram",
                 call_with_model=False):
        self.base_fn_name    = base_fn_name
        self.source          = source
        self.concat_outside  = concat_outside
        self.base_kwargs     = dict(base_kwargs or {})
        self.input_mode      = input_mode
        self.call_with_model = call_with_model
        self._printed_shape  = True # optional: one-time info print

    @staticmethod
    def _shape_summary(x):
        is_tensor = isinstance(x, torch.Tensor)
        if is_tensor:
            return tuple(x.shape)
        if isinstance(x, (list, tuple)):
            out = []
            for xi in x:
                if torch is not None and isinstance(xi, torch.Tensor):
                    out.append(tuple(xi.shape))
                else:
                    out.append(type(xi).__name__)
            return out
        return type(x).__name__

    def __call__(self, model, *_, **kwargs):
        import data.mlp_grabs as m

        fn     = getattr(m, self.base_fn_name)
        merged = {**self.base_kwargs, **kwargs}

        if self.call_with_model:
            if not self._printed_shape:
                print(f"[{self.base_fn_name}] calling directly on model with kwargs: {list(merged.keys())}")
                self._printed_shape = True
            return fn(model, **merged)

        # --- old path: we operate on W / Gram
        if "use_raw" in merged:
            try:
                merged["use_raw"] = str2bool(merged["use_raw"])
            except Exception:
                pass

        W, b = m.get_Win(model) if self.source == "in" else m.get_Wout(model)

        use_raw = merged.pop("use_raw", None)
        if (use_raw is True) or (self.input_mode == "raw"):
            X = W
        else:
            X = m.get_W_gram(W, concatenate_outside=self.concat_outside)

        if not self._printed_shape:
            print(f"[{self.base_fn_name}] grabbing object with shape(s): {self._shape_summary(X)}")
            self._printed_shape = True

        return fn(X, **merged)


def build_other_grabs(spec, *, default_source="in", concat_outside=True, per_alias_gram=None, per_alias_kwargs=None,):
    """
    spec: {"Wii1":"get_W_ii", "Tmb":"T_mixed_bias", ...}
    per_alias_gram:
        {
            "Wii1": {"source":"out","concat_outside":"True"},
        }
    per_alias_kwargs:
        {
            "Wii1": {"i":1},
        }
    """
    per_alias_gram   = per_alias_gram   or {}
    per_alias_kwargs = per_alias_kwargs or {}
    out = {}

    # normalize default concat_outside (could be string)
    try:
        concat_outside = str2bool(concat_outside)
    except Exception:
        pass

    for alias, fn_name in spec.items():
        cfg = {
            "source": default_source,
            "concat_outside": concat_outside,
            "input_mode": "gram",
            "call_with_model": False,  # default
        }
        cfg.update(per_alias_gram.get(alias, {}))

        # normalize booleans
        if "concat_outside" in cfg:
            try:
                cfg["concat_outside"] = str2bool(cfg["concat_outside"])
            except Exception:
                pass

        # legacy: use_raw -> input_mode
        if "use_raw" in cfg:
            try:
                if str2bool(cfg["use_raw"]):
                    cfg["input_mode"] = "raw"
            except Exception:
                pass
            del cfg["use_raw"]

        # call_with_model might also come in as "true"/"false"
        if "call_with_model" in cfg:
            try:
                cfg["call_with_model"] = str2bool(cfg["call_with_model"])
            except Exception:
                pass

        kw = dict(per_alias_kwargs.get(alias, {}))

        out[alias] = GrabRunner(
            base_fn_name=fn_name,
            source=cfg["source"],
            concat_outside=cfg["concat_outside"],
            base_kwargs=kw,
            input_mode=cfg["input_mode"],
            call_with_model=cfg["call_with_model"],
        )

    return out

def set_data_eigvals_for_Tmb(grabs, data_eigvals, other_model_gram=None, key="Tmb"):
    for name, runner in grabs.items():
        is_tmb = (isinstance(name, str) and key in name) \
                 or getattr(runner, "base_fn_name", None) == "T_mixed_bias"
        if not is_tmb:
            continue

        runner.call_with_model = True
        runner.base_kwargs = dict(getattr(runner, "base_kwargs", {}))
        runner.base_kwargs["data_eigvals"] = data_eigvals

        if isinstance(other_model_gram, dict):
            cfg = other_model_gram.get(name)
            if isinstance(cfg, dict):
                runner.base_kwargs.update(cfg)


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
    p.add_argument("--TARGET_FUNCTION_TYPE", type=str, default="monomial", help="Type of target function to learn.")
    p.add_argument("--TARGET_MONOMIALS", type=json.loads, default=None, help="List of target monomials as JSON string.")
    p.add_argument("--ONLYTHRESHOLDS", type=str2bool, default=True, help="If True, only record last loss instead of full curve.")
    p.add_argument("--N_SAMPLES", nargs="+", type=int, default=[1024], help="Number of samples.")
    p.add_argument("--NUM_TRIALS", type=int_from_any, default=1, help="Number of independent trials.")
    
    p.add_argument("--MAX_ITER", type=int_from_any, default=1e5, help="Steps per trial.")
    p.add_argument("--LR", type=float, default=1e-2, help="Learning rate.")
    p.add_argument("--DEPTH", type=int_from_any, default=2, help="Number of hidden layers+1.")
    p.add_argument("--WIDTH", type=int_from_any, default=8192, help="Width of hidden layers.")
    p.add_argument("--GAMMA", type=float, default=1.0, help="Richness parameter for training.")
    p.add_argument("--DEVICES", type=int, nargs="+", default=[0], help="GPU ids, e.g. --DEVICES 2 4")
    
    p.add_argument("--SEED", type=int, default=42, help="RNG seed.")
    p.add_argument("--LOSS_CHECKPOINTS", type=float, nargs="+", default=[0.15, 0.1], help="Loss checkpoints to record.")
    p.add_argument("--EMA_SMOOTHER", type=float, default=0.9, help="EMA smoother for loss tracking.")
    p.add_argument("--DETERMINISTIC", type=str2bool, default=True, help="Whether to use deterministic training.")
    p.add_argument("--VERBOSE", type=str2bool, default=False, help="Whether to print out training info.")
    
    p.add_argument("--EXPT_NAME", type=str, default="mlp-learning-curves", help="Where to save results.")

    # p.add_argument("--DATASETPATH", type=str, default=str(Path.home() / "data"), help="Path to dataset root.")
    p.add_argument("--datasethps", type=json.loads,
                    default='{"normalized": true, "cutoff_mode": 40000, "d": 200, "offset": 6, "alpha": 2.0, "noise_size": 1, "yoffset": 1.2, "beta": 1.2, "classes": null, "binarize": false, "weight_variance": 1, "bias_variance": 1}',
                    help="Dataset hyperparameters as JSON string.")
    p.add_argument("--datasethps_path", help="Path to datasethps .json")
    p.add_argument("--target_monomials_path", help="Path to target monomials .json")
    
    p.add_argument("--other_model_grabs", help="Dict of {name, fn} pairs for grabbing model parameters", type=json.loads, default={})
    p.add_argument("--W_source",  choices=["in","out"], default="in", help="Default base matrix: Win or Wout")
    p.add_argument("--concat_outside", type=str2bool, default="True", help="Choice of having the gram matrix be W^T W or W W^T")
    p.add_argument("--other_model_gram", type=json.loads, default={}, help='Per-alias overrides, e.g. {"Wii1":{"source":"out","concat_outside":"True"}}')
    p.add_argument("--other_model_kwargs", type=json.loads, default={}, help='Per-alias kwargs, e.g. {"Wii1":{"i":1}}')
    return p.parse_args()

def base_args():
    return argparse.Namespace(**{
    "ONLINE": True,
    "N_TRAIN": 4000,
    "N_TEST": 10_000,
    "TARGET_MONOMIALS": None,
    "ONLYTHRESHOLDS": True,
    "N_SAMPLES": [1024],
    "NUM_TRIALS": 1,
    "MAX_ITER": int(1e5),
    "LR": 1e-2,
    "DEPTH": 2,
    "WIDTH": 8192,
    "GAMMA": 1.0,
    "DEVICES": [0],
    "SEED": 42,
    "LOSS_CHECKPOINTS": [0.15, 0.1],
    "EMA_SMOOTHER": 0.9,
    "DETERMINSITIC": True,
    "VERBOSE": False,
    "datasethps": {
        "normalized": True,
        "cutoff_mode": 40_000,
        "d": 200,
        "offset": 6,
        "alpha": 2.0,
        "noise_size": 1,
        "yoffset": 1.2,
        "beta": 1.2,
        "classes": None,
        "binarize": False,
        "weight_variance": 1,
        "bias_variance": 1,
    },
    "other_model_grabs": {},
    "W_source": "in",
    "concat_outside": True,
    "other_model_gram": {},
    "other_model_kwargs": {},
})