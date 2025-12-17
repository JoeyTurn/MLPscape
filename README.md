**Requirements**

Please have Dhruva Karkada's [mupify](https://github.com/dkarkada/mupify/tree/main) and [experiment core](https://github.com/dkarkada/expt-core/tree/main) installed and in the path.

**Overview**

This repo is the result of the past few months of tinkering around with MLPs, finding I would often need to change my training loop for the specific problem, or I would need to change my outer loop to deal with the cartesian product of experiments, or change my code altogether if I was doing online vs offline learning. I have tried to create this repo to address all of the above, resulting in code where any functions that need to be evaluated within the trainloop can be specified once as what is essentially a hyperparameter. This code is designed to be able to use both *.py* files as well as *.ipynb* notebooks, with minimal changes going between the two settings! The core functionality is hidden within the `backend` folder, defining the trainloop as well as multiprocessing and command-line specifications; this can largely be ignored for most use cases.

See the `examples` folder for the typical use, which roughly follows

- Imports
- Hyperparameter specification
- Iterator specification
- Data selection
- Batch function selection
- Trainloop execution
- Results

It is highly recommended to import only 

```python
from backend.cli import parse_args, build_other_grabs
from backend.job_iterator import main as run_job_iterator
```

from the backend. 

The core trainloop is built off of batch functions. As long as a specified batch function is similar in format to the ones I have provided as examples, they will be able to work for both offline and online learning!

**Batch function py note:** If using a .py file, please place your bfn outside of any `if __name__ == "__main__":` calls so it can be found by an importer.
**Batch function ipynb note:** If using a .ipynb file, please don't initialize multiprocessing if you use a within-notebook batch function. Either have it in a separate .py file that gets imported, or don't call `mp.set_start_method("spawn", force=True)`.

To define within-trainloop function grabs, either **(1)** define the function in the file, and let make sure to update the *otherreturns* component of *global_config* (shown below); or **(2)** put your specified function within **data/mlp_grabs.py** and specify that the function should be used by with the *other_model_grabs* argument. For both cases, make sure \*\*kwargs is taken as an argument.

**Option 1 (RECOMMENDED)**
Build grabs and update it to include your function!
```python
grabs = build_other_grabs(args.other_model_grabs, per_alias_kwargs=args.other_model_kwargs,)
grabs.update({"sample_name": your_function}) #your function not in a string!
global_config.update({"otherreturns": grabs})
```

**Option 2 (Old)**
If your function uses the model, or if there is anything specific to your function such as a fixed index, use the *other_model_gram* argument ("call_with_model" for using the model, or else it defaults to grabbing weight matrices). An example shown below:
```python
args.other_model_grabs = {"sample_name": "your_function", "sample_name_2": "your_function_2"}
args.other_model_gram = {"sample_name": {"call_with_model": True, "index": 0}}
```

For the list of configurable hyperparameters and their default values, see below:

- ONLINE: True
- N_TRAIN: 4000
- N_TEST: 10_000
- ONLYTHRESHOLDS: True
- N_SAMPLES: [1024]
- NUM_TRIALS: 1
- MAX_ITER: int(1e5)
- LR: 1e-2
- DEPTH: 2
- WIDTH: 8192
- GAMMA: 1.0
- DEVICES: [0]
- SEED: 42
- LOSS_CHECKPOINTS: [0.15, 0.1]
- EMA_SMOOTHER: 0.9
- DETERMINSITIC: True
- VERBOSE: False
- datasethps: {
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
- TARGET_MONOMIALS: None
- other_model_grabs: {}
- other_model_kwargs: {}