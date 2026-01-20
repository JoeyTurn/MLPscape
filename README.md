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
from backend.cli import parse_args (.py files) OR base_args (.ipynb files)
from backend.job_iterator import main as run_job_iterator
```

from the backend. 

The core trainloop is built off of batch functions. As long as a specified batch function is similar in format to the ones I have provided as examples (see `MLPscape/examples/notebook_example_cifar.ipynb`), they will be able to work for both offline and online learning!

**Batch function .py note:** If using a .py file, please place your bfn outside of any `if __name__ == "__main__":` calls so it can be found by an importer.
**Batch function .ipynb note:** If using a .ipynb file, please don't initialize multiprocessing if you use a within-notebook batch function. Either have it in a separate .py file that gets imported, or don't call `mp.set_start_method("spawn", force=True)`.

To define within-trainloop function grabs, define the function in the file, and make sure to update the *otherreturns* component of *global_config* (shown below). Make sure \*\*kwargs is taken as an argument to your function!

```python
def your_function(stuff, **kwargs):
    return stuff**2

def your_2nd_fn(stuff, **kwargs):
    return stuff**0.5

grabs = {"sample_name": your_function, "sample_fn_2": your_2nd_fn}
global_config.update({"otherreturns": grabs})
```

For the list of configurable (pre-set) hyperparameters and their default values, see below:

- ONLINE: True - Sets if training is done with one set batch or with a variable training batch
- N_SAMPLES: 1024 - Number of samples that get used throughout training (the batch size if ONLINE is True)
- N_TRIAN: 4000 - Along with N_TEST, is used only to define the total number of samples that exist for your data
- N_TEST: 10_000 - Testset size (testset is defined once and is based off the batch function)
                   Note: if you want a specific testset, set it through global_config["X_te"] and global_config["y_te"].
- ONLYTHRESHOLDS: True - If True, the results will only be from the trained network instead of the full run
- NUM_TRIALS: 1 - Number of trials
- MAX_ITER: int(1e5) - Maximum number of gradient steps the network will take
- LR: 1e-2 - The base learning rate (to be rescaled in the muP setting)
- DEPTH: 2 - The number of hidden layers of the network (must be > 0)
- WIDTH: 8192 - The width of all hidden layers
- GAMMA: 1.0 - A rescaling factor: outputs->outputs/gamma, `lr->lr*gamma**(2.0) (gamma<1) or lr->lr*gamma (gamma>1)`
- DEVICES: [0] - The device IDs for any used GPUs
- SEED: 42 - Base set seed of all RNG forms; on different devices, the seed is (SEED+DEVICE_ID)
- LOSS_CHECKPOINTS: [0.15, 0.1] - The thresholds at which the trainloop will exit
- EMA_SMOOTHER: 0.9 - Exponential moving average constant of loss values
- DETERMINSITIC: True - If True, the set seed will be used; False turns this off
- VERBOSE: False - If True, will display the loss at each timestep