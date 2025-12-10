import sys
import torch
sys.path.append("../")
from backend.utils import tuple_to_numpy
from backend.job import run_job

def worker(device_id, job_queue, result_queue, global_config, bfn):
    try: torch.cuda.set_device(device_id)
    except Exception as e:
        result_queue.put(("bootstrap_err", repr(e)))
        return

    while True:
        job = job_queue.get()
        if job is None:
            break
        try:
            payload = run_job(device_id, job, global_config, bfn)
            payload = tuple_to_numpy(payload)
            
            result_queue.put(("ok", payload))
        except Exception as e:
            result_queue.put(("err", (job, repr(e))))