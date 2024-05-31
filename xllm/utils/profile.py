import time
from contextlib import contextmanager

import torch
from loguru import logger

execution_times = {}


@contextmanager
def log_exec_time(key):
    start_time = time.time()
    yield
    end_time = time.time()
    execution_time = end_time - start_time
    if key in execution_times:
        execution_times[key].append(execution_time)
    else:
        execution_times[key] = [execution_time]

    COUNT = 10
    if len(execution_times[key]) == COUNT:
        average_time = sum(execution_times[key]) / COUNT
        logger.info(f"Average execution time for {key=} is {average_time} seconds")
        execution_times[key] = []


def nvtx_range(func):
    """decorator that causes an NVTX range to be recorded for the duration of the
    function call."""

    def wrapped_fn(*args, **kwargs):
        torch.cuda.nvtx.range_push(func.__qualname__)
        ret_val = func(*args, **kwargs)
        torch.cuda.nvtx.range_pop()
        return ret_val

    return wrapped_fn
