"""Wall-clock intervals for the offline labeling pipeline."""

from contextlib import contextmanager
from time import perf_counter, thread_time


@contextmanager
def timed(timings, stage, *, cpu_stage=None):
    if timings is None:
        yield
        return
    start = perf_counter()
    cpu_start = thread_time() if cpu_stage is not None else None
    try:
        yield
    finally:
        timings[stage] += perf_counter() - start
        if cpu_start is not None:
            timings[cpu_stage] += thread_time() - cpu_start


def format_timings(timings, stages):
    return " ".join(f"{stage}={timings[stage]:.3f}s" for stage in stages)
