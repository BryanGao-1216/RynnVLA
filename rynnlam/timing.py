"""Wall-clock intervals for the offline labeling pipeline."""

from contextlib import contextmanager
from time import perf_counter


@contextmanager
def timed(timings, stage):
    if timings is None:
        yield
        return
    start = perf_counter()
    try:
        yield
    finally:
        timings[stage] += perf_counter() - start


def format_timings(timings, stages):
    return " ".join(f"{stage}={timings[stage]:.3f}s" for stage in stages)
