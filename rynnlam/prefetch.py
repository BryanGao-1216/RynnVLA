"""Bounded CPU prefetch for the offline labeler."""

from contextlib import closing
from queue import Queue
from threading import Event, Semaphore

from .timing import timed


class PrefetchedBatches:
    """Advance a batch iterator on a worker, with two image buffers in flight."""

    buffer_count = 2

    def __init__(self):
        self._queue = Queue()
        self._available = Semaphore(self.buffer_count)
        self._stopped = Event()

    def start(self, executor, batches, producer_timings):
        executor.submit(self._produce, batches, producer_timings)

    def _produce(self, batches, timings):
        try:
            with closing(batches):
                while True:
                    # Take a slot before advancing the generator. Advancing can write into a
                    # previously yielded image buffer, so dequeuing it alone cannot free a slot.
                    self._available.acquire()
                    if self._stopped.is_set():
                        return
                    batch = next(batches, None)
                    if batch is None:
                        break
                    self._queue.put((batch, timings.copy(), None))
                    timings.clear()
        except BaseException as exc:
            self._queue.put((None, timings.copy(), exc))
        else:
            self._queue.put((None, timings.copy(), None))

    def consume(self, timings):
        while True:
            with timed(timings, "prefetch_wait"):
                batch, deltas, error = self._queue.get()
            if timings is not None:
                timings.update(deltas)
            if error is not None:
                raise error
            if batch is None:
                return
            try:
                yield batch
            finally:
                self._available.release()

    def close(self):
        if not self._stopped.is_set():
            self._stopped.set()
            self._available.release()
