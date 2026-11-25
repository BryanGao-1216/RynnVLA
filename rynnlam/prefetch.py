"""Bounded CPU prefetch for the offline labeler."""

from contextlib import closing
from queue import Queue
from threading import Event, Semaphore


class PrefetchedBatches:
    """Advance a batch iterator on a worker, with two image buffers in flight."""

    buffer_count = 2

    def __init__(self):
        self._queue = Queue()
        self._available = Semaphore(self.buffer_count)
        self._stopped = Event()

    def start(self, executor, batches):
        executor.submit(self._produce, batches)

    def _produce(self, batches):
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
                    self._queue.put((batch, None))
        except BaseException as exc:
            self._queue.put((None, exc))
        else:
            self._queue.put((None, None))

    def consume(self):
        while True:
            batch, error = self._queue.get()
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
