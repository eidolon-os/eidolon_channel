"""One prewarmed, bounded inference executor owned by the Channel Provider.

No profiles, devices, Companion policy or recording persistence live here.
Cancellation of a caller never pretends to have stopped native inference.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import math


class ModelUnavailable(RuntimeError):
    pass


class InferenceBusy(RuntimeError):
    pass


class InferenceTimeout(RuntimeError):
    pass


class SpeakerInferenceEngine:
    def __init__(self, extractor, *, max_pending=2, timeout_seconds=3.0):
        if not 1 <= max_pending <= 4 or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError('bounded capacity and positive timeout required')
        self._extractor = extractor
        self._capacity = max_pending
        self._timeout = timeout_seconds
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='speaker-inference')
        self._pending = set()
        self._warm = None
        self.state = 'not_started'

    @property
    def model_identity(self):
        return self._extractor.model_identity if self.state == 'ready' else None

    def _submit(self, operation, *args):
        # Already-completed futures may still have their callbacks queued.
        self._pending = {future for future in self._pending if not future.done()}
        if len(self._pending) >= self._capacity:
            raise InferenceBusy('inference_busy')
        future = asyncio.wrap_future(self._executor.submit(operation, *args))
        self._pending.add(future)

        def complete(done):
            self._pending.discard(done)
            if not done.cancelled():
                done.exception()  # Consume errors even after caller cancellation/deadline.

        future.add_done_callback(complete)
        return future

    async def start(self):
        if self.state == 'closed':
            raise ModelUnavailable('model_closed')
        if self._warm is None:
            self.state = 'loading'
            self._warm = self._submit(self._extractor.warm_up)

            def loaded(done):
                if self.state != 'closed':
                    self.state = 'unavailable' if done.cancelled() or done.exception() else 'ready'

            self._warm.add_done_callback(loaded)
        try:
            await asyncio.shield(self._warm)
        except Exception:
            if self.state != 'closed':
                self.state = 'unavailable'
            raise ModelUnavailable('model_load_failed') from None
        if self.state != 'closed':
            self.state = 'ready'

    async def extract(self, pcm):
        if self.state != 'ready':
            raise ModelUnavailable('model_' + self.state)
        future = self._submit(self._extractor.extract, pcm)
        try:
            async with asyncio.timeout(self._timeout):
                return await asyncio.shield(future)
        except TimeoutError:
            raise InferenceTimeout('inference_timeout') from None

    async def close(self):
        if self.state == 'closed':
            return
        self.state = 'closed'
        await asyncio.to_thread(self._executor.shutdown, wait=True, cancel_futures=True)
        if self._pending:
            await asyncio.gather(*self._pending, return_exceptions=True)
