# pipeline/audio_mic.py
import asyncio
import queue

import numpy as np
import sounddevice as sd

from pipeline.audio_source import AudioSource


class MicSource(AudioSource):
    def __init__(self, device: int | str | None = None):
        self.device = device  # None = system default

    async def stream(self):
        # sounddevice callback runs on a C thread; use a threadsafe queue
        # to hand chunks to the async loop.
        q: queue.Queue = queue.Queue(maxsize=50)

        def callback(indata, frames, time_info, status):
            if status:
                print(f"mic status: {status}")
            # indata is float32 [-1, 1] by default; convert to int16
            samples = (indata[:, 0] * 32767).astype(np.int16)
            try:
                q.put_nowait(samples.copy())
            except queue.Full:
                pass  # drop when overwhelmed rather than block audio callback

        with sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=self.CHUNK_SAMPLES,
            device=self.device,
            callback=callback,
        ):
            loop = asyncio.get_running_loop()
            while True:
                # Pop from the sync queue on the async side without blocking
                # the event loop.
                chunk = await loop.run_in_executor(None, q.get)
                yield chunk