# pipeline/audio_wav.py
import asyncio
from pathlib import Path

import numpy as np
import soundfile as sf

from pipeline.audio_source import AudioSource


class WavSource(AudioSource):
    def __init__(self, path: Path, loop: bool = True, realtime: bool = True):
        self.path = path
        self.loop = loop
        self.realtime = realtime

    async def stream(self):
        while True:
            audio, sr = sf.read(str(self.path), dtype="int16")
            if audio.ndim > 1:
                audio = audio[:, 0]  # take left channel if stereo
            if sr != self.SAMPLE_RATE:
                raise ValueError(
                    f"WAV is {sr}Hz, expected {self.SAMPLE_RATE}Hz. "
                    "Resample with ffmpeg first."
                )

            for start in range(0, len(audio), self.CHUNK_SAMPLES):
                chunk = audio[start : start + self.CHUNK_SAMPLES]
                if len(chunk) < self.CHUNK_SAMPLES:
                    # pad the tail so downstream sees uniform chunks
                    chunk = np.pad(chunk, (0, self.CHUNK_SAMPLES - len(chunk)))
                yield chunk
                if self.realtime:
                    await asyncio.sleep(self.CHUNK_SAMPLES / self.SAMPLE_RATE)

            if not self.loop:
                return