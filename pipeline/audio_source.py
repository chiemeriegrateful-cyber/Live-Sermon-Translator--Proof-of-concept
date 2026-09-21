# pipeline/audio_source.py
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
import numpy as np


class AudioSource(ABC):
    """Yields 16kHz mono int16 chunks of the same fixed size."""

    CHUNK_SAMPLES = 1600  # 100ms at 16kHz
    SAMPLE_RATE = 16000

    @abstractmethod
    async def stream(self) -> AsyncIterator[np.ndarray]:
        """Yield np.int16 arrays of shape (CHUNK_SAMPLES,)."""
        ...