"""Receives audio chunks over WebSocket from an external capture client.

Deployment target: laptop at the church mixing board captures line-in audio
and streams to the pipeline over LAN. Also works same-machine over localhost
for testing.

The capture client lives in scripts/capture_client.py.
"""
import asyncio

import numpy as np

from pipeline.audio_source import AudioSource


class NetworkSource(AudioSource):
    def __init__(self):
        # Bounded queue: drop old chunks if the pipeline falls behind, rather
        # than growing memory. Audio timeliness beats completeness.
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._active_client: str | None = None

    def push(self, chunk: np.ndarray) -> None:
        """Called by the /audio-in endpoint on each received chunk."""
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            self._queue.put_nowait(chunk)
        except asyncio.QueueFull:
            pass

    def client_connected(self, client_id: str) -> None:
        self._active_client = client_id
        print(f"audio-in: capture client connected ({client_id})")

    def client_disconnected(self, client_id: str) -> None:
        if self._active_client == client_id:
            self._active_client = None
        print(f"audio-in: capture client disconnected ({client_id})")

    async def stream(self):
        while True:
            chunk = await self._queue.get()
            yield chunk