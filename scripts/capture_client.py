"""Streams captured audio to the pipeline over WebSocket.

Same-machine test:
    python scripts/capture_client.py --server ws://localhost:8001/audio-in

Cross-machine (church deployment):
    python scripts/capture_client.py --server ws://<pipeline-ip>:8001/audio-in --device <line-in-index>

List devices:
    python -c "import sounddevice as sd; print(sd.query_devices())"
"""
import argparse
import asyncio
import queue
import sys

import numpy as np
import sounddevice as sd
import websockets

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1600  # must match pipeline AudioSource.CHUNK_SAMPLES


async def stream_audio(server_url: str, device):
    q: queue.Queue = queue.Queue(maxsize=50)

    def callback(indata, frames, time_info, status):
        if status:
            print(f"mic status: {status}", file=sys.stderr)
        samples = (indata[:, 0] * 32767).astype("<i2")  # little-endian int16
        try:
            q.put_nowait(samples.copy())
        except queue.Full:
            pass

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=CHUNK_SAMPLES,
        device=device,
        callback=callback,
    ):
        loop = asyncio.get_running_loop()
        while True:
            try:
                print(f"connecting to {server_url}...")
                async with websockets.connect(server_url) as ws:
                    print("connected, streaming")
                    while True:
                        chunk = await loop.run_in_executor(None, q.get)
                        await ws.send(chunk.tobytes())
            except (websockets.exceptions.WebSocketException, OSError) as e:
                print(f"connection lost: {e}. reconnecting in 2s...")
                await asyncio.sleep(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True, help="ws://host:port/audio-in")
    parser.add_argument("--device", type=int, default=None, help="input device index")
    args = parser.parse_args()

    try:
        asyncio.run(stream_audio(args.server, args.device))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()