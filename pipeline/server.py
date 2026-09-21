import asyncio
import multiprocessing
import os
import sys
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from pipeline.audio_pipeline import AudioPipeline
from pipeline.audio_wav import WavSource
from pipeline.broker import Broker
from pipeline.router import route
from pipeline.scripture import ScriptureIndex, load_index
from pipeline.translator import Translator

import numpy as np
from pipeline.audio_source import AudioSource



SOURCES = ("wav", "mic", "network")


def _pick_source() -> str:
    """AUDIO_SOURCE from the environment, else ask once at startup.

    Falls back to wav wherever there is no console to ask: uvicorn --reload
    imports the app in a multiprocessing child whose stdin is closed (input()
    raises EOFError there), and tools that import this module must not hang.
    isatty() is deliberately not used; input() is tried and EOFError means no
    console.
    """
    env = os.getenv("AUDIO_SOURCE")
    if env in SOURCES:
        return env
    if env:
        print(f"Ignoring unknown AUDIO_SOURCE={env!r}")

    choice = "wav"
    if multiprocessing.parent_process() is None:
        print("\nSelect audio source:")
        print("  1) wav     — replay sermon_20min.wav")
        print("  2) mic     — WO Mic / default input device")
        print("  3) network — /audio-in endpoint (needs capture_client.py)")
        try:
            while True:
                answer = input("Choice [1/2/3] (default 1): ").strip().lower() or "1"
                picked = {"1": "wav", "2": "mic", "3": "network"}.get(answer, answer)
                if picked in SOURCES:
                    choice = picked
                    break
                print("Invalid choice, try again.")
        except EOFError:
            print("No console input available, using wav")
    else:
        print("No console input in a subprocess, using wav (set AUDIO_SOURCE)")

    # Inherited by reload workers, so they use this choice and don't re-ask.
    os.environ["AUDIO_SOURCE"] = choice
    print(f"Audio source: {choice}")
    return choice


STATIC_DIR = Path(__file__).parent / "static"
WAV_PATH = Path("data/audio/sermon_20min.wav")
AUDIO_SOURCE = _pick_source()
CONTEXT_SEGMENTS = 3  # previous German segments fed to gemma as context
LEADING_JUNK = " ,.;:!?-–—…"

# The [llm→opus] label isn't encodable in a cp1252 console; without this the
# print would raise after the message was already published.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

broker = Broker(ring_size=30)
# Populated by make_source() when AUDIO_SOURCE=network. Used by /audio-in.
network_source = None


def make_source():
    global network_source
    if AUDIO_SOURCE == "wav":
        return WavSource(WAV_PATH, loop=True, realtime=True)
    elif AUDIO_SOURCE == "mic":
        from pipeline.audio_mic import MicSource
        return MicSource(device=None)
    elif AUDIO_SOURCE == "network":
        from pipeline.audio_network import NetworkSource
        network_source = NetworkSource()
        return network_source
    else:
        raise ValueError(f"Unknown AUDIO_SOURCE: {AUDIO_SOURCE}")


def clean_segment(text: str) -> str:
    """Drop the stray leading punctuation Whisper sometimes emits (",,")."""
    return text.lstrip(LEADING_JUNK).strip()


async def audio_generator(translator: Translator, scripture: ScriptureIndex | None):
    router_state: dict = {}
    recent_german: deque[str] = deque(maxlen=CONTEXT_SEGMENTS)
    in_flight: set[asyncio.Task] = set()

    async def translate(decision: str, german: str, context: list[str]):
        """Returns (english, route_used), or (None, decision) if both fail."""
        try:
            english = await asyncio.to_thread(
                translator.translate, decision, german, context
            )
            return english, decision
        except Exception as e:
            print(f"translate failed [{decision}]: {e!r}")
        if decision != "opus":
            # Fall back to opus so a gemma hiccup doesn't drop the subtitle.
            try:
                return await asyncio.to_thread(translator.translate, "opus", german), "opus"
            except Exception as e:
                print(f"translate fallback failed [opus]: {e!r}")
        return None, decision

    async def translate_and_publish(german: str):
        nonlocal router_state
        decision, router_state = route(german, router_state)
        # Snapshot before any await so context follows arrival order.
        context = list(recent_german)
        recent_german.append(german)

        t0 = time.perf_counter()
        requested = decision
        english, decision = await translate(requested, german, context)
        downgraded = decision != requested
        translate_ms = (time.perf_counter() - t0) * 1000
        if english is None:
            return

        canonical = None
        if scripture is not None:
            try:
                canonical = await asyncio.to_thread(scripture.find, german)
            except Exception as e:
                print(f"scripture lookup failed: {e!r}")

        message = {
            "text_de": german,
            "text_en": english,
            "timestamp_ms": int(time.time() * 1000),
            "kind": "final",
            "route": decision,
        }
        if downgraded:
            message["downgraded"] = True
        if canonical:
            message["canonical"] = canonical
        broker.publish(message)
        label = f"{requested}→{decision}" if downgraded else decision
        print(f"[{label}] {translate_ms:.0f}ms DE: {german}")
        print(f"        EN: {english}")
        if canonical:
            print(
                f"        SCRIPTURE {canonical['tier']} {canonical['ref']} "
                f"({canonical['score']}): {canonical['en']}"
            )

    def _done(task: asyncio.Task):
        in_flight.discard(task)
        if not task.cancelled() and task.exception() is not None:
            print(f"segment task failed: {task.exception()!r}")

    def on_transcript(german: str):
        german = clean_segment(german)
        if not german:
            return
        task = asyncio.create_task(translate_and_publish(german))
        in_flight.add(task)  # strong ref so the task isn't garbage collected
        task.add_done_callback(_done)

    pipeline = AudioPipeline(source=make_source(), on_transcript=on_transcript)
    try:
        await pipeline.run()
    except asyncio.CancelledError:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading translator (opus + tokenizer)...")
    translator = Translator()
    print("Warming up gemma...")
    translator.warmup()
    scripture = load_index()
    task = asyncio.create_task(audio_generator(translator, scripture))
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    queue, catch_up = broker.subscribe()

    try:
        for message in catch_up:
            await websocket.send_json(message)

        async def sender():
            try:
                while True:
                    message = await queue.get()
                    await websocket.send_json(message)
            except (WebSocketDisconnect, asyncio.CancelledError):
                pass

        async def receiver():
            try:
                while True:
                    await websocket.receive_text()
            except (WebSocketDisconnect, asyncio.CancelledError):
                pass

        sender_task = asyncio.create_task(sender())
        receiver_task = asyncio.create_task(receiver())

        done, pending = await asyncio.wait(
            {sender_task, receiver_task}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
    except WebSocketDisconnect:
        pass
    finally:
        broker.unsubscribe(queue)



@app.websocket("/audio-in")
async def audio_in_endpoint(websocket: WebSocket):
    if network_source is None:
        await websocket.close(code=1011, reason="server not in network audio mode")
        return

    await websocket.accept()
    client_id = f"{websocket.client.host}:{websocket.client.port}"
    network_source.client_connected(client_id)

    try:
        while True:
            data = await websocket.receive_bytes()
            chunk = np.frombuffer(data, dtype="<i2")  # little-endian int16
            if len(chunk) != AudioSource.CHUNK_SAMPLES:
                print(
                    f"audio-in: bad chunk size {len(chunk)}, "
                    f"expected {AudioSource.CHUNK_SAMPLES}"
                )
                continue
            network_source.push(chunk)
    except WebSocketDisconnect:
        pass
    finally:
        network_source.client_disconnected(client_id)