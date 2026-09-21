import asyncio
import os
import time
from typing import Callable

import numpy as np
import torch
from faster_whisper import WhisperModel

from pipeline.audio_source import AudioSource

WHISPER_MODEL_PATH = "models/whisper-turbo-de-int8"

VAD_THRESHOLD = 0.5
VAD_WINDOW_SAMPLES = 512
SILENCE_CHUNKS_TO_END = 5
MAX_UTTERANCE_CHUNKS = int(os.getenv("MAX_UTTERANCE_CHUNKS", "60"))
# Past SOFT_MAX_UTTERANCE_CHUNKS an utterance ends at the next pause of
# SOFT_PAUSE_CHUNKS instead of being chopped mid-sentence at the hard limit.
# 0 disables the soft cut, which is the default: on sermon_20min.wav (a
# preacher pausing about every 8s) 66% of utterances hit the 6s hard cut, and
# SOFT_PAUSE_CHUNKS=2 with MAX_UTTERANCE_CHUNKS=100 cuts that to 44% but raises
# median utterance length from 6s to 8.5s, i.e. every subtitle ~2.5s later.
SOFT_MAX_UTTERANCE_CHUNKS = int(os.getenv("SOFT_MAX_UTTERANCE_CHUNKS", "60"))
SOFT_PAUSE_CHUNKS = int(os.getenv("SOFT_PAUSE_CHUNKS", "0"))

# Preachers speak in punchy fragments separated by pauses just over
# SILENCE_CHUNKS_TO_END ("Da war viel Volk. ... Ein großes Getümmel!"). Cut on
# every pause and each fragment reaches Whisper and the translator alone.
# An utterance with less than SHORT_SPEECH_CHUNKS of actual speech therefore
# waits SHORT_SILENCE_CHUNKS_TO_END for the next fragment and, if speech
# resumes, keeps growing as one utterance. 0 disables merging.
SHORT_SPEECH_CHUNKS = int(os.getenv("SHORT_SPEECH_CHUNKS", "15"))
SHORT_SILENCE_CHUNKS_TO_END = int(os.getenv("SHORT_SILENCE_CHUNKS_TO_END", "10"))


class Segmenter:
    """Turns per-chunk speech flags into utterances. Pure, so it can be
    replayed offline against recorded audio."""

    def __init__(
        self,
        silence_to_end: int = SILENCE_CHUNKS_TO_END,
        max_chunks: int = MAX_UTTERANCE_CHUNKS,
        short_speech: int = SHORT_SPEECH_CHUNKS,
        short_silence_to_end: int = SHORT_SILENCE_CHUNKS_TO_END,
        soft_max_chunks: int = SOFT_MAX_UTTERANCE_CHUNKS,
        soft_pause_chunks: int = SOFT_PAUSE_CHUNKS,
    ):
        self.silence_to_end = silence_to_end
        self.max_chunks = max_chunks
        self.short_speech = short_speech
        self.short_silence_to_end = short_silence_to_end
        self.soft_max_chunks = soft_max_chunks
        self.soft_pause_chunks = soft_pause_chunks
        self._reset()

    def _reset(self):
        self.buffer: list[np.ndarray] = []
        self.silence = 0
        self.speech_chunks = 0
        self.started = False

    def feed(self, chunk: np.ndarray, has_speech: bool):
        """Returns (audio, speech_chunks) when an utterance ends, else None."""
        if has_speech:
            self.buffer.append(chunk)
            self.silence = 0
            self.speech_chunks += 1
            self.started = True
        elif self.started:
            self.buffer.append(chunk)
            self.silence += 1
        else:
            return None

        is_short = self.speech_chunks < self.short_speech
        needed = self.short_silence_to_end if is_short else self.silence_to_end
        # Past the soft limit, end at the next short pause instead of waiting
        # for a full one; the hard limit still cuts mid-speech as a last resort.
        soft_cut = (
            self.soft_pause_chunks > 0
            and len(self.buffer) >= self.soft_max_chunks
            and self.silence >= self.soft_pause_chunks
        )
        if self.silence >= needed or soft_cut or len(self.buffer) >= self.max_chunks:
            done = (np.concatenate(self.buffer), self.speech_chunks)
            self._reset()
            return done
        return None


class AudioPipeline:
    def __init__(self, source: AudioSource, on_transcript: Callable[[str], None]):
        self.source = source
        self.on_transcript = on_transcript

        print("Loading Silero VAD...")
        from silero_vad import load_silero_vad
        self.vad_model = load_silero_vad()
        print("Loading Whisper turbo...")
        self.whisper = WhisperModel(
            WHISPER_MODEL_PATH, device="cuda", compute_type="float16"
        )

    def _is_speech(self, chunk: np.ndarray) -> bool:
    # Silero requires exactly 512 samples per call at 16kHz.
    # Our chunk is 1600 samples (100ms), so slice it into 512-sample windows.
        for start in range(0, len(chunk) - VAD_WINDOW_SAMPLES + 1, VAD_WINDOW_SAMPLES):
            window = chunk[start : start + VAD_WINDOW_SAMPLES]
            audio_f = torch.from_numpy(window.astype(np.float32) / 32768.0)
            prob = self.vad_model(audio_f, self.source.SAMPLE_RATE).item()
            if prob > VAD_THRESHOLD:
                return True
        return False

    async def _transcribe(self, buffer: np.ndarray) -> str:
        # Do not add initial_prompt (previous transcript as context). Tried on
        # sermon_20min.wav: with either whisper-turbo-de-int8 or
        # whisper-turbo-base the output is empty (or "... ..." loops) for every
        # utterance, on GPU float16/float32 and CPU int8, for any prompt length,
        # with the fallback thresholds disabled. The gemma context in
        # server.py is where earlier segments help instead.
        def _sync():
            segments, _ = self.whisper.transcribe(
                buffer.astype(np.float32) / 32768.0,
                language="de",
                vad_filter=False,
                condition_on_previous_text=False,
            )
            return " ".join(seg.text.strip() for seg in segments).strip()

        return await asyncio.to_thread(_sync)

    async def run(self):
        segmenter = Segmenter()

        async for chunk in self.source.stream():
            done = segmenter.feed(chunk, self._is_speech(chunk))
            if done is None:
                continue

            audio, _ = done
            start = time.perf_counter()
            try:
                german = await self._transcribe(audio)
            except Exception as e:
                # One bad utterance must not stall the live stream.
                print(f"asr failed, skipping utterance: {e!r}")
                german = ""
            ms = (time.perf_counter() - start) * 1000
            if german:
                print(f"asr {ms:.0f}ms: {german}")
                self.on_transcript(german)