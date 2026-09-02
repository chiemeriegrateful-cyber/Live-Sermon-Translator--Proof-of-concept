import time

from faster_whisper import WhisperModel



MODEL_PATH = "models/whisper-turbo-de-int8"
AUDIO_PATH = "data/audio/sermon_2min.wav"

load_start = time.perf_counter()
model = WhisperModel(MODEL_PATH, device="cuda", compute_type="float16")
load_time = time.perf_counter() - load_start

transcribe_start = time.perf_counter()
segments, info = model.transcribe(AUDIO_PATH, language="de", vad_filter=True, condition_on_previous_text=False,)

for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

transcribe_time = time.perf_counter() - transcribe_start

print(f"\nModel load time: {load_time:.2f}s")
print(f"Transcription time: {transcribe_time:.2f}s")
