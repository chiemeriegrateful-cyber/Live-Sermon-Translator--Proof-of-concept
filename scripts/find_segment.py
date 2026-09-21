from faster_whisper import WhisperModel

MODEL_PATH = "models/whisper-turbo-de-int8"
AUDIO_PATH = "data/audio/sermon_20min.wav"
TARGET = 56

model = WhisperModel(MODEL_PATH, device="cuda", compute_type="float16")
segments, _ = model.transcribe(
    AUDIO_PATH,
    language="de",
    vad_filter=True,
    condition_on_previous_text=False,
)

for idx, seg in enumerate(segments):
    if idx == TARGET:
        print(f"Segment {idx}: {seg.start:.2f}s -> {seg.end:.2f}s")
        print(f"Duration: {seg.end - seg.start:.2f}s")
        print(f"Text: {seg.text}")
        break