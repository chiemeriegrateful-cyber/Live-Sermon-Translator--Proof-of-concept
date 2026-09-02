import jiwer
from faster_whisper import WhisperModel

LARGE_MODEL_PATH = "models/whisper-large-de"
TURBO_MODEL_PATH = "models/whisper-turbo-de-int8"
AUDIO_PATH = "data/audio/sermon_2min.wav"

NORMALIZE = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)


def transcribe(model_path, audio_path):
    model = WhisperModel(model_path, device="cuda", compute_type="float16")
    segments, info = model.transcribe(
        audio_path, language="de", vad_filter=True, condition_on_previous_text=False
    )
    return " ".join(segment.text.strip() for segment in segments)


large_output = transcribe(LARGE_MODEL_PATH, AUDIO_PATH)
turbo_output = transcribe(TURBO_MODEL_PATH, AUDIO_PATH)

error_rate = jiwer.wer(
    large_output,
    turbo_output,
    reference_transform=NORMALIZE,
    hypothesis_transform=NORMALIZE,
)

print("Large output:\n", large_output)
print("\nTurbo output:\n", turbo_output)
print(f"\nWER (turbo vs. large): {error_rate:.4f}")
