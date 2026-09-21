import time

import ctranslate2
import ollama
from transformers import AutoTokenizer

REFERENCE_PATH = "data/transcripts/sermon_reference.txt"
OUTPUT_PATH = "data/transcripts/translation_bakeoff_bible.tsv"
OPUS_MODEL_PATH = "models/opus-mt-tc-bible-big-gmw-en-ct2"
OPUS_TOKENIZER_NAME = "Helsinki-NLP/opus-mt-tc-bible-big-gmw-en"

LLM_MODEL = "gemma3:12b"
LLM_SYSTEM_PROMPT = (
    "Translate the following German sermon text to English. "
    "Output only the translation, no commentary."
)


def load_segments(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def clean_field(text):
    # Tabs/newlines inside a field would break the TSV columns.
    return " ".join(text.split())


def translate_opus(translator, tokenizer, text):
    tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
    tokens = [">>eng<<"] + tokens
    results = translator.translate_batch([tokens])
    out_tokens = results[0].hypotheses[0]
    return tokenizer.decode(
        tokenizer.convert_tokens_to_ids(out_tokens), skip_special_tokens=True
    )

def translate_llm(text):
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    )
    return response["message"]["content"]


def main():
    segments = load_segments(REFERENCE_PATH)
    print(f"Loaded {len(segments)} segments from {REFERENCE_PATH}")

    print("Loading opus-mt...")
    translator = ctranslate2.Translator(
        OPUS_MODEL_PATH, device="cuda", compute_type="float16"
    )
    tokenizer = AutoTokenizer.from_pretrained(OPUS_TOKENIZER_NAME)

    print(f"Warming up {LLM_MODEL} (loads it into VRAM, not timed)...")
    translate_llm("Hallo Welt.")

    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as out:
        out.write("segment_idx\tgerman\topus_en\topus_ms\tllm_en\tllm_ms\n")

        for idx, german in enumerate(segments):
            opus_start = time.perf_counter()
            opus_en = translate_opus(translator, tokenizer, german)
            opus_ms = (time.perf_counter() - opus_start) * 1000

            llm_start = time.perf_counter()
            llm_en = translate_llm(german)
            llm_ms = (time.perf_counter() - llm_start) * 1000

            row = "\t".join(
                [
                    str(idx),
                    clean_field(german),
                    clean_field(opus_en),
                    f"{opus_ms:.1f}",
                    clean_field(llm_en),
                    f"{llm_ms:.1f}",
                ]
            )
            out.write(row + "\n")
            print(f"[{idx + 1}/{len(segments)}] opus={opus_ms:.0f}ms llm={llm_ms:.0f}ms")

    print(f"\nWrote {len(segments)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
