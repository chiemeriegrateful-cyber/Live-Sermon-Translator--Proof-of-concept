# pipeline/translator.py
import os
from typing import Protocol

import ctranslate2
import ollama
from transformers import AutoTokenizer

OPUS_MODEL_PATH = "models/opus-mt-de-en-ct2"
OPUS_TOKENIZER_NAME = "Helsinki-NLP/opus-mt-de-en"
LLM_MODEL = os.getenv("LLM_MODEL", "gemma3:12b")
# Keep the model resident so a quiet stretch doesn't trigger an 8s reload.
# Negative = never unload (ollama semantics).
LLM_KEEP_ALIVE = float(os.getenv("LLM_KEEP_ALIVE", "-1"))
# Live gemma calls take ~150-850ms. Past this the sermon has moved on, so give
# up and let the caller fall back to opus rather than wait for a perfect line.
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "5"))
# Startup warm-up can include a cold model load (~8s), so it gets more room.
LLM_WARMUP_TIMEOUT_S = float(os.getenv("LLM_WARMUP_TIMEOUT_S", "60"))
LLM_SYSTEM_PROMPT = (
    "Translate the following German sermon text to English. "
    "Output only the translation, no commentary. "
    "If previous sentences are given as context, use them only to understand "
    "where the speaker is (for example which Bible passage is being quoted). "
    "Never translate the context, only the final text."
)


class TranslationBackend(Protocol):
    """One translation engine. Add a class with this shape to compare a new
    model, then register it in Translator.backends."""

    def translate(self, text: str, context: list[str] | None = None) -> str: ...


class OpusBackend:
    def __init__(self, model_path: str = OPUS_MODEL_PATH,
                 tokenizer_name: str = OPUS_TOKENIZER_NAME):
        self.model = ctranslate2.Translator(
            model_path, device="cuda", compute_type="float16"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def translate(self, text: str, context: list[str] | None = None) -> str:
        # opus-mt is sentence-level; context is accepted for interface parity.
        tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(text))
        results = self.model.translate_batch([tokens])
        out_tokens = results[0].hypotheses[0]
        return self.tokenizer.decode(
            self.tokenizer.convert_tokens_to_ids(out_tokens),
            skip_special_tokens=True,
        )


class OllamaBackend:
    def __init__(self, model: str = LLM_MODEL,
                 system_prompt: str = LLM_SYSTEM_PROMPT):
        self.model = model
        self.system_prompt = system_prompt
        self.client = ollama.Client(timeout=LLM_TIMEOUT_S)
        self.warmup_client = ollama.Client(timeout=LLM_WARMUP_TIMEOUT_S)

    def warmup(self) -> None:
        """Load the model into VRAM before the first live segment."""
        self.warmup_client.chat(
            model=self.model,
            messages=[{"role": "user", "content": "Hallo Welt."}],
            keep_alive=LLM_KEEP_ALIVE,
        )

    def translate(self, text: str, context: list[str] | None = None) -> str:
        if context:
            user_content = (
                "Previous sentences (context only, do not translate):\n"
                + "\n".join(context)
                + "\n\nText to translate:\n"
                + text
            )
        else:
            user_content = text
        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
            ],
            keep_alive=LLM_KEEP_ALIVE,
        )
        return response["message"]["content"]


class Translator:
    def __init__(self):
        self.backends: dict[str, TranslationBackend] = {
            "opus": OpusBackend(),
            "llm": OllamaBackend(),
        }

    def translate(self, kind: str, text: str,
                  context: list[str] | None = None) -> str:
        return self.backends[kind].translate(text, context)

    def warmup(self) -> None:
        self.backends["llm"].warmup()
