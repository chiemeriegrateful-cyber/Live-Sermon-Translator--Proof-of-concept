# Live Sermon Translation: Architecture

**Version** 1.0
**Scope** One church, German sermon in, English subtitles out, subtitle-only v1 with an audio output path reserved.
**Deployment target** Single machine on the church LAN. i5-13600K, 16GB GPU, no internet dependency at service time.

---

## 1. Design principles

| Principle | What it means in this codebase |
|---|---|
| Ports and adapters | Every model sits behind an abstract interface. Swapping Whisper for Voxtral, or opus-mt for an LLM, touches one adapter file and one config line. |
| Small tailored models per stage | No single general model does everything. Each stage runs the smallest model that solves its own problem well. |
| Output sinks are subscribers | The pipeline publishes translated segments to a bus. It has no knowledge of subtitles, TTS, or logging. Adding audio output later means writing one class. |
| Deterministic where possible | Scripture references and glossary terms are resolved by lookup, not by a model. Models do not get to invent verse numbers. |
| Degrade, never stall | Under load or on failure, the pipeline drops to a faster path or skips a segment. It never blocks and never queues indefinitely. |
| Offline by construction | No component makes a network call during a service. |

---

## 2. Model registry

All weights are downloaded once and stored under `models/`. Nothing here requires an API key or internet access at runtime.

| Stage | Model | Size on disk | Runtime | Device | Notes |
|---|---|---|---|---|---|
| VAD | `silero-vad` (ONNX) | ~2 MB | onnxruntime | CPU | Utterance boundary detection with hangover. |
| ASR | `primeline/whisper-large-v3-turbo-german`, converted to CTranslate2 int8 | ~1.0 GB | faster-whisper | GPU | German fine-tune of Turbo. 4 decoder layers instead of 32. |
| ASR (alt) | `openai/whisper-large-v3-turbo` int8 | ~1.0 GB | faster-whisper | GPU | Fallback if the German fine-tune underperforms on your preacher. |
| ASR (alt) | Voxtral Realtime | ~8 GB | vLLM or transformers | GPU | Native streaming via causal encoder. Evaluate only if chunk latency proves unacceptable. |
| Translation, fast path | `Helsinki-NLP/opus-mt-de-en` | ~300 MB | CTranslate2 | GPU or CPU | Purpose-built DE to EN. Sub-100ms per sentence. No context handling. |
| Translation, context path | `gemma3:12b` at Q4_K_M, or `qwen3:8b` at Q5_K_M | 7-8 GB / ~6 GB | Ollama | GPU | Pronoun resolution across chunks, glossary enforcement, sermon register. |
| Scripture | none | ~200 KB CSV | lookup | CPU | Book lexicon (DE and EN forms) plus chapter/verse regex. |
| Glossary | none | SQLite table | lookup | CPU | Church-preferred renderings, applied pre and post translation. |
| TTS (future) | Piper, English voice | ~60 MB | piper | CPU | Not installed in v1. The sink interface reserves its slot. |
| Embeddings (future) | `multilingual-e5-small` | ~470 MB | sentence-transformers | GPU | Only if sermon search is added. |

**VRAM budget at steady state**

```
Whisper turbo int8 + beam overhead      ~2.0 GB
opus-mt (CTranslate2 int8)              ~0.3 GB
Gemma 3 12B Q4_K_M                      ~7.5 GB
CUDA context + fragmentation            ~1.0 GB
-----------------------------------------------
Total                                   ~10.8 GB of 16 GB
```

Headroom is deliberate. Both models stay resident for the whole service with `keep_alive` pinned so no reload stall lands mid-sentence.

---

## 3. Latency budget

Measured from the moment the preacher stops speaking a sentence to the moment text appears on a phone.

| Stage | Fast path (opus-mt) | Context path (LLM) |
|---|---|---|
| VAD hangover | 300 ms | 300 ms |
| ASR (4s chunk, GPU int8) | 150-250 ms | 150-250 ms |
| Enrichment (lookups) | < 10 ms | < 10 ms |
| Translation | 40-90 ms | 500-800 ms |
| Bus and WebSocket push (LAN) | ~20 ms | ~20 ms |
| Client render | ~50 ms | ~50 ms |
| **Total** | **~0.6 s** | **~1.3 s** |

Target is under 4 seconds. Both paths clear it with room, which is why the hybrid strategy is affordable.

---

## 4. Translation strategy

Run both translators. Route per segment.

```
if segment has unresolved pronouns
   or contains a scripture reference
   or contains a glossary term
   or previous segment ended mid-clause:
       -> LLM path (context-aware)
else:
       -> opus-mt path (fast)

if translation queue depth > 3:
       -> force opus-mt path for everything until drained
```

The last rule is the backpressure valve. If the preacher speeds up or the GPU stalls, quality degrades gracefully instead of the subtitles falling behind.

Benchmark both on a real recording before committing. DE to EN is a high-resource pair and opus-mt may be closer to the LLM than expected, in which case the routing simplifies.

---

## 5. Component structure

```
sermon-translate/
├── app/
│   ├── core/                 domain models, events, errors, ids
│   ├── ports/                abstract interfaces (no implementations)
│   │   ├── audio_source.py
│   │   ├── vad.py
│   │   ├── asr.py
│   │   ├── translator.py
│   │   ├── tts.py
│   │   ├── output_sink.py
│   │   └── repositories.py
│   ├── adapters/
│   │   ├── audio/            microphone_source.py, file_source.py
│   │   ├── vad/              silero_vad.py
│   │   ├── asr/              faster_whisper_asr.py, voxtral_asr.py
│   │   ├── translate/        opus_mt.py, ollama_llm.py, hybrid.py
│   │   └── tts/              piper_tts.py          (stub in v1)
│   ├── context/              enricher.py, glossary.py, scripture.py
│   ├── pipeline/             stages.py, queues.py, orchestrator.py
│   ├── delivery/             segment_bus.py, subtitle_sink.py,
│   │                         transcript_log_sink.py, ws_gateway.py
│   ├── persistence/          sqlite_repos.py, migrations/
│   ├── api/                  routers/, schemas/
│   └── config/               settings.py, model_registry.py
├── web/                      attendee.html, operator.html, app.js
├── data/                     bible_books.csv, glossary.csv, sermon.db
├── models/                   downloaded weights
├── tests/                    fixtures/sermon_sample.wav
├── docker-compose.yml
└── config.yaml
```

The `ports/` directory is the contract. Nothing in `adapters/` imports anything from another adapter.

---

## 6. Configuration

```yaml
audio:
  device: "hw:1,0"
  sample_rate: 16000
  chunk_ms: 4000
  vad:
    model: silero
    threshold: 0.5
    min_speech_ms: 400
    hangover_ms: 300

asr:
  adapter: faster_whisper
  model: models/whisper-large-v3-turbo-german-ct2-int8
  device: cuda            # flip to cpu without code changes
  compute_type: int8_float16
  beam_size: 1
  language: de
  condition_on_previous_text: false

translation:
  strategy: hybrid        # opus_mt | ollama | hybrid
  fast:
    adapter: opus_mt
    model: models/opus-mt-de-en-ct2
    device: cuda
  context:
    adapter: ollama
    model: gemma3:12b
    keep_alive: -1
    num_ctx: 2048
    temperature: 0.2
  context_window_segments: 4
  target_languages: [en]

context_engine:
  glossary_enabled: true
  scripture_enabled: true
  bible_books: data/bible_books.csv

delivery:
  sinks: [subtitle, transcript_log]
  websocket_path: /ws/subtitles
  max_listeners: 120

pipeline:
  queue_depth:
    audio: 8
    asr: 4
    translate: 4
  overflow_policy: drop_oldest
  degrade_threshold: 3

persistence:
  url: sqlite:///data/sermon.db
  persist_segments: true
```

---

## 7. API surface

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/session/start` | Begin a service. Warms models, opens mic. |
| POST | `/api/session/stop` | End the service, flush sinks. |
| GET | `/api/session/current` | State, uptime, listener count, queue depths. |
| GET | `/api/languages` | Active target languages. |
| GET | `/api/health` | Model residency, device, GPU memory, last segment age. |
| GET | `/api/metrics` | Per-stage latency percentiles. |
| GET | `/api/glossary`, POST, DELETE | Manage church terminology. |
| GET | `/join` | QR landing page for attendees. |
| WS | `/ws/subtitles?lang=en` | Attendee subtitle stream. |
| WS | `/ws/operator` | Operator telemetry stream. |

WebSocket message from the subtitle channel:

```json
{
  "type": "segment",
  "seq": 142,
  "lang": "en",
  "text": "Turn with me to Romans chapter eight.",
  "source_text": "Schlagt mit mir Römer Kapitel acht auf.",
  "scripture": [{ "book": "Romans", "chapter": 8 }],
  "final": true,
  "ts": 1755600000.123
}
```

---

## 8. Failure modes and responses

| Failure | Detection | Response |
|---|---|---|
| Mic unplugged | Audio source read error | Operator alert, auto-retry every 2s, session stays alive |
| GPU OOM | CUDA allocation error | Drop LLM path, run opus-mt only, alert operator |
| Ollama model evicted | Inference call latency spike | Reload with `keep_alive: -1`, degrade to opus-mt during reload |
| ASR hallucination on silence | Empty or repeated VAD segment | VAD gate rejects segments under `min_speech_ms`, dedupe repeated output |
| Queue backpressure | Depth over threshold | Force fast path, then drop oldest audio chunks |
| Listener WiFi drop | WebSocket close | Client auto-reconnects, requests last N segments on rejoin |
| Whole service crashes | Process exit | Docker restart policy, session state recovered from SQLite |

---

## 9. Build phases

**Phase 0, offline evaluation.** No pipeline. Record 20 minutes of your actual service. Run ASR on it standalone, measure WER and real-time factor. Run both translators on the resulting transcript, compare side by side. This decides the routing strategy in section 4.

**Phase 1, vertical slice on file input.** `FileSource` feeds the full pipeline as if live. All the hard problems (chunk seams, pronoun continuity, glossary consistency) surface here and are debuggable on repeat.

**Phase 2, live microphone.** Swap `FileSource` for `MicrophoneSource`. One line of config. Everything downstream is already tested.

**Phase 3, delivery.** WebSocket gateway, attendee page, QR join. Test with two phones at home, then four people at a real service.

**Phase 4, hardening.** Operator page, health checks, restart policy, the failure table above.

**Phase 5, audio output.** Write `TTSSink`, subscribe it to the bus. Nothing upstream changes.

---

## 10. Diagrams

Rendered separately as `.mermaid` files:

- `component-architecture.mermaid` — runtime components and their wiring
- `class-ports-adapters.mermaid` — interfaces and implementations
- `class-domain.mermaid` — domain model
- `sequence-utterance.mermaid` — one utterance end to end, including fallback
- `state-session.mermaid` — session lifecycle
- `er-persistence.mermaid` — SQLite schema
- `deployment.mermaid` — physical deployment on the church LAN
