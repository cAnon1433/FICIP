# F.I.C.I.P. — Documentation

A note before anything else: the line numbers below are a snapshot of `main.py`
as of this writing. Every time you edit the file, later lines shift — treat
these as "roughly here" rather than exact addresses. The section headers
(the `# ---` comment blocks in the code) are the more reliable way to find
your way around, since those move with their code.

---

## File layout

```
F.I.C.I.P./
├── main.py                  # the program
├── requirements.txt         # pip dependencies
├── personality/
│   └── elric.json           # Elric's character definition
├── docs/
│   └── README.md            # this file
├── venv/                    # your Python 3.11 virtual environment (not tracked/shared)
└── elric_memory.json        # generated automatically — conversation history, safe to delete
```

## Code sections in main.py (as of this snapshot)

| Lines     | Section                | What it does |
|-----------|-------------------------|--------------|
| 1–12      | Imports                | Standard library + third-party packages |
| 14–27     | Config constants        | All the tunable knobs — see table below |
| 30–40     | `load_personality()`    | Reads `personality/elric.json`, builds the system prompt |
| 43–46     | `NPCResponseSchema`     | Pydantic schema forcing Ollama to return valid structured JSON |
| 49–75     | Memory functions         | `load_memory()`, `save_memory()`, `trim_memory()` — persistent, capped conversation history |
| 78–130    | `Recorder` class         | Mic input — opens the audio stream once at startup, records on demand |
| 133–148   | Speech-to-text           | Loads Whisper, `transcribe_audio()` |
| 151–180   | Text-to-speech           | Loads Kokoro, `speak()`, `warmup_tts()` |
| 183–207   | `chat_with_npc()`        | Sends conversation to Ollama, updates + saves memory |
| 210–265   | Main loop                | Mic device listing, then the actual talk-to-Elric loop |

## Config constants — what to touch and why

All near the top of `main.py`:

- `SAMPLE_RATE = 16000` — required by Whisper, don't change unless you know why.
- `MEMORY_FILE` — filename for persisted conversation history.
- `MAX_TURNS_KEPT = 10` — how many back-and-forth exchanges get kept before older ones are trimmed. Raise this for more context/continuity, at the cost of more RAM and slower responses over a long session.
- `WHISPER_MODEL_SIZE` — currently `"base.en"`. Options in increasing size/accuracy: `tiny.en` → `base.en` → `small.en` → `medium.en`. Bigger costs more RAM and is slower per transcription.
- `WHISPER_COMPUTE_TYPE = "int8"` — smallest CPU memory footprint. Leave this alone unless you have a specific reason.
- `TTS_SAMPLE_RATE = 24000` — fixed by Kokoro itself, same regardless of voice. Not something to change.
- `PERSONALITY_FILE` — path to the JSON file defining the NPC's character, prompt, and voice (see below).

## Kokoro voices

Voice and language are set per-character inside that character's personality
JSON file (e.g. `personality/elric.json`), via the `"voice"` and `"lang_code"`
fields — not in `main.py` directly. To change Elric's voice, edit those two
fields in `personality/elric.json`.

Only American and British English voice IDs are confirmed below (these are the two you can safely use). Kokoro supports 7 more languages (Japanese, Mandarin, French, Spanish, Hindi, Italian, Brazilian Portuguese) but those need extra packages (e.g. `misaki[ja]`) and their exact voice IDs weren't verified for this doc — check the model card on Hugging Face (`hexgrad/Kokoro-82M`, file `VOICES.md`) before using one.

**American English** — set `"lang_code": "a"`:
- Female: `af_alloy`, `af_aoede`, `af_bella`, `af_heart`, `af_jessica`, `af_kore`, `af_nicole`, `af_nova`, `af_river`, `af_sarah`, `af_sky`
- Male: `am_adam`, `am_echo`, `am_eric`, `am_fenrir`, `am_liam`, `am_michael`, `am_onyx`, `am_puck`, `am_santa`

**British English** — set `"lang_code": "b"`:
- Female: `bf_alice`, `bf_emma`, `bf_isabella`, `bf_lily`
- Male: `bm_daniel`, `bm_fable`, `bm_george`, `bm_lewis`

To use one: set `"voice"` to the ID in the personality file. If switching between American/British, also update `"lang_code"` to match — the phoneme processing is language-specific.

**Voice blending**: two different things are supported, and it matters which you use:
- **Equal-weight blend** (native to Kokoro): `"am_fenrir,am_onyx"` — comma-separated, no colons, averages them 50/50.
- **Custom weighted blend** (built specifically for this script, not native to Kokoro): `"am_fenrir:60,am_onyx:40"` — the `resolve_voice()` function in `main.py` loads each voice tensor individually and computes the weighted average itself before playback. This only works inside this script — the raw `kokoro` package would fail on that syntax if used directly.

**Voice Suggestions**: below are some custom voices to try using default voices blended together:
- `"use am_fenrir:60,bm_george"` lang_code a for a neutral male voice

## Adding a new NPC

The personality system is built to scale to more than one character:

1. Copy `personality/elric.json` to a new file, e.g. `personality/marla.json`.
2. Edit `name`, `role`, `rules`, `voice`, and `lang_code` to match the new character.
3. In `main.py`, change `PERSONALITY_FILE` to point at the new file (or extend the script to switch between multiple NPCs at runtime — not built yet, but the file structure supports it).

## Mood-driven voice delivery

Kokoro doesn't have native emotion conditioning — it won't change tone based
on punctuation (`!` vs `.`) or wording alone. Instead, `main.py` uses the
`mood` field already returned by the LLM (`neutral` / `annoyed` / `furious`)
to adjust three things after synthesis: speaking pace, a light pitch shift,
and volume. This is defined in the `MOOD_TTS_SETTINGS` dict near the top of
the Text-to-speech section — tune the numbers there if a mood sounds off.
This mapping is generic (not Elric-specific), so any NPC using the same
mood schema in their `NPCResponseSchema` gets this for free.

## Known constraints / gotchas

- **8GB RAM total.** Running `gemma3:4b` + Whisper + Kokoro simultaneously is workable but tight. If things get sluggish, drop to `gemma3:1b` before downgrading Whisper or Kokoro, since RAM was the intended sacrifice point, not audio/transcription quality.
- **Python 3.14 breaks Kokoro's dependencies.** This project must run inside the `venv/` folder (Python 3.11), not your system Python. Always `source venv/bin/activate` before running.
- **`espeak-ng` is a system dependency**, not a pip package — installed via `brew install espeak-ng`, required for Kokoro's phoneme processing.
- **macOS mic permissions**: if the mic silently captures nothing, check System Settings → Privacy & Security → Microphone and make sure your terminal app is allowed.