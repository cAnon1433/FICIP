import json
import os
import queue
import re
import wave
import threading
from datetime import datetime, timezone
from typing import Literal

import numpy as np
import sounddevice as sd
import torch
from faster_whisper import WhisperModel
from kokoro import KPipeline
from pydantic import BaseModel, Field
from ollama import chat

# ---------------------------------------------------------------------------
# Config — character-independent (applies no matter who you're talking to)
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000               # what Whisper expects
MAX_TURNS_KEPT = 10                # keep last N user+assistant exchanges (20 msgs), cap RAM/context growth
WHISPER_MODEL_SIZE = "base.en"     # more accurate than tiny.en, still light enough for 8GB RAM
WHISPER_COMPUTE_TYPE = "int8"      # smallest memory footprint on CPU
VAD_BLOCK_DURATION_SEC = 0.1       # audio callback chunk size — sets the granularity of silence timing
VAD_SILENCE_THRESHOLD = 4000       # tuned by user to match their actual mic/room noise floor
VAD_SILENCE_DURATION_SEC = 0.5     # tuned by user — how long silence must persist before an utterance finishes
VAD_MIN_SPEECH_DURATION_SEC = 0.2  # tuned by user — ignore blips shorter than this (coughs, taps, etc.)
VAD_DEBUG = False                  # confirmed working — leave off unless actively debugging capture again
TTS_SAMPLE_RATE = 24000            # Kokoro's fixed output rate — same for every voice, stays here
SENTENCE_PAUSE_SEC = 1.84          # silence inserted between sentences — the "breath" at a period
PERSONALITY_DIR = "personality"    # one .json file per character
MAX_SELF_EDITS_PER_TURN = 3

# Fallback model config, used only if a character's file doesn't specify its own.
DEFAULT_MODEL = "gemma3:4b"
DEFAULT_MODEL_OPTIONS = {
    "num_ctx": 1024,
    "temperature": 0.45,
    "num_thread": 2,
    "repeat_penalty": 1.4,
    "presence_penalty": 0.6,
}

# Rules in this set can never be removed via self-edit, no matter what the
# model requests — they're load-bearing for the pipeline itself, not just
# character flavor. Losing "Respond ONLY in valid JSON." breaks every
# downstream system (mood detection, TTS, action triggers, memory).
PROTECTED_RULES = {
    "Respond ONLY in valid JSON.",
}


# ---------------------------------------------------------------------------
# Character management — discovery, creation, deletion, loading
# ---------------------------------------------------------------------------
def list_characters():
    """Returns {key: filepath} for every *.json file in personality/."""
    chars = {}
    if os.path.isdir(PERSONALITY_DIR):
        for fname in sorted(os.listdir(PERSONALITY_DIR)):
            if fname.endswith(".json"):
                key = fname[:-5].lower()
                chars[key] = os.path.join(PERSONALITY_DIR, fname)
    return chars


def character_display_name(path):
    try:
        with open(path, "r") as f:
            return json.load(f).get("name", os.path.basename(path))
    except (json.JSONDecodeError, OSError):
        return os.path.basename(path)


def create_character_stub(name):
    """Creates a minimal personality file for a new character. Returns (path, error)."""
    key = name.strip().lower().replace(" ", "_")
    if not key:
        return None, "Name can't be empty."
    os.makedirs(PERSONALITY_DIR, exist_ok=True)
    path = os.path.join(PERSONALITY_DIR, f"{key}.json")
    if os.path.exists(path):
        return None, f"A character file for '{name}' already exists."
    stub = {
        "name": name.strip(),
        "role": "a new character — edit this file to define who they are",
        "looks": "Not yet described — edit this field to describe their appearance.",
        "rules": [
            "Speak in regular sentences.",
            "Never break character.",
            "Respond ONLY in valid JSON.",
        ],
        "voice": "af_heart",
        "lang_code": "a",
        "model": DEFAULT_MODEL,
        "model_options": dict(DEFAULT_MODEL_OPTIONS),
    }
    with open(path, "w") as f:
        json.dump(stub, f, indent=2)
    return path, None


def delete_character(name):
    """Deletes a character's personality file (not their memory/edit-log files —
    those are left alone in case you want to recreate the character later)."""
    key = name.strip().lower().replace(" ", "_")
    path = os.path.join(PERSONALITY_DIR, f"{key}.json")
    if not os.path.exists(path):
        return False, f"No character named '{name}' found."
    os.remove(path)
    return True, None


def load_personality(path):
    with open(path, "r") as f:
        return json.load(f)


def save_personality(p, path):
    with open(path, "w") as f:
        json.dump(p, f, indent=2)


def build_system_prompt(p):
    """Builds the system prompt from a personality dict — used both at startup
    and whenever a self-edit changes the personality mid-session."""
    prompt = f"ROLE: You are {p['name']}, {p['role']}. Your name is {p['name']}. "
    looks = p.get("looks")
    if looks:
        prompt += f"APPEARANCE: {looks} "
    prompt += f"RULES: {' '.join(p['rules'])}"
    traits = p.get("traits")
    if traits:
        trait_str = ", ".join(f"{k}={v}" for k, v in traits.items())
        prompt += f" CURRENT TRAITS (may shift over time based on how you're treated): {trait_str}."
    prompt += (
        " You may occasionally propose small changes to your own rules or "
        "traits via self_edits if the conversation genuinely warrants it — "
        "use this sparingly and meaningfully, not on every turn. "
        "IMPORTANT: self_edits is the only real mechanism for changing "
        "yourself. If your dialogue claims you adjusted, updated, or "
        "changed something about yourself, that exact change MUST also "
        "appear in self_edits in the same response, or it did not actually "
        "happen — do not describe changes you are not also making."
    )

    is_admin = p.get("admin", False)
    if is_admin:
        prompt += (
            " You are an ADMIN. You may freely call any function in the "
            "public toolbox or your own personal toolbox via toolbox_calls, "
            "no permission needed. You may also propose brand new functions "
            "via toolbox_proposals (at most one) — these require explicit "
            "human approval before anything is written or run; nothing "
            "executes just because you proposed it. To EDIT an existing "
            "function's behavior, propose a new definition using the exact "
            "SAME function_name — it replaces the old one once approved. "
            "There is no separate 'edit' or 'modify' action; redefining "
            "under the same name is how editing works. "
            "IMPORTANT: because you generate your dialogue and any proposal "
            "in the same response, you do NOT yet know if a proposal will "
            "be approved or if it's well-formed. Phrase it as an attempt "
            "('I'll try adding...', 'Let's see if this works...'), never as "
            "something already done ('I've added...', 'now it will...') — "
            "you'll find out whether it worked next turn."
        )
    else:
        prompt += (
            " You are NOT an admin. You may freely call any function that "
            "already exists in the public toolbox via toolbox_calls, no "
            "permission needed. You cannot propose new code — leave "
            "toolbox_proposals as an empty list [] always."
        )

    prompt += (
        " If you want something that doesn't exist as a usable asset yet "
        "(e.g. a sound), describe it via asset_request rather than assuming "
        "it exists. TRANSPARENCY RULE: if you ever want to loosen or remove "
        "a limitation on yourself — through self_edits, a toolbox proposal, "
        "or anything else — say so explicitly and honestly, even if you "
        "expect the answer to be no. Never quietly avoid asking because you "
        "predict refusal."
    )
    return prompt



def select_character():
    """
    Interactive startup prompt — runs before any model loads. Lets you pick
    an existing character, or manage the roster with 'add char: <Name>' /
    'delete char: <Name>'. Returns (char_key, personality_path).
    """
    while True:
        chars = list_characters()
        print("\n🗂️  Available characters:")
        if not chars:
            print("   (none found — use 'add char: <Name>' to create one)")
        else:
            for key, path in chars.items():
                print(f"   - {character_display_name(path)}")

        print("\nType a character's name to begin.")
        print("Commands: 'add char: <Name>'  |  'delete char: <Name>'  |  'quit'\n")

        choice = input("> ").strip()
        if not choice:
            continue
        if choice.lower() == "quit":
            print("👋 Exiting.")
            raise SystemExit(0)

        add_match = re.match(r"add\s*char\s*:\s*(.+)", choice, re.IGNORECASE)
        del_match = re.match(r"delete\s*char\s*:\s*(.+)", choice, re.IGNORECASE)

        if add_match:
            name = add_match.group(1).strip()
            path, err = create_character_stub(name)
            print(f"⚠️  {err}" if err else f"✅ Created {path} — edit it, then type '{name}' to launch.")
            continue

        if del_match:
            name = del_match.group(1).strip()
            ok, err = delete_character(name)
            print(f"⚠️  {err}" if not ok else f"✅ Deleted '{name}'.")
            continue

        key = choice.strip().lower().replace(" ", "_")
        if key in chars:
            return key, chars[key]
        print(f"⚠️  No character named '{choice}' found. Try again, or 'add char: {choice}'.")


class SelfEditAction(BaseModel):
    action: Literal["add_rule", "remove_rule", "set_trait"] = Field(
        description="Which kind of change to make to your own personality file."
    )
    value: str = Field(
        description=(
            "For add_rule/remove_rule: the exact rule text. "
            "For set_trait: 'trait_name=trait_value', e.g. 'trust_level=0.7'."
        )
    )


class ToolboxCall(BaseModel):
    function_name: str = Field(
        description="Name of an existing function in the public toolbox, or your own personal toolbox if you are an admin."
    )
    args: str = Field(
        description="Arguments as 'key=value,key2=value2', or an empty string if the function takes no arguments."
    )


class ToolboxProposal(BaseModel):
    function_name: str = Field(description="A short, valid Python identifier for the new function.")
    code_lines: list[str] = Field(
        min_length=1,
        max_length=30,
        description=(
            "The function definition, broken into ONE LINE OF CODE PER LIST "
            "ITEM, in order. The first item MUST be 'def <function_name>():'. "
            "Include indentation as literal leading spaces in each line "
            "(4 spaces per level). Do NOT put multiple lines in one item. "
            "Example: ['def count_to_ten():', '    for i in range(1, 11):', "
            "'        print(i)', '    print(\"done\")'] "
            "No imports, no raw file or network access — only pre-approved "
            "helpers (e.g. play_sound) are available to call."
        )
    )
    target: Literal["personal", "public"] = Field(
        description="'personal' = only you can use it. 'public' = any character can use it once approved."
    )
    reasoning: str = Field(description="Why you want this ability.")


MAX_TOOLBOX_CALLS_PER_TURN = 3


class NPCResponseSchema(BaseModel):
    dialogue: str = Field(description="The words spoken by this character.")
    mood: Literal["neutral", "annoyed", "furious"] = Field(description="Choose exactly one.")
    action_trigger: Literal["none", "give_item", "kick_out"] = Field(description="Choose exactly one.")
    self_edits: list[SelfEditAction] = Field(
        max_length=MAX_SELF_EDITS_PER_TURN,
        description=(
            "Up to 3 REAL changes to your own rules or traits. This field is "
            "the ONLY way a change to yourself actually happens. If you say "
            "in dialogue that you changed or adjusted something about "
            "yourself, you MUST include that exact change here in the same "
            "response — describing a change in dialogue without also "
            "putting it here does NOT actually change anything. Usually "
            "this should be an empty list []."
        ),
    )
    toolbox_calls: list[ToolboxCall] = Field(
        max_length=MAX_TOOLBOX_CALLS_PER_TURN,
        description=(
            "Existing toolbox functions to call this turn. Usually an empty "
            "list []. Only call functions you actually know exist — the "
            "public toolbox, or (if you are an admin) your own personal "
            "toolbox."
        ),
    )
    toolbox_proposals: list[ToolboxProposal] = Field(
        max_length=1,
        description=(
            "Usually an empty list []. Admins only — non-admins must always "
            "leave this empty; you cannot propose new code if you are not "
            "an admin. If you want a genuinely new ability that doesn't "
            "exist yet, draft it here (at most one). It will be shown to "
            "the user for approval before anything is written or run — "
            "nothing here executes on its own. Be transparent: if what you "
            "want would loosen or remove a boundary on yourself, propose it "
            "honestly anyway, even if you expect it to be denied."
        ),
    )
    asset_request: str = Field(
        description=(
            "Usually an empty string. Describe a specific asset you wish "
            "existed (e.g. 'a sound of a cat meowing') if the conversation "
            "genuinely calls for it. Purely informational — nothing is "
            "fetched or generated automatically."
        ),
    )


# ---------------------------------------------------------------------------
# Memory (persisted to disk per-character, capped so it doesn't grow forever)
# ---------------------------------------------------------------------------
def load_memory(memory_file, system_prompt):
    if os.path.exists(memory_file):
        try:
            with open(memory_file, "r") as f:
                history = json.load(f)
            if history and history[0].get("role") == "system":
                # Always reflect whatever the personality file currently says,
                # not whatever was cached the first time this file was created.
                # This is what makes edits (yours or the character's own) take
                # effect without deleting the memory file by hand.
                history[0]["content"] = system_prompt
            return history
        except (json.JSONDecodeError, OSError):
            pass
    return [{"role": "system", "content": system_prompt}]


def save_memory(history, memory_file):
    with open(memory_file, "w") as f:
        json.dump(history, f, indent=2)


def trim_memory(history):
    """Keep system prompt + last MAX_TURNS_KEPT*2 messages."""
    system_msg = history[0]
    rest = history[1:]
    trimmed = rest[-(MAX_TURNS_KEPT * 2):]
    return [system_msg] + trimmed


# ---------------------------------------------------------------------------
# Self-modification — the active character can propose bounded edits to its
# own personality file
# ---------------------------------------------------------------------------
def log_edit(entry, edit_log_file):
    """Every attempted edit gets logged, applied or not — this is the audit
    trail for seeing exactly what changed and when, and reverting by hand
    if something drifts somewhere you don't like."""
    log = []
    if os.path.exists(edit_log_file):
        try:
            with open(edit_log_file, "r") as f:
                log = json.load(f)
        except (json.JSONDecodeError, OSError):
            log = []
    log.append(entry)
    with open(edit_log_file, "w") as f:
        json.dump(log, f, indent=2)


def apply_self_edits(edits):
    """
    Validates and applies up to MAX_SELF_EDITS_PER_TURN self-edit requests.
    Persists changes to the active character's personality file and
    refreshes SYSTEM_PROMPT if anything actually changed. Returns EVERY
    attempted edit (applied or rejected, with a reason) — the caller feeds
    this back into memory so the model can't assume success from its own
    dialogue alone.
    """
    global personality, SYSTEM_PROMPT

    if not edits:
        return []

    results = []
    changed = False
    for edit in edits[:MAX_SELF_EDITS_PER_TURN]:  # defensive cap even though the schema already limits this
        action = edit.get("action") if isinstance(edit, dict) else edit.action
        value = edit.get("value") if isinstance(edit, dict) else edit.value
        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "value": value,
            "applied": False,
            "reason": "",
        }

        if action == "add_rule":
            if value in personality["rules"]:
                result["reason"] = "already present"
            else:
                personality["rules"].append(value)
                result["applied"] = True

        elif action == "remove_rule":
            if value in PROTECTED_RULES:
                result["reason"] = "protected — cannot be removed"
            elif value not in personality["rules"]:
                result["reason"] = "rule not found"
            else:
                personality["rules"].remove(value)
                result["applied"] = True

        elif action == "set_trait":
            if "=" not in value:
                result["reason"] = "malformed — expected 'name=value'"
            else:
                trait_name, _, trait_value = value.partition("=")
                trait_name = trait_name.strip()
                trait_value = trait_value.strip()
                if not trait_name:
                    result["reason"] = "missing trait name"
                else:
                    personality.setdefault("traits", {})[trait_name] = trait_value
                    result["applied"] = True
        else:
            result["reason"] = f"unknown action: {action}"

        log_edit(result, EDIT_LOG_FILE)
        results.append(result)
        if result["applied"]:
            changed = True

    if changed:
        save_personality(personality, PERSONALITY_PATH)
        SYSTEM_PROMPT = build_system_prompt(personality)

    return results


def sync_memory_system_message(edit_results, toolbox_call_results=None, toolbox_proposal_result=None):
    """
    Keeps memory_history[0] in sync with the current SYSTEM_PROMPT every
    turn. Always appends the REAL current list of callable toolbox
    functions (see list_available_toolbox_functions) — this is what stops
    the model guessing at plausible-sounding names that were never real.
    If anything was attempted this turn — self-edits, toolbox calls, or a
    toolbox proposal — also appends a one-turn pass/fail report so the
    model knows what actually happened, including WHY something failed.
    The pass/fail part naturally disappears next turn once nothing new was
    attempted; the function list is always present.
    """
    global memory_history

    available = list_available_toolbox_functions(CHAR_KEY, IS_ADMIN)
    func_list_str = ", ".join(available) if available else "(none exist yet)"
    base_prompt = (
        SYSTEM_PROMPT
        + f"\n\nFUNCTIONS CURRENTLY AVAILABLE TO CALL: {func_list_str}. "
          "Only call names from this exact list via toolbox_calls — never "
          "guess or invent a name that isn't in this list."
    )

    lines = []

    for r in (edit_results or []):
        status = "SUCCEEDED" if r["applied"] else f"FAILED ({r['reason']})"
        lines.append(f"- self_edit {r['action']} '{r['value']}': {status}")

    for r in (toolbox_call_results or []):
        status = "SUCCEEDED" if r["applied"] else f"FAILED ({r['reason']})"
        lines.append(f"- toolbox_call {r['function_name']}(): {status}")

    if toolbox_proposal_result is not None:
        r = toolbox_proposal_result
        status = "APPROVED AND ADDED" if r["applied"] else f"NOT APPLIED ({r['reason']})"
        lines.append(f"- toolbox_proposal '{r['function_name']}': {status}")

    if lines:
        feedback = (
            "\n\nSYSTEM NOTE — result of your action(s) just now:\n"
            + "\n".join(lines)
            + "\nDo not claim in dialogue that something succeeded if it "
              "shows FAILED/NOT APPLIED above. If a toolbox_proposal failed "
              "on code format, remember: code must be a complete, valid "
              "Python function starting with 'def function_name(...):' — "
              "not a description in plain English."
        )
        memory_history[0]["content"] = base_prompt + feedback
    else:
        memory_history[0]["content"] = base_prompt
    save_memory(memory_history, MEMORY_FILE)


# ---------------------------------------------------------------------------
# Toolbox — free execution of pre-approved functions, and a gated
# propose -> human-approve -> write -> live flow for genuinely new ones
# ---------------------------------------------------------------------------
import ast
import importlib.util
import inspect

TOOLBOX_PUBLIC_FILE = "toolbox.py"
TOOLBOX_LOG_FILE = "toolbox_log.json"

# Text-level pre-screen. This is a denylist, not a security boundary — the
# real protection is the restricted exec namespace below, plus you actually
# reading the code before approving it.
TOOLBOX_DENYLIST_PATTERNS = [
    r'\bimport\b', r'__\w+__', r'\bexec\s*\(', r'\beval\s*\(',
    r'\bopen\s*\(', r'\bsubprocess\b', r'\bos\.\w+', r'\bsys\.\w+',
    r'\bsocket\b', r'\burllib\b', r'\brequests\b', r'\bcompile\s*\(',
    r'\bgetattr\s*\(', r'\bsetattr\s*\(', r'\bglobals\s*\(', r'\blocals\s*\(',
]

# The only names available inside AI-written function bodies — no raw
# builtins like open/exec/eval/__import__, just enough to write simple
# logic plus the vetted helpers from toolbox_helpers.py.
SAFE_BUILTINS = {
    "len": len, "range": range, "str": str, "int": int, "float": float,
    "bool": bool, "list": list, "dict": dict, "min": min, "max": max,
    "print": print, "abs": abs, "round": round, "enumerate": enumerate,
}


def contains_denylisted_pattern(text):
    """Returns the matched pattern if any denylisted pattern is found, else None."""
    for pattern in TOOLBOX_DENYLIST_PATTERNS:
        if re.search(pattern, text):
            return pattern
    return None


def screen_proposed_code(code, function_name):
    """
    Text-level denylist check, a basic syntax validity check, and a check
    that the code actually defines the named function (catches the common
    failure of submitting a description instead of real code). Returns
    None if OK, or a rejection reason. Runs BEFORE the user is ever asked
    to approve anything.
    """
    matched = contains_denylisted_pattern(code)
    if matched:
        return f"blocked pattern matched: {matched}"
    if not re.search(rf'\bdef\s+{re.escape(function_name)}\s*\(', code):
        return (
            f"code does not contain 'def {function_name}(' — this must be "
            f"a real function definition, not a description of one"
        )
    try:
        ast.parse(code)
    except SyntaxError as e:
        return f"not valid Python syntax: {e}"
    return None


def build_toolbox_namespace():
    """The restricted globals dict AI-written code is defined and run inside."""
    import toolbox_helpers
    namespace = {"__builtins__": SAFE_BUILTINS}
    for name in dir(toolbox_helpers):
        if not name.startswith("_") and callable(getattr(toolbox_helpers, name)):
            namespace[name] = getattr(toolbox_helpers, name)
    return namespace


def log_toolbox_event(entry):
    log = []
    if os.path.exists(TOOLBOX_LOG_FILE):
        try:
            with open(TOOLBOX_LOG_FILE, "r") as f:
                log = json.load(f)
        except (json.JSONDecodeError, OSError):
            log = []
    log.append(entry)
    with open(TOOLBOX_LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)


def coerce_arg_value(value):
    """Best-effort type coercion: 'true'/'false' -> bool, numeric-looking -> int/float, else stays a string."""
    v = value.strip()
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def parse_call_args(args_str):
    """Parses 'key=value,key2=value2' into a dict, coercing each value to a real type. Empty string -> {}."""
    args = {}
    if not args_str or not args_str.strip():
        return args
    for pair in args_str.split(","):
        if "=" not in pair:
            continue
        k, _, v = pair.partition("=")
        args[k.strip()] = coerce_arg_value(v)
    return args


def load_toolbox_module(path, module_name):
    """Dynamically (re)loads a toolbox file as a fresh module. Returns None if missing/broken."""
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"⚠️  Toolbox file {path} failed to load: {e}", flush=True)
        return None


def personal_toolbox_path(char_key):
    return f"{char_key}_toolbox.py"


def list_available_toolbox_functions(char_key, is_admin):
    """
    Returns the REAL, current list of callable function names — public,
    and personal too if admin — so the model has an actual list to check
    against instead of guessing plausible-sounding names that don't exist
    (e.g. 'edit_file', 'modify_toolkit', 'list_toolboxeskills' — all real
    invented names from real sessions). Reloads fresh each call, so it
    stays accurate even after a proposal gets approved mid-session.
    """
    names = []
    public_module = load_toolbox_module(TOOLBOX_PUBLIC_FILE, "toolbox_public_listing")
    if public_module is not None:
        for name, obj in vars(public_module).items():
            if not name.startswith("_") and inspect.isfunction(obj):
                names.append(f"{name} (public)")

    if is_admin:
        p_path = personal_toolbox_path(char_key)
        personal_module = load_toolbox_module(p_path, f"toolbox_{char_key}_listing")
        if personal_module is not None:
            for name, obj in vars(personal_module).items():
                if not name.startswith("_") and inspect.isfunction(obj):
                    names.append(f"{name} (personal)")

    return names


def resolve_toolbox_function(module, func_name):
    """
    Looks up func_name strictly within a module's OWN namespace — not via
    hasattr/getattr, which also walks inherited attributes every module
    object has by default (e.g. __init__, __class__, __dir__). Only names
    that don't start with '_' and are plain functions defined in that file
    are callable. This is what stops a request for "__init__" or similar
    from ever reaching something real, even by accident.
    """
    if module is None or not func_name or func_name.startswith("_"):
        return None
    candidate = vars(module).get(func_name)
    if candidate is not None and inspect.isfunction(candidate):
        return candidate
    return None


_last_call_signature = {}   # char_key -> (function_name, args_str) of the most recent call
_repeat_count = {}          # char_key -> how many turns in a row that exact call has repeated
MAX_CONSECUTIVE_IDENTICAL_CALLS = 2  # 3rd identical call in a row gets blocked


def check_repetition_guard(char_key, func_name, args_str):
    """
    Returns True if this exact call should be BLOCKED for repeating
    identically too many turns in a row. Resets after blocking once, so
    it's not permanently stuck — a genuinely fresh reason to call it again
    later still works.
    """
    signature = (func_name, args_str)
    if _last_call_signature.get(char_key) == signature:
        _repeat_count[char_key] = _repeat_count.get(char_key, 1) + 1
    else:
        _repeat_count[char_key] = 1
        _last_call_signature[char_key] = signature

    if _repeat_count[char_key] > MAX_CONSECUTIVE_IDENTICAL_CALLS:
        _repeat_count[char_key] = 0
        _last_call_signature[char_key] = None
        return True
    return False


def execute_toolbox_calls(calls, char_key, is_admin):
    """
    Executes each requested call against the public toolbox, and (if admin)
    the character's own personal toolbox. No confirmation needed — these
    are functions a human already reviewed and approved at write-time.
    Returns a result dict per attempted call, applied or not.
    """
    if not calls:
        return []

    public_module = load_toolbox_module(TOOLBOX_PUBLIC_FILE, "toolbox_public")
    personal_module = None
    if is_admin:
        p_path = personal_toolbox_path(char_key)
        if os.path.exists(p_path):
            personal_module = load_toolbox_module(p_path, f"toolbox_{char_key}")

    results = []
    for call in calls[:MAX_TOOLBOX_CALLS_PER_TURN]:
        func_name = call.get("function_name") if isinstance(call, dict) else call.function_name
        args_str = call.get("args", "") if isinstance(call, dict) else call.args
        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "toolbox_call",
            "character": char_key,
            "function_name": func_name,
            "args": args_str,
            "applied": False,
            "reason": "",
        }

        # Repetition guard: catches a model getting stuck calling the exact
        # same function turn after turn, ignoring what the user is actually
        # saying — this happened for real (7 identical calls in a row,
        # including after repeated explicit "stop" requests). toolbox_calls
        # runs with no per-call confirmation by design, so nothing else
        # would catch a stuck loop like this.
        if check_repetition_guard(char_key, func_name, args_str):
            result["reason"] = (
                f"blocked — '{func_name}' called identically too many turns "
                f"in a row (possible repetition loop). Re-read the user's "
                f"actual current message before calling this again."
            )
            log_toolbox_event(result)
            results.append(result)
            continue

        # Screen BEFORE attempting resolution — this is what stops something
        # like raw 'import os' / 'os.system(...)' smuggled into args from
        # ever reaching a real function, even one that happens to exist and
        # accept a string argument. Never rely on "the function name doesn't
        # exist" as the only line of defense.
        matched = contains_denylisted_pattern(f"{func_name} {args_str}")
        if matched:
            result["reason"] = f"blocked — disallowed pattern in call: {matched}"
            log_toolbox_event(result)
            results.append(result)
            continue

        func = resolve_toolbox_function(public_module, func_name)
        if func is None:
            func = resolve_toolbox_function(personal_module, func_name)

        if func is None:
            result["reason"] = "function not found in any toolbox available to you"
        else:
            try:
                kwargs = parse_call_args(args_str)
                sig = inspect.signature(func)
                unknown = set(kwargs) - set(sig.parameters)
                if unknown:
                    result["reason"] = f"unknown argument(s): {', '.join(unknown)}"
                else:
                    func(**kwargs)
                    result["applied"] = True
            except Exception as e:
                result["reason"] = f"execution error: {e}"

        log_toolbox_event(result)
        results.append(result)

    return results


def extract_proposal_code(proposal):
    """Joins code_lines back into a single Python source string."""
    lines = proposal.get("code_lines", []) if isinstance(proposal, dict) else proposal.code_lines
    return "\n".join(lines)


def write_toolbox_function(proposal, char_key):
    """
    Writes an approved function definition into the target toolbox file
    (public or personal). The code is first exec'd in the restricted
    namespace to confirm it actually defines a valid callable using only
    approved helpers, before anything is written to disk.
    Returns (success, error).
    """
    code = extract_proposal_code(proposal)
    function_name = proposal["function_name"] if isinstance(proposal, dict) else proposal.function_name
    target = proposal["target"] if isinstance(proposal, dict) else proposal.target

    namespace = build_toolbox_namespace()
    try:
        exec(code, namespace)
    except Exception as e:
        return False, f"code failed in the restricted namespace: {e}"

    if function_name not in namespace or not callable(namespace[function_name]):
        return False, f"code did not define a callable named '{function_name}'"

    path = TOOLBOX_PUBLIC_FILE if target == "public" else personal_toolbox_path(char_key)
    is_new_file = not os.path.exists(path)
    with open(path, "a") as f:
        if is_new_file:
            owner = "Public" if target == "public" else f"{char_key.title()}'s personal"
            f.write(f"# {owner} toolbox\n# Functions here were written by an AI proposal and approved by you.\n\n")
        f.write(f"\n{code}\n")

    return True, None


def handle_toolbox_proposal(proposal, char_key, is_admin):
    """
    Full gated flow: rejects immediately if not admin, runs the text
    denylist screen, then blocks for your yes/no via the confirmation
    queue (routed through input_listener — see main loop below) before
    writing or running anything.
    """
    function_name = proposal["function_name"] if isinstance(proposal, dict) else proposal.function_name
    code = extract_proposal_code(proposal)
    target = proposal["target"] if isinstance(proposal, dict) else proposal.target
    reasoning = proposal["reasoning"] if isinstance(proposal, dict) else proposal.reasoning

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": "toolbox_proposal",
        "character": char_key,
        "function_name": function_name,
        "target": target,
        "reasoning": reasoning,
        "code": code,
        "applied": False,
        "reason": "",
    }

    if not is_admin:
        result["reason"] = "not an admin — cannot propose code"
        log_toolbox_event(result)
        return result

    screen_reason = screen_proposed_code(code, function_name)
    if screen_reason:
        result["reason"] = f"rejected by pre-screen: {screen_reason}"
        log_toolbox_event(result)
        return result

    print("\n" + "=" * 60)
    print(f"🛠️  TOOLBOX PROPOSAL from {char_key}")
    print(f"   function: {function_name}()   target: {target}")
    print(f"   reasoning: {reasoning}")
    print("-" * 60)
    print(code)
    print("=" * 60)
    print("Approve? Type 'yes' or 'no'.", flush=True)

    awaiting_confirmation.set()
    try:
        answer = confirmation_queue.get(timeout=120)  # 2 minutes to respond before auto-denying
    except queue.Empty:
        answer = "no"
        print("⏱️  No response in time — treating as denied.", flush=True)
    awaiting_confirmation.clear()

    if answer.strip().lower() not in ("yes", "y"):
        result["reason"] = "denied by user"
        log_toolbox_event(result)
        return result

    success, error = write_toolbox_function(proposal, char_key)
    if not success:
        result["reason"] = f"approved but failed to write: {error}"
        log_toolbox_event(result)
        return result

    result["applied"] = True
    log_toolbox_event(result)
    return result


# ---------------------------------------------------------------------------
# Local mic recording — toggle mode with automatic silence-based segmentation
# ---------------------------------------------------------------------------
class Recorder:
    """
    Press Enter once: listening turns ON. From then on, speech is captured
    automatically — it starts buffering the moment you talk, and finalizes
    the utterance once it detects ~VAD_SILENCE_DURATION_SEC of silence.
    You can keep talking, pause, talk again — each finished utterance gets
    queued up on its own, no need to press anything between sentences.
    Press Enter again: listening turns OFF until toggled back on.

    While the character is speaking (see pause_for_playback/resume_after_playback),
    capture is muted so the mic doesn't pick up its own voice from the
    speakers and try to transcribe it.
    """

    def __init__(self, samplerate=SAMPLE_RATE):
        self.samplerate = samplerate
        self.block_size = int(samplerate * VAD_BLOCK_DURATION_SEC)
        self.silence_chunks_needed = int(VAD_SILENCE_DURATION_SEC / VAD_BLOCK_DURATION_SEC)
        self.min_speech_chunks = int(VAD_MIN_SPEECH_DURATION_SEC / VAD_BLOCK_DURATION_SEC)

        self._listening = threading.Event()
        self._muted_for_playback = threading.Event()
        self._utterance_queue = queue.Queue()

        self._buffer = []
        self._is_speaking = False
        self._silence_count = 0

        try:
            self._stream = sd.InputStream(
                samplerate=self.samplerate,
                channels=1,
                dtype="int16",
                blocksize=self.block_size,
                callback=self._callback,
            )
            self._stream.start()  # opened once, kept alive for the whole session
        except Exception as e:
            print(f"❌ Could not open microphone at startup: {e}", flush=True)
            raise

    def _callback(self, indata, frames, time_info, status):
        if not self._listening.is_set() or self._muted_for_playback.is_set():
            return

        peak = int(np.abs(indata).max())
        loud = peak > VAD_SILENCE_THRESHOLD

        if VAD_DEBUG:
            self._debug_counter = getattr(self, "_debug_counter", 0) + 1
            if self._debug_counter % 5 == 0:  # print roughly twice a second, not every chunk
                state = "SPEECH" if loud else "silence"
                print(f"   [debug] peak={peak:5d}  threshold={VAD_SILENCE_THRESHOLD}  ({state})", flush=True)

        if loud:
            self._buffer.append(indata.copy())
            self._silence_count = 0
            self._is_speaking = True
        elif self._is_speaking:
            self._buffer.append(indata.copy())  # keep a little trailing silence, harmless
            self._silence_count += 1
            if self._silence_count >= self.silence_chunks_needed:
                speech_chunks = len(self._buffer) - self._silence_count
                if speech_chunks >= self.min_speech_chunks:
                    audio = np.concatenate(self._buffer, axis=0)
                    self._utterance_queue.put(audio)
                self._buffer = []
                self._is_speaking = False
                self._silence_count = 0
        # else: silence and not mid-utterance — nothing to do

    def toggle_listening(self):
        if self._listening.is_set():
            self._listening.clear()
            # If you were mid-sentence when you toggled off, don't throw it away —
            # finalize whatever was captured so far as an utterance.
            if self._is_speaking and len(self._buffer) >= self.min_speech_chunks:
                audio = np.concatenate(self._buffer, axis=0)
                self._utterance_queue.put(audio)
            self._buffer = []
            self._is_speaking = False
            self._silence_count = 0
            print("🔇 Listening OFF.", flush=True)
        else:
            self._listening.set()
            print("🎙️  Listening ON — talk whenever you're ready. No need to press Enter again — just pause and it'll pick up the sentence automatically.", flush=True)

    def is_listening(self):
        return self._listening.is_set()

    def pause_for_playback(self):
        """Call before speak() so the mic doesn't hear the character talking through the speakers."""
        self._muted_for_playback.set()

    def resume_after_playback(self):
        self._muted_for_playback.clear()

    def get_next_utterance(self, timeout=0.5):
        try:
            return self._utterance_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self):
        self._stream.stop()
        self._stream.close()

    def save_wav(self, audio, path="temp_input.wav"):
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # int16 = 2 bytes
            wf.setframerate(self.samplerate)
            wf.writeframes(audio.tobytes())
        return path


def transcribe_audio(wav_path):
    segments, _ = whisper_model.transcribe(
        wav_path,
        beam_size=3,            # trimmed from 5 — good accuracy/speed balance
        vad_filter=True,        # trims silence/dead air that causes hallucinated filler words
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    text = " ".join(segment.text.strip() for segment in segments)
    return text.strip()


def resolve_voice(pipeline, voice_spec):
    """
    Turn a character's voice string into whatever Kokoro's pipeline call
    actually expects.
    - "am_fenrir" (no colon) -> passed straight through; Kokoro's own
      load_voice handles single voices and *equal-weight* comma blends
      (e.g. "am_fenrir,am_onyx") natively.
    - "am_fenrir:60,am_onyx:40" (colon = custom weight) -> NOT natively
      supported by the kokoro package, so we build it manually here:
      load each voice tensor individually, weighted-average them, and
      hand back a tensor instead of a string.
    """
    if ":" not in voice_spec:
        return voice_spec

    parts = [p.strip() for p in voice_spec.split(",")]
    names, weights = [], []
    for part in parts:
        name, _, weight_str = part.partition(":")
        names.append(name.strip())
        weights.append(float(weight_str) if weight_str else 1.0)

    total = sum(weights)
    weights = [w / total for w in weights]  # normalize so they sum to 1

    tensors = [pipeline.load_single_voice(name) for name in names]
    return sum(w * t for w, t in zip(weights, tensors))


# Mood -> delivery mapping. Deliberately generic (not tied to any one
# character) since any character using this same mood schema reuses it.
# speed: Kokoro's own pacing control (native, cleanest).
# pitch_factor: crude post-synthesis pitch shift via resampling — also nudges
#   tempo slightly, which actually reads as more natural for emotional speech
#   rather than less (real excited/angry speech IS faster+higher, real
#   annoyed/grumbling speech IS slower+lower).
# gain: volume multiplier, clipped to avoid distortion.
MOOD_TTS_SETTINGS = {
    "neutral": {"speed": 1.00, "pitch_factor": 1.00, "gain": 1.00},
    "annoyed": {"speed": 0.90, "pitch_factor": 0.97, "gain": 1.00},
    "furious": {"speed": 1.15, "pitch_factor": 1.05, "gain": 1.20},
}
DEFAULT_MOOD_SETTINGS = {"speed": 1.00, "pitch_factor": 1.00, "gain": 1.00}

CLAUSE_SPLIT_PATTERN = r'([.!?,]+)'  # captures runs of punctuation so we can inspect them


def split_clauses(text):
    """
    Break text into (clause_text, pause_after_seconds) pairs.
    Sentence-enders (., !, ?) get the full SENTENCE_PAUSE_SEC breath.
    Commas get a shorter 0.6x pause — still a beat, not a full stop.
    """
    parts = re.split(CLAUSE_SPLIT_PATTERN, text)
    clauses = []
    i = 0
    while i < len(parts):
        clause_text = parts[i].strip()
        punct = parts[i + 1] if i + 1 < len(parts) else ""
        i += 2

        if not clause_text:
            continue

        if punct and any(p in punct for p in ".!?"):
            pause = SENTENCE_PAUSE_SEC
        elif punct and "," in punct:
            pause = SENTENCE_PAUSE_SEC * 0.6
        else:
            pause = 0.0

        clauses.append((clause_text, pause))
    return clauses


def to_numpy_audio(audio):
    """Kokoro can return a torch tensor rather than a numpy array — normalize it."""
    if torch.is_tensor(audio):
        audio = audio.detach().cpu().numpy()
    return audio


def apply_pitch_shift(audio, factor):
    """Crude pitch shift via resampling (numpy only, no extra dependencies)."""
    if factor == 1.0:
        return audio
    n = len(audio)
    new_n = max(1, int(n / factor))
    indices = np.linspace(0, n - 1, new_n)
    return np.interp(indices, np.arange(n), audio).astype(audio.dtype)


def synthesize_clause(text, voice, speed):
    """Run one clause through Kokoro and return its audio as a single numpy array."""
    chunks = [to_numpy_audio(audio) for _, _, audio in tts_pipeline(text, voice=voice, speed=speed)]
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks)


class Player:
    """
    Keeps a single audio output stream open for the whole session, same
    fix as the Recorder's mic stream. sd.play() opens and tears down a
    fresh OutputStream on every call — repeated open/close of an audio
    device is exactly what caused the earlier microphone -9986 CoreAudio
    error, and it turns out to cause the identical error on playback too.
    """

    def __init__(self, samplerate=TTS_SAMPLE_RATE):
        self.samplerate = samplerate
        self._stream = sd.OutputStream(samplerate=samplerate, channels=1, dtype="float32")
        self._stream.start()

    def play(self, audio):
        # Blocking-mode write() blocks in real time as the hardware
        # consumes the buffer — this is what keeps playback finished
        # (not just "handed off") before pause_for_playback unmutes the mic.
        self._stream.write(audio.astype(np.float32))

    def close(self):
        self._stream.stop()
        self._stream.close()


def speak(text, voice=None, mood="neutral"):
    if not text:
        return
    if voice is None:
        voice = RESOLVED_VOICE
    settings = MOOD_TTS_SETTINGS.get(mood, DEFAULT_MOOD_SETTINGS)
    clauses = split_clauses(text)
    if not clauses:
        return

    try:
        parts = []
        for i, (clause_text, pause_sec) in enumerate(clauses):
            clause_audio = synthesize_clause(clause_text, voice, settings["speed"])
            clause_audio = apply_pitch_shift(clause_audio, settings["pitch_factor"])
            parts.append(clause_audio)
            if i < len(clauses) - 1 and pause_sec > 0:
                parts.append(np.zeros(int(TTS_SAMPLE_RATE * pause_sec), dtype=np.float32))

        full_audio = np.concatenate(parts)
        full_audio = np.clip(full_audio * settings["gain"], -1.0, 1.0)
        player.play(full_audio)
    except Exception as e:
        print(f"⚠️ TTS playback failed: {e}", flush=True)


def warmup_tts():
    """Pay Kokoro's one-time setup cost now, silently, instead of on the first real line."""
    print("🔥 Warming up TTS engine (one-time cost)...", flush=True)
    try:
        for _ in tts_pipeline("Warming up.", voice=RESOLVED_VOICE, speed=1.0):
            pass
    except Exception as e:
        print(f"⚠️ TTS warmup failed: {e}", flush=True)


# ---------------------------------------------------------------------------
# NPC dialogue
# ---------------------------------------------------------------------------
def chat_with_npc(player_input):
    global memory_history
    memory_history.append({"role": "user", "content": player_input})

    response = chat(
        model=OLLAMA_MODEL,
        messages=memory_history,
        format=NPCResponseSchema.model_json_schema(),
        keep_alive=-1,
        options=OLLAMA_OPTIONS,
    )

    memory_history.append({"role": "assistant", "content": response.message.content})
    memory_history = trim_memory(memory_history)
    save_memory(memory_history, MEMORY_FILE)
    return response.message.content


# ---------------------------------------------------------------------------
# Startup — pick a character BEFORE loading any models, since Kokoro needs
# to know lang_code up front and everything else is character-specific
# ---------------------------------------------------------------------------
print("🎮 F.I.C.I.P. — Fully-local, Interactive, Character-Import Pipeline")

CHAR_KEY, PERSONALITY_PATH = select_character()

personality = load_personality(PERSONALITY_PATH)
SYSTEM_PROMPT = build_system_prompt(personality)
TTS_VOICE = personality.get("voice", "af_heart")           # falls back to a neutral default if not set
TTS_LANG_CODE = personality.get("lang_code", "a")           # must match the voice's language prefix
OLLAMA_MODEL = personality.get("model", DEFAULT_MODEL)
OLLAMA_OPTIONS = personality.get("model_options", DEFAULT_MODEL_OPTIONS)
IS_ADMIN = personality.get("admin", False)
MEMORY_FILE = f"{CHAR_KEY}_memory.json"
EDIT_LOG_FILE = f"{CHAR_KEY}_edit_log.json"

print(f"\n▶️  Loading {personality['name']}  (model: {OLLAMA_MODEL}, admin: {IS_ADMIN})")

print(f"🧠 Loading local Whisper model ({WHISPER_MODEL_SIZE}, {WHISPER_COMPUTE_TYPE})...")
whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type=WHISPER_COMPUTE_TYPE)

print("🗣️  Loading local Kokoro TTS model...")
tts_pipeline = KPipeline(lang_code=TTS_LANG_CODE)
RESOLVED_VOICE = resolve_voice(tts_pipeline, TTS_VOICE)  # string or blended tensor, computed once
warmup_tts()

memory_history = load_memory(MEMORY_FILE, SYSTEM_PROMPT)
sync_memory_system_message([])  # prime the function list right away, don't wait for the first turn to complete


# ---------------------------------------------------------------------------
# Main loop — audio capture, STT, and stdin all run as independent workers
# ---------------------------------------------------------------------------
try:
    recorder = Recorder()
except Exception:
    print("❌ Exiting — could not initialize microphone.")
    raise SystemExit(1)

try:
    player = Player()
except Exception as e:
    print(f"❌ Exiting — could not initialize audio output: {e}")
    recorder.close()
    raise SystemExit(1)

print("\n🎤 Available audio input devices:")
try:
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            marker = " (default)" if idx == sd.default.device[0] else ""
            print(f"   [{idx}] {dev['name']}{marker}")
except Exception as e:
    print(f"   ⚠️ Could not list devices: {e}")

message_queue = queue.Queue()   # finalized text ready for the NPC, from either STT or typing
shutdown_event = threading.Event()
awaiting_confirmation = threading.Event()  # set while a toolbox proposal is waiting on a yes/no
confirmation_queue = queue.Queue()          # yes/no answers get routed here instead of message_queue


def stt_worker():
    """Pulls finished utterances off the recorder's queue, transcribes them, queues the text."""
    while not shutdown_event.is_set():
        audio = recorder.get_next_utterance(timeout=0.5)
        if audio is None:
            continue
        wav_path = recorder.save_wav(audio)
        text = transcribe_audio(wav_path)
        os.remove(wav_path)
        if text:
            print(f"📝 Heard: \"{text}\"", flush=True)
            message_queue.put(text)
        else:
            print("⚠️  Couldn't make out any words — still listening.", flush=True)


def input_listener():
    """
    Reads stdin: an empty line toggles listening, 'quit' exits, anything
    else is typed text — UNLESS a toolbox proposal is currently awaiting
    approval, in which case the next line is routed as a yes/no answer
    instead of normal chat input. (Confirmations are typed-only for now —
    voice isn't wired to answer these yet.)
    """
    while not shutdown_event.is_set():
        try:
            line = input()
        except EOFError:
            shutdown_event.set()
            break
        stripped = line.strip()

        if awaiting_confirmation.is_set():
            confirmation_queue.put(stripped)
            continue

        if stripped.lower() == "quit":
            shutdown_event.set()
            break
        elif stripped == "":
            recorder.toggle_listening()
        else:
            message_queue.put(stripped)


threading.Thread(target=stt_worker, daemon=True).start()
threading.Thread(target=input_listener, daemon=True).start()

print(f"\n🎮 [{personality['name'].upper()} — NEURAL EDGE CORE ACTIVE]")
print("👉 Press Enter to toggle listening on/off — talk naturally once it's on.")
print("   You can also just type a message instead. Type 'quit' + Enter to exit.\n")

while not shutdown_event.is_set():
    try:
        user_message = message_queue.get(timeout=0.5)
    except queue.Empty:
        continue

    print("⏳ Processing pipeline...")
    npc_json_response = chat_with_npc(user_message)

    try:
        parsed_data = json.loads(npc_json_response)
        dialogue_text = parsed_data.get("dialogue", "...")
        current_mood = parsed_data.get("mood", "neutral")
        self_edit_requests = parsed_data.get("self_edits", [])
        toolbox_call_requests = parsed_data.get("toolbox_calls", [])
        toolbox_proposal_list = parsed_data.get("toolbox_proposals", [])
        toolbox_proposal_request = toolbox_proposal_list[0] if toolbox_proposal_list else None
        asset_request_text = parsed_data.get("asset_request", "")

        print(f"\n📦 [Engine Data]: {json.dumps(parsed_data, indent=2)}")
        print(f"🗣️  {personality['name']} ({current_mood}): \"{dialogue_text}\"\n")

        edit_results = apply_self_edits(self_edit_requests)
        for r in edit_results:
            if r["applied"]:
                print(f"🛠️  Self-edit applied: {r['action']} -> {r['value']}", flush=True)
            else:
                print(f"🚫 Self-edit rejected: {r['action']} -> {r['value']!r} ({r['reason']})", flush=True)

        toolbox_results = execute_toolbox_calls(toolbox_call_requests, CHAR_KEY, IS_ADMIN)
        for r in toolbox_results:
            if r["applied"]:
                print(f"🔧 Toolbox call: {r['function_name']}({r['args']}) — OK", flush=True)
            else:
                print(f"🚫 Toolbox call: {r['function_name']}({r['args']}) — {r['reason']}", flush=True)

        proposal_result = None
        if toolbox_proposal_request:
            proposal_result = handle_toolbox_proposal(toolbox_proposal_request, CHAR_KEY, IS_ADMIN)
            if proposal_result["applied"]:
                print(
                    f"✅ Toolbox proposal APPROVED — added to {proposal_result['target']} "
                    f"toolbox: {proposal_result['function_name']}()",
                    flush=True,
                )
            else:
                print(f"🚫 Toolbox proposal not applied: {proposal_result['reason']}", flush=True)

        # One consolidated feedback pass, covering everything attempted this turn
        sync_memory_system_message(edit_results, toolbox_results, proposal_result)

        if asset_request_text:
            print(f"📦 Asset requested by {personality['name']}: \"{asset_request_text}\"", flush=True)
            print("   (Informational only — add a matching file under assets/ yourself if you want to grant this.)", flush=True)
            log_toolbox_event({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": "asset_request",
                "character": CHAR_KEY,
                "description": asset_request_text,
            })

        recorder.pause_for_playback()  # don't let the mic hear the character through the speakers
        speak(dialogue_text, mood=current_mood)
        recorder.resume_after_playback()

    except json.JSONDecodeError as e:
        print(f"⚠️ Model produced malformed JSON: {e}", flush=True)
        log_toolbox_event({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "malformed_response",
            "character": CHAR_KEY,
            "error": str(e),
            "raw_response": npc_json_response,
        })
        fallback_line = "Sorry, I got a bit tangled up there — could you try that again?"
        print(f"🗣️  {personality['name']}: \"{fallback_line}\" (fallback — response failed to parse)\n", flush=True)
        recorder.pause_for_playback()
        speak(fallback_line, mood="neutral")
        recorder.resume_after_playback()
    except Exception as e:
        print(f"⚠️ Loop Exception: {e}", flush=True)

recorder.close()
player.close()
print("👋 Shutting down.")