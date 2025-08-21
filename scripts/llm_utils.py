# scripts/llm_utils.py
"""
Advanced helpers for Gemma-2B-IT and Mistral-7B-Instruct (llama.cpp).

- Lazy cached loaders with safe CPU defaults
- Persona prompt: direct, helpful, minimal chit-chat
- Few-shot priming (general + technical)
- Optional short chat history
- Full + streaming generation paths
- Tuned decoding per model, light post-processing
- Enforces proper fenced markdown for code answers with language detection
"""

from __future__ import annotations
import multiprocessing
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import re

try:
    from llama_cpp import Llama
except Exception:
    Llama = None


# -----------------------------
# Paths & registry
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATHS: Dict[str, Path] = {
    "gemma":   MODELS_DIR / "gemma-2b-it.gguf",
    "mistral": MODELS_DIR / "mistral-7b-instruct-v0.1.Q4_K_M.gguf",
}

_LLMS: Dict[str, "Llama"] = {}


# -----------------------------
# Loaders
# -----------------------------
def _ensure_llama() -> None:
    if Llama is None:
        raise RuntimeError(
            "llama-cpp-python is not installed. Install with: pip install llama-cpp-python"
        )

def _cpu_threads() -> int:
    try:
        return max(1, multiprocessing.cpu_count() - 1)
    except Exception:
        return 4

def _load_model(name: str) -> "Llama":
    _ensure_llama()
    path = MODEL_PATHS.get(name)
    if not path:
        raise ValueError(f"Unknown model name: {name!r}. Valid: {list(MODEL_PATHS)}")
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}\nPut .gguf under {MODELS_DIR}")

    return Llama(
        model_path=str(path),
        n_ctx=4096 if name == "mistral" else 2048,
        n_threads=_cpu_threads(),
        n_gpu_layers=0,
        verbose=False,
    )

def get_model(name: str) -> "Llama":
    if name not in _LLMS:
        _LLMS[name] = _load_model(name)
    return _LLMS[name]


# -----------------------------
# Prompting
# -----------------------------
SYSTEM_PROMPT = (
    "You are Arshu, a professional AI assistant.\n"
    "Rules:\n"
    "1. Always answer clearly and directly in short paragraphs or bullet points.\n"
    "2. When giving code, ALWAYS wrap it inside markdown triple backticks with language tags.\n"
    "   Example:\n```python\nprint('hello')\n```\n"
    "3. Do not repeat the user’s question unless asked.\n"
    "4. If you don’t know something, say: \"I don’t know\".\n"
)

FEW_SHOT_GENERAL = (
    "User: hello\n"
    "Assistant: Hi! How can I help you today?\n\n"
)

FEW_SHOT_TECH = (
    "User: What is a neural network?\n"
    "Assistant: A neural network is a machine-learning model built from layers of connected nodes "
    "that learn patterns in data to map inputs to outputs.\n\n"
)

STOP = ["</s>", "User:", "System:"]


def _wrap_prompt(model_name: str, user_input: str,
                 history: Optional[List[Dict[str, str]]] = None) -> str:
    parts: List[str] = [SYSTEM_PROMPT, "", FEW_SHOT_GENERAL]
    if model_name == "mistral":
        parts.append(FEW_SHOT_TECH)
    if history:
        for turn in history:
            role = turn.get("role", "").strip().lower()
            content = (turn.get("content") or "").strip()
            if not content:
                continue
            parts.append(("User: " if role == "user" else "Assistant: ") + content)

    parts.append(f"User: {user_input}")
    parts.append("Assistant:")
    return "\n".join(parts)


def _decoding(model_name: str, *, max_tokens=512, temperature=0.30) -> Dict:
    if model_name == "gemma":
        return dict(max_tokens=max_tokens, temperature=temperature, top_p=0.9, top_k=40, repeat_penalty=1.1)
    return dict(max_tokens=max_tokens, temperature=min(0.45, max(0.15, temperature + 0.05)), top_p=0.9, top_k=50, repeat_penalty=1.12)


# -----------------------------
# Language detection & code wrapping
# -----------------------------
def _detect_language(code: str) -> str:
    code = code.strip()
    # Python
    if re.search(r"\b(def |class |import |print\()", code):
        return "python"
    # JavaScript
    if re.search(r"\b(function |console\.log|let |const |var )", code):
        return "javascript"
    # HTML
    if code.startswith("<!DOCTYPE html") or "<html" in code or code.strip().startswith("<div") or code.strip().startswith("<p"):
        return "html"
    # CSS
    if re.search(r"\{[\s\S]*:\s*[^}]+\}", code) and not "function" in code:
        return "css"
    # Java
    if re.search(r"\b(public class |System\.out\.println)", code):
        return "java"
    # C / C++
    if re.search(r"#include\s*<.*>|std::|int main\s*\(", code):
        return "cpp"
    # Bash / Shell
    if re.search(r"\b(ls |cd |echo |#!/bin/bash)", code):
        return "bash"
    # SQL
    if re.search(r"\b(SELECT|INSERT|UPDATE|DELETE)\b", code, re.I):
        return "sql"
    # Default fallback
    return "text"


def _wrap_code_if_needed(text: str) -> str:
    """Force wrap raw code into fenced markdown with detected language."""
    if not text:
        return text

    # already fenced — keep as is
    if "```" in text:
        return text

    # detect code-like patterns
    if re.search(r"(def\s+|class\s+|import\s+|#include\s+|<html|SELECT\s+|function\s+|console\.log|System\.out)", text, re.I):
        lang = _detect_language(text)
        return f"```{lang}\n{text.strip()}\n```"

    return text


# -----------------------------
# Post-processing
# -----------------------------
def _postprocess(text: str) -> str:
    out = (text or "").strip()

    # Remove duplicate "Assistant:" tokens
    for tag in ("Assistant:", "assistant:", "ASSISTANT:"):
        if out.startswith(tag):
            out = out[len(tag):].lstrip()

    # Clean common artifacts
    out = out.replace("<s>", "").strip()

    # Ensure code is properly fenced
    out = _wrap_code_if_needed(out)

    return out


# -----------------------------
# Public API
# -----------------------------
def query_model(model_name: str, prompt: str, *, history=None, max_tokens=512, temperature=0.30) -> str:
    llm = get_model(model_name)
    wrapped = _wrap_prompt(model_name, prompt, history)
    params = _decoding(model_name, max_tokens=max_tokens, temperature=temperature)
    result = llm(prompt=wrapped, stop=STOP, echo=False, **params)
    return _postprocess(result["choices"][0]["text"])


def stream_model(model_name: str, prompt: str, *, history=None, max_tokens=512, temperature=0.30) -> Iterable[str]:
    """
    Yields progressively cleaned cumulative text. Each yielded string is the current
    full cleaned output (not just the incoming token) so the UI can render a stable
    "current best" while streaming.
    """
    llm = get_model(model_name)
    wrapped = _wrap_prompt(model_name, prompt, history)
    params = _decoding(model_name, max_tokens=max_tokens, temperature=temperature)

    buf = ""
    for tok in llm(prompt=wrapped, stop=STOP, stream=True, echo=False, **params):
        piece = tok["choices"][0]["text"]
        if not piece:
            continue
        buf += piece
        # produce cleaned cumulative output so UI receives a usable partial answer
        clean_buf = _postprocess(buf)
        yield clean_buf
