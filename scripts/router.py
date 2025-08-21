# scripts/router.py
import re
from pathlib import Path
from typing import Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KEYWORD_FILE = PROJECT_ROOT / "tech_keywords.txt"

# In-memory state
_TECH_KEYWORDS: Set[str] = set()
_SINGLE_WORDS_REGEX = None

# Base keywords to avoid empty routing (covers 'ai', etc.)
BASE_TECH_KEYWORDS: Set[str] = {
    "ai", "artificial intelligence", "machine learning", "deep learning",
    "neural", "transformer", "llm", "nlp", "ner", "gpu", "cpu",
    "python", "pytorch", "numpy", "pandas", "sql", "database",
    "api", "algorithm", "computer vision", "classification", "regression",
}


def _compile_regex_from_keywords(keywords: Set[str]):
    singles = [re.escape(k) for k in keywords if " " not in k and "-" not in k and len(k) > 2]
    if not singles:
        return None
    pattern = r"\b(" + "|".join(sorted(singles, key=len, reverse=True)) + r")\b"
    return re.compile(pattern, flags=re.IGNORECASE)


def load_keywords(filepath: Path = KEYWORD_FILE) -> Set[str]:
    if not filepath.exists():
        return set(BASE_TECH_KEYWORDS)
    with filepath.open("r", encoding="utf-8") as f:
        file_kws = {line.strip().lower() for line in f if line.strip()}
    return set(BASE_TECH_KEYWORDS).union(file_kws)


def refresh_keywords():
    """Reload keywords and rebuild the regex cache."""
    global _TECH_KEYWORDS, _SINGLE_WORDS_REGEX
    _TECH_KEYWORDS = load_keywords()
    _SINGLE_WORDS_REGEX = _compile_regex_from_keywords(_TECH_KEYWORDS)


# Initialize on import
refresh_keywords()


def _is_greeting(text: str) -> bool:
    t = text.strip().lower()
    return any(t.startswith(x) for x in ("hi", "hello", "hey", "sup", "good morning", "good evening"))


def is_technical(text: str, min_hits: int = 1) -> bool:
    """Heuristic: 1+ keyword hits makes it technical (was 2; caused misses like 'what is ai')."""
    if not text:
        return False

    # greetings should stay casual
    if _is_greeting(text):
        return False

    lowered = text.lower()
    hits = 0

    if _SINGLE_WORDS_REGEX:
        hits += len(_SINGLE_WORDS_REGEX.findall(lowered))

    for kw in _TECH_KEYWORDS:
        if (" " in kw or "-" in kw) and kw in lowered:
            hits += 1

    return hits >= min_hits


def choose_model(user_input: str) -> str:
    return "mistral" if is_technical(user_input) else "gemma"
