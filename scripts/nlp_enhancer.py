# scripts/nlp_enhancer.py
import re

# Optional stopword removal with safe fallback
try:
    import nltk
    from nltk.corpus import stopwords
    _STOPWORDS_READY = True
    try:
        _ = stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)
except Exception:
    _STOPWORDS_READY = False
    stopwords = None


def clean_text(text: str) -> str:
    # collapse whitespace and strip control characters
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\s+\n", "\n", text)
    return text.strip()


def enhance_text(text: str) -> str:
    if not _STOPWORDS_READY:
        return clean_text(text)
    sw = set(stopwords.words("english"))
    words = [w for w in text.split() if w.lower() not in sw]
    # Keep very short inputs intact (like "ai") to avoid over-filtering
    if len(text.split()) <= 2:
        return clean_text(text)
    return " ".join(words)
