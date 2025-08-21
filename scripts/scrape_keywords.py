# scripts/scrape_keywords.py
import re
import time
from pathlib import Path
from typing import Iterable, Set, Optional
import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KEYWORD_FILE = PROJECT_ROOT / "tech_keywords.txt"

DEFAULT_PAGES = [
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://en.wikipedia.org/wiki/Deep_learning",
    "https://en.wikipedia.org/wiki/Natural_language_processing",
    "https://en.wikipedia.org/wiki/Computer_vision",
    "https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)",
    "https://en.wikipedia.org/wiki/Algorithm",
    "https://en.wikipedia.org/wiki/Database",
    "https://en.wikipedia.org/wiki/Cloud_computing",
    "https://en.wikipedia.org/wiki/Blockchain",
]

HEADERS = {"User-Agent": "Mozilla/5.0 (HybridLLMKeyScraper/1.0)"}


def _clean_token(t: str) -> Optional[str]:
    t = re.sub(r"\s+", " ", t).strip().lower()
    if not t or len(t) < 3:
        return None
    if t.startswith("help") or t in {"edit", "talk", "main page", "contents"}:
        return None
    return t


def _extract_keywords_from_html(html: str) -> Set[str]:
    soup = BeautifulSoup(html, "html.parser")
    keywords: Set[str] = set()

    for tag in soup.find_all(["h1", "h2", "h3", "b", "strong"]):
        txt = tag.get_text(" ", strip=True)
        tok = _clean_token(txt)
        if tok:
            keywords.add(tok)

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not href.startswith("/wiki/"):
            continue
        txt = a.get_text(" ", strip=True)
        tok = _clean_token(txt)
        if tok:
            keywords.add(tok)

    return keywords


def scrape_page(url: str) -> Set[str]:
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return _extract_keywords_from_html(r.text)


def scrape_many(urls: Iterable[str], sleep_sec: float = 0.4) -> Set[str]:
    all_kws: Set[str] = set()
    for url in urls:
        try:
            kws = scrape_page(url)
            all_kws.update(kws)
            print(f"✓ {url} -> +{len(kws)} (total {len(all_kws)})")
        except Exception as e:
            print(f"✗ {url} -> {e}")
        time.sleep(sleep_sec)
    return all_kws


def merge_into_file(keywords: Set[str], path: Path = KEYWORD_FILE):
    existing = set()
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            existing = {line.strip().lower() for line in f if line.strip()}
    merged = sorted(existing.union(keywords))
    with path.open("w", encoding="utf-8") as f:
        for k in merged:
            f.write(k + "\n")
    print(f"✅ Saved {len(merged)} total keywords to {path}")


if __name__ == "__main__":
    kws = scrape_many(DEFAULT_PAGES)
    merge_into_file(kws)
