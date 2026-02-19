import re
import unicodedata

# Arabic harakat range + dagger alif
DIACRITICS_RE = re.compile(r"[\u064B-\u065F\u0670]")
TATWEEL_RE = re.compile(r"\u0640")


def has_diacritics(text: str) -> bool:
    """Check if text contains Arabic diacritics."""
    return bool(DIACRITICS_RE.search(text or ""))


def normalize_ar_for_search(text: str) -> str:
    """
    Normalize Arabic text for search. remove tatweel,remove diacritics,normalize common letter variants
    
    """
    if not text:
        return ""

    t = unicodedata.normalize("NFC", text)
    t = TATWEEL_RE.sub("", t)
    t = DIACRITICS_RE.sub("", t)

    # normalize common letter variants
    t = t.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    t = t.replace("ى", "ي")
    t = t.replace("ة", "ه")

    t = re.sub(r"\s+", " ", t).strip()
    return t

