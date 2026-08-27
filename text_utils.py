import re
import unicodedata


REPLACEMENTS = {
    "fu r": "für",
}


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)

    for wrong, correct in REPLACEMENTS.items():
        text = re.sub(rf"\b{re.escape(wrong)}\b", correct, text)

    text = re.sub(r"\s+", " ", text)
    return text.strip()