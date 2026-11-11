"""Shared tokeniser for the sparse (BM25) and local dense (hashing) retrieval paths."""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    """a an the and or of to in on for is are was were be been by with at as it its this that
    these those from what which who whom how when where why do does did can could should would
    will shall may might must our we you your they their there here have has had not no nor
    than then so if into over about any all each per much many more most other some such only
    own same too very just also i me my he she him her them us""".split()
)


def _stem(word: str) -> str:
    """Very light suffix stripping so 'returns'/'returned' match 'return'."""
    for suffix in ("ing", "ed"):
        if word.endswith(suffix) and len(word) > len(suffix) + 3:
            return word[: -len(suffix)]
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    return [_stem(w) for w in _TOKEN_RE.findall(text.lower()) if w not in STOPWORDS]
