# -*- coding: utf-8 -*-

"""OCR accuracy metrics for traditional Mongolian.

The same *visual* text can map to several different code-point sequences: free
variation selectors (FVS, U+180B-180D), the Mongolian vowel separator (MVS,
U+180E), narrow no-break space (NNBSP, U+202F) and presentation variants.
Comparing raw code points conflates genuine recognition errors with harmless
encoding/rendering differences.

Two character error rates are reported:

- **normalized CER** (primary): prediction and reference are first folded to
  nominal Mongolian Unicode (strip FVS/MVS/joiners/BOM, NNBSP -> space), then
  compared. This keeps the metric meaningful without penalizing rendering noise.
- **raw CER** (secondary): compares the unmodified code points.

Plus word accuracy (WER over whitespace tokens) and an exact line-match rate.

torch-free so it can be unit-tested without loading a model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

# Code points that are pure encoding/rendering variation and must not count as
# recognition errors.
_FVS = {0x180B, 0x180C, 0x180D, 0x180F}  # free variation selectors (incl. FVS4)
_MVS = {0x180E}  # Mongolian vowel separator
_JOINERS = {0x200C, 0x200D}  # ZWNJ / ZWJ
_BOM = {0xFEFF, 0xFFFE}
_NNBSP = 0x202F  # narrow no-break space -> regular space

_DELETE = _FVS | _MVS | _JOINERS | _BOM


def nominal_fold(text: str) -> str:
    """Fold to nominal Mongolian Unicode: drop FVS/MVS/joiners/BOM, NNBSP->space."""
    out = []
    for ch in text:
        cp = ord(ch)
        if cp in _DELETE:
            continue
        out.append(" " if cp == _NNBSP else ch)
    return "".join(out)


def nominal_normalize(texts: Sequence[str]) -> list[str]:
    return [nominal_fold(t) for t in texts]


def edit_distance(a: Sequence, b: Sequence) -> int:
    """Levenshtein edit distance (O(len(a)*len(b)) time, O(min) space)."""
    if a is b or a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def _corpus_rate(preds: Sequence[Sequence], refs: Sequence[Sequence]) -> float:
    """Micro-averaged error rate: sum(edit distance) / sum(max(len(ref), 1)).

    ``max(len(ref), 1)`` keeps pure-insertion errors against an empty reference
    reflected and avoids a 0/0 denominator.
    """
    total_dist = 0
    total_len = 0
    for p, r in zip(preds, refs):
        total_dist += edit_distance(p, r)
        total_len += max(len(r), 1)
    return total_dist / total_len if total_len else 0.0


def cer(preds: Sequence[str], refs: Sequence[str], *, normalize: bool = True) -> float:
    """Corpus character error rate (normalized to nominal Unicode by default)."""
    if normalize:
        preds = nominal_normalize(preds)
        refs = nominal_normalize(refs)
    return _corpus_rate(preds, refs)


def wer(preds: Sequence[str], refs: Sequence[str], *, normalize: bool = True) -> float:
    """Corpus word error rate over whitespace-split tokens."""
    if normalize:
        preds = nominal_normalize(preds)
        refs = nominal_normalize(refs)
    return _corpus_rate([p.split() for p in preds], [r.split() for r in refs])


@dataclass
class OCRReport:
    n: int
    norm_cer: float
    raw_cer: float
    wer: float
    line_exact: float


def ocr_report(preds: Sequence[str], refs: Sequence[str]) -> OCRReport:
    """Full OCR quality report over aligned ``preds``/``refs``."""
    preds = list(preds)
    refs = list(refs)
    if len(preds) != len(refs):
        raise ValueError(f"preds/refs length mismatch: {len(preds)} != {len(refs)}")
    norm_p = nominal_normalize(preds)
    norm_r = nominal_normalize(refs)
    norm_cer = _corpus_rate(norm_p, norm_r)
    raw_cer = _corpus_rate(preds, refs)
    wer_rate = _corpus_rate([p.split() for p in norm_p], [r.split() for r in norm_r])
    exact = sum(1 for p, r in zip(norm_p, norm_r) if p == r)
    line_exact = exact / len(preds) if preds else 0.0
    return OCRReport(
        n=len(preds),
        norm_cer=norm_cer,
        raw_cer=raw_cer,
        wer=wer_rate,
        line_exact=line_exact,
    )


__all__ = [
    "OCRReport",
    "cer",
    "edit_distance",
    "nominal_fold",
    "nominal_normalize",
    "ocr_report",
    "wer",
]
