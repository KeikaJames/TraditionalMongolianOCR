# -*- coding: utf-8 -*-

"""Frozen character alphabet for the CRNN.

The alphabet is the exact set of code points present in the rendered labels
(``meta.jsonl`` ``text`` field, which the renderer already whitespace-normalized).
It is built once by a full streaming scan of the labels (never sampled — a code
point missing from the frozen vocab can never be decoded), saved with a sha256
identity, and loaded read-only at train/eval time. ``blank`` is ``len(alphabet)``.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Alphabet:
    chars: list[str]            # sorted by code point; id == index
    stoi: dict[str, int]
    sha256: str

    @property
    def blank(self) -> int:
        return len(self.chars)

    @property
    def n_classes(self) -> int:
        return len(self.chars) + 1

    def encode(self, text: str) -> list[int]:
        """Map a label to character ids, dropping any code point not in vocab."""
        return [self.stoi[c] for c in text if c in self.stoi]


def _hash(chars: list[str]) -> str:
    return hashlib.sha256("".join(chars).encode("utf-8")).hexdigest()


def scan_labels(meta_paths: list[Path]) -> Counter:
    """Full scan of ``text`` fields across meta.jsonl files -> per-char Counter."""
    counts: Counter = Counter()
    for mp in meta_paths:
        with open(mp, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    text = json.loads(line)["text"]
                except (json.JSONDecodeError, KeyError):
                    continue
                counts.update(text)
    return counts


def from_counts(counts: Counter) -> Alphabet:
    chars = sorted(counts, key=ord)
    stoi = {c: i for i, c in enumerate(chars)}
    return Alphabet(chars=chars, stoi=stoi, sha256=_hash(chars))


def save(alpha: Alphabet, path: Path, *, source: str = "", n_labels: int = 0) -> None:
    """Write the frozen vocab. Counts/histogram are NOT written here (kept as a
    separate local QA artifact) so the committed file leaks no corpus statistics."""
    path = Path(path)
    path.write_text(
        json.dumps(
            {
                "chars": alpha.chars,
                "sha256": alpha.sha256,
                "n_chars": len(alpha.chars),
                "blank": alpha.blank,
                "source": source,
                "n_labels_scanned": n_labels,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def load(path: Path) -> Alphabet:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    chars = obj["chars"]
    alpha = Alphabet(chars=chars, stoi={c: i for i, c in enumerate(chars)},
                     sha256=_hash(chars))
    if obj.get("sha256") and obj["sha256"] != alpha.sha256:
        raise ValueError(
            f"alphabet sha256 mismatch: file={obj['sha256']} computed={alpha.sha256}"
        )
    return alpha


__all__ = ["Alphabet", "scan_labels", "from_counts", "save", "load"]
