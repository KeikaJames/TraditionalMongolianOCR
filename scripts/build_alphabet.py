# -*- coding: utf-8 -*-

"""Build the frozen character alphabet from a full scan of rendered labels.

Scans the ``text`` field of one or more ``meta.jsonl`` files (the exact labels
the model will be trained against), never sampled, and writes ``alphabet.json``
(char->id + sha256). A per-character count histogram is written separately as a
LOCAL QA artifact (``*.counts.json``) and is git-ignored so committed files leak
no corpus statistics.

Usage::

    python3 -m scripts.build_alphabet \
        --meta /path/to/image2text/meta.jsonl \
        --meta /path/to/image2text/hanshi/meta.jsonl \
        --out alphabet.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mongocr.alphabet import from_counts, save, scan_labels


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--meta", action="append", required=True,
                    help="meta.jsonl path (repeatable; all are scanned in full)")
    ap.add_argument("--out", default="alphabet.json")
    ap.add_argument("--counts", default="",
                    help="optional path for the per-char count QA artifact "
                         "(default: <out>.counts.json, git-ignored)")
    args = ap.parse_args()

    meta_paths = [Path(m) for m in args.meta]
    for mp in meta_paths:
        if not mp.exists():
            raise SystemExit(f"meta not found: {mp}")

    print(f"[alphabet] scanning {len(meta_paths)} meta file(s) in full ...", flush=True)
    counts = scan_labels(meta_paths)
    total = sum(counts.values())
    alpha = from_counts(counts)
    save(alpha, Path(args.out), source=",".join(map(str, meta_paths)), n_labels=total)

    counts_path = Path(args.counts) if args.counts else Path(str(args.out) + ".counts.json")
    counts_path.write_text(
        json.dumps({f"U+{ord(c):04X}": (c, n) for c, n in counts.most_common()},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[alphabet] chars={len(alpha.chars)} (+blank={alpha.blank}) "
          f"sha256={alpha.sha256[:12]} total_codepoints={total:,}", flush=True)
    print(f"[alphabet] wrote {args.out} ; counts QA -> {counts_path}", flush=True)
    # QA: show the codepoint range and a few rarest chars to eyeball for junk.
    rarest = counts.most_common()[-15:]
    print("[alphabet] 15 rarest code points (eyeball for junk):", flush=True)
    for c, n in rarest:
        print(f"    U+{ord(c):04X} {c!r}  x{n}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
