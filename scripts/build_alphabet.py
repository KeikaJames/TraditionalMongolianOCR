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
    ap.add_argument("--min-count", type=int, default=10,
                    help="keep only characters occurring >= this many times; "
                         "lines containing a dropped char are removed in training")
    ap.add_argument("--counts", default="",
                    help="optional path for the per-char count QA artifact "
                         "(default: <out>.counts.json, git-ignored)")
    ap.add_argument("--counts-in", default="",
                    help="reuse an existing *.counts.json instead of rescanning")
    args = ap.parse_args()

    if args.counts_in:
        from collections import Counter
        raw = json.loads(Path(args.counts_in).read_text(encoding="utf-8"))
        counts = Counter({v[0]: v[1] for v in raw.values()})
        print(f"[alphabet] reusing counts from {args.counts_in}", flush=True)
    else:
        meta_paths = [Path(m) for m in args.meta]
        for mp in meta_paths:
            if not mp.exists():
                raise SystemExit(f"meta not found: {mp}")
        print(f"[alphabet] scanning {len(meta_paths)} meta file(s) in full ...", flush=True)
        counts = scan_labels(meta_paths)

    total = sum(counts.values())
    alpha = from_counts(counts, min_count=args.min_count)
    dropped = sorted((c for c, n in counts.items() if n < args.min_count),
                     key=lambda c: -counts[c])
    dropped_occ = sum(counts[c] for c in dropped)
    save(alpha, Path(args.out), source=",".join(args.meta), n_labels=total,
         min_count=args.min_count)

    counts_path = Path(args.counts) if args.counts else Path(str(args.out) + ".counts.json")
    counts_path.write_text(
        json.dumps({f"U+{ord(c):04X}": (c, n) for c, n in counts.most_common()},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[alphabet] kept={len(alpha.chars)} (+blank={alpha.blank}) "
          f"sha256={alpha.sha256[:12]} min_count={args.min_count}", flush=True)
    print(f"[alphabet] dropped {len(dropped)} chars as noise "
          f"({dropped_occ:,}/{total:,} = {100*dropped_occ/max(total,1):.5f}% of codepoints)",
          flush=True)
    print(f"[alphabet] wrote {args.out} ; counts QA -> {counts_path}", flush=True)
    print("[alphabet] sample of dropped (kept out of vocab):", flush=True)
    for c in dropped[:20]:
        print(f"    U+{ord(c):04X} {c!r}  x{counts[c]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
