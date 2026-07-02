# -*- coding: utf-8 -*-

"""Build synthetic WHOLE-COLUMN validation images (Validation L2).

The CRNN is trained/validated only on short single-line strips (8-28 chars,
resized to 64x1024, see README "Results"), but deployment reads whole vertical
COLUMNS (40-60+ chars, 2400-2900px tall) — feeding a column straight to the
model squashes glyphs well below training scale and reads garbage (a
scale-control experiment proved this). Deployment therefore TILES a column at
ink valleys, decodes each tile, and concatenates (see
``mongocr/segment.py:chunk_column`` + ``scripts/eval_column_pipeline.py``).

This script does not render new synthetic text — it reuses already-rendered,
already-labeled held-out LINE strips (the same val/test WebDataset shards L1
scores on) and vertically STACKS ``k`` consecutive strips from the same
``src_doc`` (document-level split, same convention as ``mongocr.data
.src_doc_bands``) into one tall column image with a known ground-truth label.
This keeps L2 labels exactly as trustworthy as L1's (same source strips, same
alphabet, same OOV policy) while adding the one thing L1 cannot measure:
segmentation error from tiling a long column back apart at inference time.

Label join convention: the column label is the ``k`` strip labels joined with
a SINGLE SPACE at each junction (``" ".join(labels)``). L1 line labels can
already legitimately contain internal spaces (word breaks within one line), so
a junction space is not visually or lexically distinguishable from a normal
word-internal space once concatenated — this is intentional and documented,
not an oversight: the eval pipeline (eval_column_pipeline.py) concatenates
decoded tiles WITHOUT inserting its own separator (CTC output already contains
spaces where the model reads them), so a junction is only "correct" if the
model's own decode naturally produces a space there or if a tiling cut lands
near the junction. This makes the junction-space convention the fairest
apples-to-apples GT: it does not charge the model for a synthetic artifact
(e.g. an artificial marker character) that would never appear in a real
document either.

Output is a small DIRECTORY dataset (NOT a tar/webdataset shard — L2 columns
are synthesized once, are far fewer in number than the line corpus, and are
consumed by a plain glob, not streamed): ``<idx>.png`` (grayscale column image)
+ ``<idx>.json`` (``{"text": <label>, "n_lines": k, "src_doc": <id>,
"line_texts": [...]}``).

Determinism: given the same ``--shards``, ``--alphabet``, split thresholds,
``--columns``, ``--lines-per-column``, ``--line-gap``, and ``--seed``, the
output directory is byte-for-byte reproducible (the source WebDataset shard
stream order is itself deterministic for a fixed shard glob + ``training=
False`` — no shuffle stage is wired in — and only the small "which src_doc
groups to keep / how many lines to draw from each" decision uses
``--seed``).

Usage::

    python3 -m scripts.build_column_val \
        --shards '/path/to/eval_cache/test-*.tar' \
        --alphabet alphabet.json \
        --val-threshold 434600 --test-threshold 435200 --gap 200 --split test \
        --columns 200 --lines-per-column 40 60 --line-gap 24 \
        --seed 0 --out-dir /path/to/column_val
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from mongocr.data import list_shards, src_doc_bands

_WHITE = 255


def _decode_gray(png_bytes: bytes) -> np.ndarray:
    """png bytes -> [H, W] uint8 grayscale array (no ink-crop/resize — this is
    the RAW rendered strip, matching what eval_column_pipeline will later
    ink_crop/resize per TILE, not per source strip)."""
    import io

    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(png_bytes)).convert("L"), dtype=np.uint8)


def group_consecutive_by_src_doc(records: list[dict]) -> list[list[dict]]:
    """Group a stream-ordered list of ``{"src_doc": ..., ...}`` records into
    consecutive runs sharing one ``src_doc``, preserving the ORIGINAL stream
    order both across and within groups (a plain ``itertools.groupby``, made
    explicit + testable on its own: the packer writes one document's lines
    contiguously, so a run boundary is a genuine document boundary, not an
    interleaving artifact — see scripts/pack_wds_shards.py, which walks
    meta.jsonl in file order and appends each row's (png,json) pair to the
    current tar immediately)."""
    groups: list[list[dict]] = []
    for rec in records:
        if groups and groups[-1][-1]["src_doc"] == rec["src_doc"]:
            groups[-1].append(rec)
        else:
            groups.append([rec])
    return groups


def stack_column(
    strips: list[np.ndarray], *, line_gap: int
) -> np.ndarray:
    """Vertically stack ``strips`` (each ``[h_i, w_i]`` uint8, dark ink on
    light paper) into one ``[sum(h_i) + (k-1)*line_gap, max(w_i)]`` column,
    joined by ``line_gap`` px of solid white. Narrower strips are padded to the
    column's max width, CENTERED on white (matches source strips: L-mode PNGs
    of dark ink on light paper, so the pad value is the same white=255 used
    for the inter-line gap — a padded strip's background is indistinguishable
    from the gap it sits in)."""
    if not strips:
        raise ValueError("need at least one strip to stack")
    max_w = max(s.shape[1] for s in strips)
    total_h = sum(s.shape[0] for s in strips) + line_gap * (len(strips) - 1)
    out = np.full((total_h, max_w), _WHITE, dtype=np.uint8)
    y = 0
    for i, s in enumerate(strips):
        h, w = s.shape
        x0 = (max_w - w) // 2  # center-pad narrower strips on white
        out[y:y + h, x0:x0 + w] = s
        y += h
        if i < len(strips) - 1:
            y += line_gap
    return out


def build_columns(
    groups: list[list[dict]],
    *,
    n_columns: int,
    lines_min: int,
    lines_max: int,
    line_gap: int,
    seed: int,
) -> list[dict]:
    """Deterministically pick ``n_columns`` (src_doc group, k, start offset)
    windows and materialize each into a stacked column + label. Only documents
    with ``>= lines_min`` strips are eligible (an under-length document cannot
    fill even the smallest requested column). Groups are visited round-robin
    (not "first N groups") so a run with more eligible columns than the corpus
    has documents still gets coverage across MANY documents rather than piling
    every column onto the first few — deterministic given ``seed`` via a single
    ``random.Random(seed)`` stream that decides both the visiting order and
    each column's ``k``.
    """
    eligible = [g for g in groups if len(g) >= lines_min]
    if not eligible:
        raise SystemExit(
            f"no src_doc group has >= {lines_min} strips (need --lines-per-column "
            f"min <= the shortest usable document); found {len(groups)} groups, "
            f"max length {max((len(g) for g in groups), default=0)}"
        )
    rng = random.Random(seed)
    order = list(range(len(eligible)))
    rng.shuffle(order)

    out: list[dict] = []
    gi = 0
    while len(out) < n_columns:
        group = eligible[order[gi % len(order)]]
        gi += 1
        k = rng.randint(lines_min, min(lines_max, len(group)))
        start = rng.randint(0, len(group) - k)
        window = group[start:start + k]
        strips = [_decode_gray(r["png"]) for r in window]
        col = stack_column(strips, line_gap=line_gap)
        line_texts = [r["text"] for r in window]
        out.append({
            "image": col,
            "text": " ".join(line_texts),
            "n_lines": k,
            "src_doc": window[0]["src_doc"],
            "line_texts": line_texts,
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--shards", action="append", required=True,
                    help="glob for source line-strip shard tars (repeatable); "
                         "typically the extract_eval_shards val-*.tar/test-*.tar "
                         "cache so this reuses the same held-out strips L1 scores")
    ap.add_argument("--alphabet", required=True)
    ap.add_argument("--val-threshold", type=int, required=True)
    ap.add_argument("--test-threshold", type=int, required=True)
    ap.add_argument("--gap", type=int, default=200)
    ap.add_argument("--split", choices=["train", "val", "test"], default="test",
                    help="which src_doc_bands split to draw strips from "
                         "(default: test = headline)")
    ap.add_argument("--columns", type=int, default=200, help="number of columns to build")
    ap.add_argument("--lines-per-column", type=int, nargs=2, default=[40, 60],
                    metavar=("MIN", "MAX"),
                    help="k range (inclusive); a single value repeated twice "
                         "fixes k (default: 40 60)")
    ap.add_argument("--line-gap", type=int, default=24,
                    help="white px inserted between stacked strips")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    lines_min, lines_max = args.lines_per_column
    if lines_min <= 0 or lines_max < lines_min:
        raise SystemExit(f"invalid --lines-per-column {args.lines_per_column}")

    from mongocr.alphabet import load as load_alphabet

    alpha = load_alphabet(Path(args.alphabet))
    shard_urls = list_shards(*args.shards)
    if not shard_urls:
        raise SystemExit(f"no shards matched: {args.shards}")
    is_train, is_val, is_test = src_doc_bands(args.val_threshold, args.test_threshold, args.gap)
    keep = {"train": is_train, "val": is_val, "test": is_test}[args.split]
    print(f"[build_column_val] {len(shard_urls)} source shards, split={args.split}, "
          f"alphabet={len(alpha.chars)} chars", flush=True)

    import webdataset as wds

    pipe = wds.DataPipeline(
        wds.SimpleShardList(shard_urls),
        wds.tarfile_to_samples(handler=wds.warn_and_continue),
    )
    records: list[dict] = []
    for s in pipe:
        try:
            meta = json.loads(s["json"])
            sd = int(meta["src_doc"])
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
        if not keep(sd):
            continue
        if not alpha.covers(meta["text"]):  # same OOV policy as build_pipeline/eval
            continue
        records.append({"png": s["png"], "text": meta["text"], "src_doc": sd})
    print(f"[build_column_val] {len(records):,} strips kept in {args.split} split "
          f"(stream order preserved)", flush=True)
    if not records:
        raise SystemExit("no strips kept — check split thresholds / alphabet coverage")

    groups = group_consecutive_by_src_doc(records)
    print(f"[build_column_val] grouped into {len(groups)} consecutive src_doc runs "
          f"(lengths: min={min(len(g) for g in groups)} "
          f"max={max(len(g) for g in groups)} "
          f"median={sorted(len(g) for g in groups)[len(groups)//2]})", flush=True)

    columns = build_columns(groups, n_columns=args.columns, lines_min=lines_min,
                            lines_max=lines_max, line_gap=args.line_gap, seed=args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    for idx, col in enumerate(columns):
        img = Image.fromarray(col["image"], mode="L")
        img.save(out_dir / f"{idx}.png")
        meta = {"text": col["text"], "n_lines": col["n_lines"], "src_doc": col["src_doc"],
                "line_texts": col["line_texts"]}
        (out_dir / f"{idx}.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    heights = [c["image"].shape[0] for c in columns]
    print(f"\n[build_column_val] wrote {len(columns)} columns -> {out_dir} "
          f"(height px: min={min(heights)} max={max(heights)} "
          f"mean={sum(heights)/len(heights):.0f}; n_lines: "
          f"min={min(c['n_lines'] for c in columns)} "
          f"max={max(c['n_lines'] for c in columns)})", flush=True)
    print(f"[build_column_val] index-0 spot-check: n_lines={columns[0]['n_lines']} "
          f"src_doc={columns[0]['src_doc']} image_shape={columns[0]['image'].shape} "
          f"text[:70]={columns[0]['text'][:70]!r}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
