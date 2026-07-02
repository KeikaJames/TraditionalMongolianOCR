# -*- coding: utf-8 -*-

"""Validation L2: score the DEPLOYMENT tiling pipeline end-to-end on synthetic
whole-column images with known labels.

L1 (``scripts/eval_crnn.py``) measures the model on short single-line strips —
exactly the training distribution, and therefore blind to the one failure mode
that only shows up on real deployment input: a whole scanned COLUMN (40-60+
chars, 2400-2900px tall) fed straight to the model squashes glyphs below
training scale and reads garbage (scale-control experiment). Deployment copes
by TILING a column into ~training-length pieces at ink valleys, decoding each
tile, and concatenating — this script runs and scores THAT exact path,
including its segmentation error, not a shortcut:

  1. chunk each column image along HEIGHT via ``mongocr.segment.chunk_column``
     (the same function deployment calls — not a reimplementation);
  2. each tile -> ``mongocr.data.ink_crop`` -> resize to (img_h, img_w) ->
     CRNN greedy decode (identical preprocessing to line-level eval);
  3. concatenate the tile decodes IN ORDER, NO SEPARATOR inserted. This is
     deliberate, not an oversight: CTC output already contains space
     characters wherever the model itself reads a word break, so an inserted
     separator would double up on real spaces. Expected size of the junction
     effect: ``chunk_column`` cuts at low-ink valleys — which ARE the
     inter-line gaps — and ``ink_crop`` then trims the whitespace on both
     sides of each cut, so a missing junction space is roughly ONE DELETION
     PER CUT by construction (the modal outcome, not a rare edge case). It is
     a genuine, measurable pipeline error, identical for every checkpoint
     scored here, and exactly what L2 exists to catch (see mongocr/segment.py
     + scripts/build_column_val.py's docstrings for the label-join convention);
  4. score the concatenated decode against the column's ground-truth label
     with ``mongocr.metrics.ocr_report`` (nominal-fold CER/WER, same yardstick
     as L1).

Loads the checkpoint via ``mongocr.model.resolve_arch_from_ckpt`` (works for
the legacy production checkpoint and all 4 width_collapse ablation variants).

CAVEAT (same spirit as the README's L1 caveat, restated because it is easy to
misread this number as directly comparable to L1's): this CER is NOT a
recognizer-only number. By construction it also charges every segmentation
error the deployment tiler makes (a cut through a glyph, a cut that merges or
splits a word at a junction) — that is the point of L2, not a bug in it. A
regression here can come from the tiler, the recognizer, or their interaction;
do not attribute a delta to "the model got worse" without also checking
n_tiles / tile-boundary behavior.

Usage::

    python3 -m scripts.eval_column_pipeline \
        --columns-dir /path/to/column_val \
        --alphabet alphabet.json --ckpt crnn.pt \
        --tile-target 900 --tile-window 200 --min-tile 300
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch

from mongocr.alphabet import load as load_alphabet
from mongocr.data import ink_crop
from mongocr.metrics import ocr_report
from mongocr.model import CRNN, WIDTH_COLLAPSE_MODES, greedy_decode, resolve_arch_from_ckpt
from mongocr.segment import chunk_column


def load_columns(columns_dir: Path) -> list[dict]:
    """Load the ``<idx>.png``/``<idx>.json`` directory dataset written by
    ``scripts/build_column_val.py``, sorted by numeric index (NOT lexicographic
    string sort — "10.png" must not sort before "2.png", or index-0 in the
    printed spot-check would silently stop meaning "the first column built")."""
    from PIL import Image

    pairs = []
    for jf in columns_dir.glob("*.json"):
        idx = jf.stem
        pf = columns_dir / f"{idx}.png"
        if not pf.exists():
            continue
        pairs.append((int(idx), pf, jf))
    pairs.sort(key=lambda t: t[0])
    if not pairs:
        raise SystemExit(f"no <idx>.png/<idx>.json pairs found under {columns_dir}")

    out = []
    for idx, pf, jf in pairs:
        meta = json.loads(jf.read_text(encoding="utf-8"))
        img = np.asarray(Image.open(pf).convert("L"), dtype=np.uint8)
        out.append({"idx": idx, "image": img, "text": meta["text"],
                    "n_lines": meta.get("n_lines"), "src_doc": meta.get("src_doc")})
    return out


def tile_to_tensor(tile_img: np.ndarray, img_h: int, img_w: int) -> torch.Tensor:
    """A tiled column slice -> [1, H, W] float tensor, identical preprocessing
    to line-level eval: ink_crop (ink-fills-frame) then resize, ink high /
    paper low. ``mongocr.data.decode_image`` takes PNG bytes; tiles here are
    already-decoded in-memory arrays (cut out of one big column image, never
    written to disk), so this re-implements the same two preprocessing calls
    directly over the array instead of round-tripping through PNG encode."""
    from PIL import Image

    img = Image.fromarray(tile_img, mode="L")
    img = ink_crop(img)
    img = img.resize((img_w, img_h), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).float() / 255.0
    x = 1.0 - x
    return x.unsqueeze(0)


def decode_column(
    model: CRNN, alpha, column: np.ndarray, *, img_h: int, img_w: int, device,
    tile_target: int, tile_window: int, min_tile: int,
) -> tuple[str, int, list[int]]:
    """Chunk one column with the deployment tiler, decode each tile, and
    concatenate IN ORDER with no separator (see module docstring). Returns
    ``(decoded_text, n_tiles, tile_heights)``."""
    tiles = chunk_column(column, tile_target=tile_target, window=tile_window,
                         min_tile=min_tile)
    batch = torch.stack(
        [tile_to_tensor(t.image, img_h, img_w) for t in tiles], dim=0
    ).to(device)
    with torch.no_grad():
        log_probs = model(batch)
    decoded = greedy_decode(log_probs.float().cpu(), alpha.blank, alpha.chars)
    return "".join(decoded), len(tiles), [t.height for t in tiles]


def _pct(values: list[float], p: float) -> float:
    """Nearest-rank percentile (no numpy/scipy dependency for a single-shot
    print; ``values`` is small — one entry per column, never streamed)."""
    if not values:
        return 0.0
    s = sorted(values)
    k = min(len(s) - 1, max(0, round(p / 100 * (len(s) - 1))))
    return s[k]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--columns-dir", required=True,
                    help="directory of <idx>.png/<idx>.json from build_column_val.py")
    ap.add_argument("--alphabet", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--img-h", type=int, default=1024)
    ap.add_argument("--img-w", type=int, default=64)
    ap.add_argument("--lstm-hidden", type=int, default=384,
                    help="overridden by the checkpoint's saved --lstm-hidden when "
                         "present (older checkpoints predating this arg use this "
                         "flag's value)")
    ap.add_argument("--width-collapse", default=None, choices=[*WIDTH_COLLAPSE_MODES, None],
                    help="overridden by the checkpoint's saved --width-collapse "
                         "when present; only needed as a fallback for checkpoints "
                         "saved before this flag existed (default: mean)")
    ap.add_argument("--tile-target", type=int, default=900,
                    help="deployment tiler nominal tile height in px")
    ap.add_argument("--tile-window", type=int, default=200,
                    help="deployment tiler cut-search radius in px")
    ap.add_argument("--min-tile", type=int, default=300,
                    help="deployment tiler minimum tile height in px")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--show", type=int, default=10,
                    help="index-0 + this many strided samples printed as a spot-check")
    args = ap.parse_args()

    device = (torch.device("cuda") if (args.device in ("auto", "cuda") and torch.cuda.is_available())
              else torch.device("mps") if (args.device in ("auto", "mps")
                                           and torch.backends.mps.is_available())
              else torch.device("cpu"))
    alpha = load_alphabet(Path(args.alphabet))
    columns = load_columns(Path(args.columns_dir))
    print(f"[eval_column] {len(columns)} columns loaded from {args.columns_dir}", flush=True)

    ckpt = torch.load(args.ckpt, map_location=device)
    # Same legacy/ablation-variant resolution as eval_crnn.py / eval_confusion.py
    # — see mongocr.model.resolve_arch_from_ckpt for the fallback reasoning.
    lstm_hidden, width_collapse, img_w = resolve_arch_from_ckpt(
        ckpt, cli_lstm_hidden=args.lstm_hidden,
        cli_width_collapse=args.width_collapse, cli_img_w=args.img_w)
    print(f"[eval_column] arch: lstm_hidden={lstm_hidden} width_collapse={width_collapse} "
          f"img_w={img_w} (from checkpoint args: "
          f"{'yes' if ckpt.get('args') else 'no, using CLI/defaults'})", flush=True)

    model = CRNN(alpha.n_classes, lstm_hidden=lstm_hidden,
                 width_collapse=width_collapse, img_w=img_w).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[eval_column] params={n_params/1e6:.2f}M ({n_params:,})", flush=True)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    if ckpt.get("alphabet") and ckpt["alphabet"] != alpha.chars:
        raise SystemExit("checkpoint alphabet != provided alphabet.json")

    preds, refs, n_tiles_list, per_col_cer = [], [], [], []
    tile_heights_all: list[int] = []
    for col in columns:
        pred, n_tiles, tile_heights = decode_column(
            model, alpha, col["image"], img_h=args.img_h, img_w=img_w, device=device,
            tile_target=args.tile_target, tile_window=args.tile_window,
            min_tile=args.min_tile)
        preds.append(pred)
        refs.append(col["text"])
        n_tiles_list.append(n_tiles)
        tile_heights_all.extend(tile_heights)
        per_col_cer.append(ocr_report([pred], [col["text"]]).norm_cer)

    rep = ocr_report(preds, refs)
    print(f"\n[eval_column] HEADLINE n={rep.n} norm_CER={rep.norm_cer:.4f} "
          f"raw_CER={rep.raw_cer:.4f} WER={rep.wer:.4f} "
          f"line_exact={rep.line_exact:.4f}", flush=True)
    print("[eval_column] CAVEAT: this number includes SEGMENTATION error BY "
          "DESIGN (the deployment tiler's cuts are part of the measured path) "
          "— it is not directly comparable to L1's line-level CER; see the "
          "module docstring.", flush=True)

    print(f"\n[eval_column] per-column norm_CER distribution: "
          f"mean={statistics.fmean(per_col_cer):.4f} "
          f"median={statistics.median(per_col_cer):.4f} "
          f"p90={_pct(per_col_cer, 90):.4f} "
          f"min={min(per_col_cer):.4f} max={max(per_col_cer):.4f}", flush=True)
    print(f"[eval_column] n_tiles per column: mean={statistics.fmean(n_tiles_list):.1f} "
          f"median={statistics.median(n_tiles_list):.1f} "
          f"min={min(n_tiles_list)} max={max(n_tiles_list)}", flush=True)
    print(f"[eval_column] tile height (px) across all tiles: "
          f"mean={statistics.fmean(tile_heights_all):.0f} "
          f"min={min(tile_heights_all)} max={max(tile_heights_all)} "
          f"(target={args.tile_target} min_tile={args.min_tile})", flush=True)

    print("\n[eval_column] index-0 + samples (decoded text vs GT) — trust decoded "
          "text only, not any confidence proxy:", flush=True)
    step = max(1, len(preds) // max(1, args.show))
    for i in range(0, len(preds), step):
        print(f"  [{i}] n_tiles={n_tiles_list[i]} norm_CER={per_col_cer[i]:.4f}", flush=True)
        print(f"       GT : {refs[i][:70]}", flush=True)
        print(f"       OUT: {preds[i][:70]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
