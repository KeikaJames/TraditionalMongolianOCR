# -*- coding: utf-8 -*-

"""Streaming WebDataset pipeline for line-strip OCR.

The corpus is hundreds of millions of (png, json) line strips packed into
WebDataset tar shards. Loading meta into RAM or stat-ing every image (the old
map-style path) does not scale; this module streams shards instead:

- shard list is globbed (shard indices are sparse, never assume contiguous);
- ``split_by_node`` + ``split_by_worker`` partition shards so no sample is seen
  twice across workers/ranks (asserted by the verification script);
- corrupt/truncated shards are skipped (``warn_and_continue``), not fatal;
- document-level train/eval split is by the json ``src_doc`` value (NOT shard
  index — the packer chunks by line count, so a src_doc can span shards);
- the frozen alphabet maps label text -> char ids.

Preprocessing (grayscale, variance-based ink crop, resize, invert) is identical
to the original from-scratch trainer.
"""

from __future__ import annotations

import glob as _glob
import io
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image

from .alphabet import Alphabet


# --------------------------------------------------------------------------- #
# Preprocessing (identical logic to the original trainer)
# --------------------------------------------------------------------------- #


def _to_array(img):
    return np.asarray(img, dtype=np.uint8)


def ink_crop(img, pad: int = 8):
    """Crop a grayscale strip to its ink bounding box (+ ``pad`` px).

    Uses per-row/col pixel VARIANCE (not a darkness bbox): black glyphs on light
    paper give HIGH variance; degraded/wrinkled blank paper stays LOW variance
    even when darkened, so a darkness bbox is defeated by noise. Cropping to the
    ink extent makes glyphs fill the frame height so the CTC frame axis carries
    signal instead of mostly-blank background. Falls back to the original image
    if no ink is found.
    """
    a = np.asarray(img, dtype=np.float32)
    if a.ndim != 2 or min(a.shape) < 4:
        return img
    row_sig = a.std(axis=1)
    col_sig = a.std(axis=0)

    def _span(d, frac=0.30):
        k = max(1, len(d) // 150)
        if k > 1:
            d = np.convolve(d, np.ones(k) / k, mode="same")
        m = float(d.max())
        if m <= 1e-6:
            return 0, len(d) - 1
        on = np.nonzero(d > frac * m)[0]
        return (int(on[0]), int(on[-1])) if on.size else (0, len(d) - 1)

    r0, r1 = _span(row_sig)
    c0, c1 = _span(col_sig)
    top = max(0, r0 - pad)
    bottom = min(a.shape[0], r1 + 1 + pad)
    left = max(0, c0 - pad)
    right = min(a.shape[1], c1 + 1 + pad)
    return Image.fromarray(np.asarray(img)[top:bottom, left:right])


def decode_image(png_bytes: bytes, img_h: int, img_w: int) -> torch.Tensor:
    """png bytes -> [1, H, W] float tensor, ink high (1) / paper low (0)."""
    img = Image.open(io.BytesIO(png_bytes)).convert("L")
    img = ink_crop(img)
    img = img.resize((img_w, img_h), Image.BILINEAR)
    x = torch.from_numpy(_to_array(img)).float() / 255.0
    x = 1.0 - x          # ink high, paper low
    return x.unsqueeze(0)  # [1, H, W]


def collate(batch):
    """Pad targets to a rectangle; CTC ignores past ``target_lengths``."""
    imgs = torch.stack([b[0] for b in batch], dim=0)  # [B,1,H,W]
    targets = [b[1] for b in batch]
    lengths = torch.tensor([t.numel() for t in targets], dtype=torch.long)
    u_max = int(lengths.max().item()) if len(targets) else 1
    padded = torch.zeros(len(targets), max(u_max, 1), dtype=torch.long)
    for i, t in enumerate(targets):
        padded[i, : t.numel()] = t
    return imgs, padded, lengths


# --------------------------------------------------------------------------- #
# Shard discovery + document-level split
# --------------------------------------------------------------------------- #


def list_shards(*patterns: str) -> list[str]:
    """Glob shard tars (sparse indices -> sort the actual files, no contiguity)."""
    urls: list[str] = []
    for pat in patterns:
        urls.extend(_glob.glob(pat))
    return sorted(urls)


def src_doc_split(threshold: int, gap: int = 200) -> tuple[Callable, Callable]:
    """Return (is_train, is_eval) predicates over ``src_doc``.

    eval  : src_doc >= threshold
    train : src_doc <  threshold - gap   (a gap band is dropped from both so a
            src_doc straddling a shard boundary can never leak across the split)
    """
    def is_train(src_doc: int) -> bool:
        return src_doc < threshold - gap

    def is_eval(src_doc: int) -> bool:
        return src_doc >= threshold

    return is_train, is_eval


# --------------------------------------------------------------------------- #
# WebDataset pipeline
# --------------------------------------------------------------------------- #


def build_pipeline(
    shard_urls: list[str],
    alpha: Alphabet,
    *,
    img_h: int,
    img_w: int,
    batch_size: int,
    keep_src_doc: Callable[[int], bool] | None = None,
    training: bool = True,
    shard_shuffle: int = 200,
    sample_shuffle: int = 4000,
):
    """Build a ``wds.DataPipeline`` yielding ``(imgs, padded, lengths)`` batches.

    ``keep_src_doc`` filters samples by their json ``src_doc`` (train/eval split).
    Worker/node splitting guarantees each shard is consumed by exactly one
    worker, so samples are never duplicated across the pool.
    """
    import webdataset as wds

    def parse(sample):
        meta = json.loads(sample["json"])
        return {"png": sample["png"], "meta": meta, "key": sample["__key__"]}

    def keep(sample):
        meta = sample["meta"]
        if keep_src_doc is not None and not keep_src_doc(int(meta["src_doc"])):
            return False
        # Drop lines whose label has an out-of-vocab char: the rendered image
        # still contains that glyph, so we cannot supervise it — keeping the line
        # (with the char silently dropped from the target) would teach the model
        # to skip glyphs. Curate the alphabet, then drop uncovered lines.
        return alpha.covers(meta["text"])

    def to_tensors(sample):
        x = decode_image(sample["png"], img_h, img_w)
        target = torch.tensor(alpha.encode(sample["meta"]["text"]), dtype=torch.long)
        return x, target

    stages = [
        wds.SimpleShardList(shard_urls),
        wds.split_by_node,
        wds.split_by_worker,
    ]
    if training and shard_shuffle:
        stages.append(wds.shuffle(shard_shuffle))   # shard-order shuffle
    stages.append(wds.tarfile_to_samples(handler=wds.warn_and_continue))
    stages.append(wds.map(parse, handler=wds.warn_and_continue))
    stages.append(wds.select(keep))
    if training and sample_shuffle:
        stages.append(wds.shuffle(sample_shuffle))  # sample-buffer shuffle
    stages.append(wds.map(to_tensors, handler=wds.warn_and_continue))
    stages.append(wds.batched(batch_size, collation_fn=collate, partial=not training))
    return wds.DataPipeline(*stages)


def key_pipeline(shard_urls: list[str], keep_src_doc: Callable[[int], bool] | None = None):
    """A ``wds.DataPipeline`` yielding sample keys, runnable under ``WebLoader``
    with ``num_workers >= 2`` so ``split_by_worker`` is actually exercised — used
    to assert no sample is emitted by more than one worker (GATE-D)."""
    import webdataset as wds

    def keep(sample):
        if keep_src_doc is None:
            return True
        return keep_src_doc(int(json.loads(sample["json"])["src_doc"]))

    return wds.DataPipeline(
        wds.SimpleShardList(shard_urls),
        wds.split_by_node,
        wds.split_by_worker,
        wds.tarfile_to_samples(handler=wds.warn_and_continue),
        wds.select(keep),
        wds.map(lambda s: s["__key__"]),
    )


def keys_of(
    shard_urls: list[str], keep_src_doc: Callable[[int], bool] | None = None, limit: int | None = None
):
    """Yield (key, src_doc, text, font) without decoding images — for verification
    (no-duplication assert, src_doc disjointness, font coverage, alphabet check)."""
    import webdataset as wds

    pipe = wds.DataPipeline(
        wds.SimpleShardList(shard_urls),
        wds.split_by_node,
        wds.split_by_worker,
        wds.tarfile_to_samples(handler=wds.warn_and_continue),
    )
    n = 0
    for sample in pipe:
        meta = json.loads(sample["json"])
        sd = int(meta["src_doc"])
        if keep_src_doc is not None and not keep_src_doc(sd):
            continue
        yield sample["__key__"], sd, meta["text"], meta.get("font", "")
        n += 1
        if limit is not None and n >= limit:
            return


__all__ = [
    "build_pipeline",
    "collate",
    "decode_image",
    "ink_crop",
    "key_pipeline",
    "keys_of",
    "list_shards",
    "src_doc_split",
]
