# -*- coding: utf-8 -*-

"""Deployment-path column tiler: chunk a tall vertical-Mongolian column image
into training-length pieces at low-ink valleys.

The CRNN is trained/validated on short single-line strips (8-28 chars, resized
to 64x1024). A whole scanned COLUMN (40-60+ chars, 2400-2900px tall) fed to the
model directly squashes glyphs well below training scale and reads garbage
(scale-control experiment). Deployment therefore tiles a column into
~training-length pieces cut at ink valleys (the whitespace between lines /
between glyph clusters, never through a glyph), decodes each tile independently,
and concatenates. This module is that tiler, factored out so validation (see
``scripts/eval_column_pipeline.py``) and any deployment consumer share ONE cut
implementation — segmentation error is part of what is being measured, not an
artifact of a validation-only stand-in. (No in-repo deployment entry point
calls this yet; deployment tooling must import ``chunk_column`` from here
rather than reimplementing it, or the L2 guarantee is void.)

torch-free (numpy only) so it is unit-testable without a model or GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Tile:
    """One chunk of a column: pixel row span ``[top, bottom)`` and the crop."""

    top: int
    bottom: int
    image: np.ndarray  # [tile_h, W] uint8, same dtype/orientation as input

    @property
    def height(self) -> int:
        return self.bottom - self.top


def row_ink_profile(a: np.ndarray) -> np.ndarray:
    """Per-row ink signal of a grayscale column array, HIGH where a row is
    mostly light paper (a good cut line), LOW where ink is present.

    Uses row-mean pixel VALUE (not variance like ``mongocr.data.ink_crop`` —
    that crops one strip to its content bbox; here we want to tell a fully
    blank inter-line gap row apart from a row with ANY glyph ink in it, which a
    grayscale mean does directly: dark ink pulls the row mean down, a blank
    paper row stays near white). Input is dark-ink-on-light-paper (matches the
    source strips / packed shard PNGs), so a HIGH mean == a good cut row.
    """
    if a.ndim != 2:
        raise ValueError(f"expected a 2-D grayscale array, got shape {a.shape}")
    return a.astype(np.float32).mean(axis=1)


def _best_cut_row(profile: np.ndarray, center: int, window: int, min_gap_from: int) -> int:
    """Row index in ``[center-window, center+window]`` (clipped to the profile
    and to ``>= min_gap_from``) with the HIGHEST ink-profile value (lightest
    row = best valley). Falls back to a forced cut at ``center`` (clipped) if
    the window is empty after clipping — "never skip a cut" beats "wait for a
    valley that never comes" for a deployment tiler that must always terminate.
    """
    n = len(profile)
    lo = max(min_gap_from, center - window, 0)
    hi = min(n - 1, center + window)
    if lo > hi:
        # window collapsed (near the end of the column, or min_gap_from already
        # past center) -> force the cut at the nearest valid row.
        return max(0, min(n - 1, center))
    seg = profile[lo:hi + 1]
    return lo + int(np.argmax(seg))


def chunk_column(
    a: np.ndarray,
    *,
    tile_target: int = 900,
    window: int = 200,
    min_tile: int = 300,
) -> list[Tile]:
    """Split a tall column array into tiles of roughly ``tile_target`` px,
    cutting at the lightest (lowest-ink) row within ``+-window`` of each target
    boundary — this is the deployment tiler; validation must call this same
    function, not a reimplementation, to measure the real segmentation error.

    Algorithm: walk down the column placing a cut every ``tile_target`` px,
    but each cut is snapped to the row of maximum ``row_ink_profile`` (i.e. the
    row with the least ink, a plausible inter-line or inter-glyph gap) within
    ``+-window`` px of that target row — never through a glyph if a lighter row
    exists nearby. If a column is shorter than ``tile_target + min_tile`` it is
    returned as a single tile (no cut is useful). The final tile always runs to
    the bottom of the column. A cut that would leave either side shorter than
    ``min_tile`` is skipped (merged into the following/preceding tile) so no
    tile is ever pathologically thin — thin tiles both destabilize the CRNN
    (starved of context) and are a common source of spuriously bad CER in a
    segmentation-inclusive eval.

    Args:
        a: ``[H, W]`` grayscale column array (dark ink on light paper).
        tile_target: nominal tile height in px (matches the ~900px-tall
            training-scale strips this model was validated on at img_h=1024
            after ink-crop/resize; see the module docstring's scale-control
            note).
        window: cut search radius in px around each target boundary.
        min_tile: never emit a tile shorter than this (px); the last real cut
            is skipped rather than producing a sliver final tile.

    Returns:
        Ordered list of ``Tile`` covering ``[0, H)`` with no gaps or overlaps.
    """
    if a.ndim != 2:
        raise ValueError(f"expected a 2-D grayscale array, got shape {a.shape}")
    h = a.shape[0]
    if tile_target <= 0:
        raise ValueError(f"tile_target must be positive, got {tile_target}")
    if min_tile > tile_target:
        raise ValueError(
            f"min_tile ({min_tile}) must not exceed tile_target ({tile_target})"
        )
    if h <= tile_target + min_tile:
        return [Tile(top=0, bottom=h, image=a)]

    profile = row_ink_profile(a)
    cuts: list[int] = []  # interior cut rows, strictly increasing, in (0, h)
    target = tile_target
    last_cut = 0
    while target < h:
        cut = _best_cut_row(profile, center=target, window=window, min_gap_from=last_cut)
        # Reject a cut that starves either the tile just closed or the
        # remaining tail below min_tile — skip it and let the loop retry at
        # the NEXT target boundary instead (the current segment simply grows).
        if cut - last_cut >= min_tile and h - cut >= min_tile:
            cuts.append(cut)
            last_cut = cut
        target += tile_target

    tiles: list[Tile] = []
    prev = 0
    for c in cuts:
        tiles.append(Tile(top=prev, bottom=c, image=a[prev:c]))
        prev = c
    tiles.append(Tile(top=prev, bottom=h, image=a[prev:h]))
    return tiles


__all__ = ["Tile", "chunk_column", "row_ink_profile"]
