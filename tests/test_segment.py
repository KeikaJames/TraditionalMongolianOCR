# -*- coding: utf-8 -*-
import unittest

import numpy as np

from mongocr.segment import Tile, chunk_column, row_ink_profile


def _bars_column(bar_h=150, gap_h=60, n_bars=5, w=100):
    """Synthetic column: n_bars black bars (ink) separated by white gaps —
    deterministic ground truth for where a "good cut" (white gap row) is."""
    rows = []
    for i in range(n_bars):
        rows.append(np.full((bar_h, w), 20, dtype=np.uint8))  # dark bar
        if i < n_bars - 1:
            rows.append(np.full((gap_h, w), 250, dtype=np.uint8))  # white gap
    return np.concatenate(rows, axis=0), bar_h, gap_h, n_bars


class TestRowInkProfile(unittest.TestCase):
    def test_shape_and_high_on_white_low_on_ink(self):
        col, bar_h, gap_h, _n = _bars_column()
        prof = row_ink_profile(col)
        self.assertEqual(prof.shape, (col.shape[0],))
        self.assertGreater(prof[bar_h + gap_h // 2], prof[bar_h // 2])  # gap > bar

    def test_rejects_non_2d(self):
        with self.assertRaises(ValueError):
            row_ink_profile(np.zeros((4, 4, 3), dtype=np.uint8))


class TestChunkColumn(unittest.TestCase):
    def test_short_column_returns_single_tile(self):
        col = np.full((500, 20), 240, dtype=np.uint8)
        tiles = chunk_column(col, tile_target=900, window=200, min_tile=300)
        self.assertEqual(len(tiles), 1)
        self.assertEqual((tiles[0].top, tiles[0].bottom), (0, 500))
        self.assertTrue(np.array_equal(tiles[0].image, col))

    def test_tiles_are_contiguous_and_cover_whole_column(self):
        col, *_ = _bars_column(bar_h=150, gap_h=60, n_bars=5)  # h=990
        tiles = chunk_column(col, tile_target=300, window=150, min_tile=100)
        self.assertGreater(len(tiles), 1)
        self.assertEqual(tiles[0].top, 0)
        self.assertEqual(tiles[-1].bottom, col.shape[0])
        for a, b in zip(tiles, tiles[1:]):
            self.assertEqual(a.bottom, b.top)  # no gap, no overlap

    def test_cuts_land_in_white_gaps_not_through_bars(self):
        # Bars at [0,150) [210,360) [420,570) [630,780) [840,990); gaps at
        # [150,210) [360,420) [570,630) [780,840). A target of 300 with a wide
        # +-150 window should snap into a nearby gap, not a bar interior.
        col, bar_h, gap_h, n_bars = _bars_column(bar_h=150, gap_h=60, n_bars=5)
        tiles = chunk_column(col, tile_target=300, window=150, min_tile=50)
        gap_spans = []
        y = 0
        for i in range(n_bars):
            y += bar_h
            if i < n_bars - 1:
                gap_spans.append((y, y + gap_h))
                y += gap_h
        for t in tiles[:-1]:  # every interior cut (= every tile's bottom but the last)
            cut = t.bottom
            in_a_gap = any(g0 <= cut <= g1 for g0, g1 in gap_spans)
            self.assertTrue(in_a_gap, f"cut at {cut} not inside any gap {gap_spans}")

    def test_no_tile_shorter_than_min_tile_except_when_column_itself_is_shorter(self):
        col, *_ = _bars_column(bar_h=150, gap_h=60, n_bars=5)  # h=990
        min_tile = 200
        tiles = chunk_column(col, tile_target=300, window=150, min_tile=min_tile)
        for t in tiles:
            self.assertGreaterEqual(t.height, min_tile)

    def test_forced_cut_when_no_valley_in_window(self):
        # Uniform dark column: no valley anywhere, but the tiler must still
        # terminate and produce contiguous tiles covering the whole height
        # (never hang / never raise waiting for a valley that doesn't exist).
        col = np.full((2000, 30), 20, dtype=np.uint8)
        tiles = chunk_column(col, tile_target=500, window=100, min_tile=200)
        self.assertEqual(tiles[0].top, 0)
        self.assertEqual(tiles[-1].bottom, 2000)
        for a, b in zip(tiles, tiles[1:]):
            self.assertEqual(a.bottom, b.top)

    def test_single_tall_column_produces_multiple_tiles_at_training_scale(self):
        # A realistic deployment-scale column (2400-2900px, per the task spec)
        # with default tile_target=900 must yield multiple tiles, each within
        # shouting distance of training-strip scale.
        col = np.full((2600, 64), 230, dtype=np.uint8)
        col[100:2500:400, :] = 20  # sparse dark rows so a profile exists at all
        tiles = chunk_column(col, tile_target=900, window=200, min_tile=300)
        self.assertGreaterEqual(len(tiles), 2)
        self.assertEqual(sum(t.height for t in tiles), 2600)

    def test_rejects_bad_params(self):
        col = np.full((100, 10), 200, dtype=np.uint8)
        with self.assertRaises(ValueError):
            chunk_column(col, tile_target=0)
        with self.assertRaises(ValueError):
            chunk_column(col, tile_target=100, min_tile=200)  # min_tile > tile_target
        with self.assertRaises(ValueError):
            chunk_column(np.zeros((4, 4, 3), dtype=np.uint8))  # not 2-D

    def test_deterministic_given_same_input(self):
        col, *_ = _bars_column()
        t1 = chunk_column(col, tile_target=300, window=150, min_tile=100)
        t2 = chunk_column(col, tile_target=300, window=150, min_tile=100)
        self.assertEqual([(t.top, t.bottom) for t in t1], [(t.top, t.bottom) for t in t2])

    def test_tile_dataclass_height_property(self):
        t = Tile(top=10, bottom=110, image=np.zeros((100, 5), dtype=np.uint8))
        self.assertEqual(t.height, 100)


if __name__ == "__main__":
    unittest.main()
